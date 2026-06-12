import os
import sys
from dataclasses import dataclass
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from config import Config

try:
    from exp_utils import ALL_MACRO_FEATURES, prepare_data_split
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(__file__))
    from exp_utils import ALL_MACRO_FEATURES, prepare_data_split


@dataclass
class StressContext:
    threshold_scope: str
    quantile: float
    dates_trainval: pd.DatetimeIndex
    dates_test: pd.DatetimeIndex
    threshold_info: dict[str, float]
    macro_frame: pd.DataFrame
    stress_masks: pd.DataFrame


def _safe_log_diff(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return np.log(s).diff()


def _build_macro_daily(df: pd.DataFrame) -> pd.DataFrame:
    req = ["Global_VIX", "Global_US2Y", "Global_Oil"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing macro columns: {missing}")
    date_idx = pd.to_datetime(df["Date"])
    out = pd.DataFrame(index=date_idx)
    out["VIX"] = pd.Series(pd.to_numeric(df["Global_VIX"], errors="coerce").to_numpy(), index=date_idx)
    out["Delta_US2Y_abs"] = pd.Series(
        pd.to_numeric(df["Global_US2Y"], errors="coerce").diff().abs().to_numpy(),
        index=date_idx,
    )
    out["Oil_return_abs"] = pd.Series(_safe_log_diff(df["Global_Oil"]).abs().to_numpy(), index=date_idx)
    # Composite macro shock score
    changes = {}
    for c in ALL_MACRO_FEATURES:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if c in ("Global_VIX", "Global_US2Y", "Global_US10Y"):
            dx = pd.Series(x.diff().abs().to_numpy(), index=date_idx)
        else:
            dx = pd.Series(_safe_log_diff(x).abs().to_numpy(), index=date_idx)
        st = float(dx.std(ddof=0)) + 1e-12
        changes[c] = dx / st
    zdf = pd.DataFrame(changes, index=out.index)
    out["ShockScore"] = zdf.mean(axis=1)
    return out.sort_index()


def _target_abs_extreme_dates(data_bundle: dict, threshold_scope: str, q: float) -> tuple[pd.DatetimeIndex, float]:
    # uses test target dates from bundle and raw Y values to define date-level FX stress
    L = data_bundle["L"]
    n = data_bundle["n"]
    val_end = data_bundle["val_end"]
    feat_dates = pd.to_datetime(data_bundle["feature_dates"])
    y = data_bundle["Y"]  # [T, N]
    usd_idx = data_bundle["usd_idx"]
    mask = np.ones(data_bundle["n_ccy"], dtype=bool)
    mask[usd_idx] = False

    # map sample idx -> target row idx = idx + L
    all_idx = np.arange(n)
    target_rows = all_idx + L
    target_dates = pd.DatetimeIndex(feat_dates.iloc[target_rows])
    date_score = pd.Series(np.abs(y[target_rows][:, mask]).max(axis=1), index=target_dates).sort_index()
    test_dates = pd.DatetimeIndex(target_dates[val_end:])
    trainval_dates = pd.DatetimeIndex(target_dates[:val_end])
    scope_dates = trainval_dates if threshold_scope == "trainval" else test_dates
    th = float(date_score.reindex(scope_dates).dropna().quantile(q))
    stress_dates = pd.DatetimeIndex(date_score.index[date_score >= th]).intersection(test_dates)
    return stress_dates, th


def build_stress_context(
    config: Config,
    data_path: str,
    lookback: int = 20,
    quantile: float = 0.90,
    threshold_scope: str = "trainval",
) -> StressContext:
    if threshold_scope not in ("trainval", "test"):
        raise ValueError("threshold_scope must be one of ['trainval','test']")

    cfg = Config()
    cfg.file_path = data_path
    cfg.lookback = lookback
    data_bundle = prepare_data_split(cfg, split_mode="602020", data_path=data_path, macro_features=ALL_MACRO_FEATURES)

    L = data_bundle["L"]
    n = data_bundle["n"]
    val_end = data_bundle["val_end"]
    feat_dates = pd.to_datetime(data_bundle["feature_dates"])
    sample_target_dates = pd.DatetimeIndex(feat_dates.iloc[np.arange(n) + L])
    dates_trainval = pd.DatetimeIndex(sample_target_dates[:val_end])
    dates_test = pd.DatetimeIndex(sample_target_dates[val_end:])

    raw_df = pd.read_csv(data_path)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    macro_daily = _build_macro_daily(raw_df).dropna().sort_index()

    scope_dates = dates_trainval if threshold_scope == "trainval" else dates_test
    md_scope = macro_daily.reindex(scope_dates).dropna()
    vix_th = float(md_scope["VIX"].quantile(quantile))
    us2y_th = float(md_scope["Delta_US2Y_abs"].quantile(quantile))
    oil_th = float(md_scope["Oil_return_abs"].quantile(quantile))
    comp_th = float(md_scope["ShockScore"].quantile(quantile))
    vix_median = float(md_scope["VIX"].quantile(0.50))
    comp_median = float(md_scope["ShockScore"].quantile(0.50))

    fx_dates, fx_th = _target_abs_extreme_dates(data_bundle, threshold_scope, quantile)

    md_test = macro_daily.reindex(dates_test)
    is_vix = md_test["VIX"] >= vix_th
    is_us2y = md_test["Delta_US2Y_abs"] >= us2y_th
    is_oil = md_test["Oil_return_abs"] >= oil_th
    is_comp = md_test["ShockScore"] >= comp_th
    is_fx = pd.Series(md_test.index.isin(fx_dates), index=md_test.index)
    is_any_stress = is_vix | is_us2y | is_oil | is_comp | is_fx
    is_normal = (~is_any_stress) | ((md_test["VIX"] <= vix_median) & (md_test["ShockScore"] <= comp_median))

    stress_masks = pd.DataFrame(
        {
            "all_test": True,
            "normal": is_normal.fillna(False),
            "VIX_top10": is_vix.fillna(False),
            "FX_move_top10": is_fx.fillna(False),
            "US2Y_shock_top10": is_us2y.fillna(False),
            "Oil_shock_top10": is_oil.fillna(False),
            "composite_macro_shock_top10": is_comp.fillna(False),
        },
        index=md_test.index,
    )

    return StressContext(
        threshold_scope=threshold_scope,
        quantile=quantile,
        dates_trainval=dates_trainval,
        dates_test=dates_test,
        threshold_info={
            "VIX_top10": vix_th,
            "FX_move_top10": fx_th,
            "US2Y_shock_top10": us2y_th,
            "Oil_shock_top10": oil_th,
            "composite_macro_shock_top10": comp_th,
            "VIX_median": vix_median,
            "ShockScore_median": comp_median,
        },
        macro_frame=md_test,
        stress_masks=stress_masks,
    )


def apply_stress_mask(
    pred: np.ndarray,
    target: np.ndarray,
    dates: pd.DatetimeIndex,
    mask_dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sel = dates.isin(mask_dates)
    return pred[sel], target[sel], sel


def edge_turnover_rate(a_prev: np.ndarray, a_cur: np.ndarray, top_k: int = 3) -> float:
    n = a_prev.shape[0]
    top_k = max(1, min(top_k, n - 1))

    def _edge_set(a):
        s = set()
        for i in range(n):
            idx = np.argsort(a[i])[::-1]
            idx = [j for j in idx if j != i][:top_k]
            for j in idx:
                s.add((i, int(j)))
        return s

    e1 = _edge_set(a_prev)
    e2 = _edge_set(a_cur)
    den = len(e1 | e2)
    if den == 0:
        return 0.0
    return 1.0 - len(e1 & e2) / den
