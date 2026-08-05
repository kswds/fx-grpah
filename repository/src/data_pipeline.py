from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

try:
    from .config import UNIVERSE_PRESETS
except ImportError:
    from config import UNIVERSE_PRESETS


USD_PER_UNIT = {"EUR", "GBP", "AUD", "NZD"}
FOREIGN_PER_USD = {"JPY", "CAD", "CHF", "SEK", "NOK", "KRW"}

LOCAL_FEATURE_TEMPLATES = [
    "{ccy}_FX_ret_1d",
    "{ccy}_FX_mom_5",
    "{ccy}_FX_mom_20",
    "{ccy}_FX_vol_5",
    "{ccy}_FX_vol_20",
    "{ccy}_FX_zscore_20",
]
RATE_LOCAL_TEMPLATES = [
    "{ccy}_Yield10Y",
    "{ccy}_Yield10Y_change",
    "{ccy}_Yield10Y_minus_US10Y",
    "{ccy}_RealRate10Y",
    "{ccy}_Yield10Y_Available",
]
RATE_GLOBAL_COLUMNS = [
    "Global_US1Y",
    "Global_US2Y",
    "Global_US5Y",
    "Global_US10Y",
    "Global_US10Y_minus_US2Y",
    "Global_US2Y_change",
    "Global_US10Y_change",
]
EQUITY_TEMPLATES = [
    "{ccy}_Stock_ret",
    "{ccy}_Stock_mom_5",
    "{ccy}_Stock_mom_20",
    "{ccy}_Stock_vol_20",
    "{ccy}_Stock_ret_minus_SP500_ret",
]
COUNTRY_MACRO_TEMPLATES = [
    "{ccy}_GDP_growth",
    "{ccy}_CPIInfl",
    "{ccy}_PPIInfl",
    "{ccy}_GDP_growth_minus_USD",
    "{ccy}_CPIInfl_minus_USD",
    "{ccy}_PPIInfl_minus_USD",
]
GLOBAL_COLUMNS = [
    "Global_VIX",
    "Global_VIX_change",
    "Global_BroadDollar_ret",
    "Global_DXY_ret",
    "Global_SP500_ret",
    "Global_Oil_ret",
    "Global_Gold_ret",
    "Global_Copper_ret",
    "Global_US2Y_change",
    "Global_US10Y_change",
    "Global_US10Y_minus_US2Y",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fx_to_log(series: pd.Series, ccy: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    if ccy in USD_PER_UNIT:
        return np.log(x.where(x > 0))
    if ccy in FOREIGN_PER_USD:
        return -np.log(x.where(x > 0))
    if ccy == "USD":
        return pd.Series(0.0, index=x.index)
    raise ValueError(f"Unknown currency: {ccy}")


def safe_log_diff(series: pd.Series, periods: int = 1) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    return np.log(x.where(x > 0)).diff(periods)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window).mean()
    sig = series.rolling(window).std()
    return (series - mu) / (sig + 1e-8)


def normalize_date_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None).dt.normalize()
    return out


def merge_prefer_right(left: pd.DataFrame, right: pd.DataFrame, on: str = "Date") -> pd.DataFrame:
    merged = left.merge(right, on=on, how="inner", suffixes=("_fx", "_nonfx"))
    for col in [c for c in merged.columns if c.endswith("_fx")]:
        base = col[:-3]
        right_name = f"{base}_nonfx"
        if right_name in merged.columns:
            merged[base] = merged[right_name].combine_first(merged[col])
            merged = merged.drop(columns=[col, right_name])
        else:
            merged = merged.rename(columns={col: base})
    leftover = [c for c in merged.columns if c.endswith("_nonfx")]
    if leftover:
        merged = merged.rename(columns={c: c[:-6] for c in leftover})
    return merged


def resolve_currency_names(universe: str, custom_currencies: Optional[Sequence[str]]) -> List[str]:
    if custom_currencies:
        clean = [c.upper() for c in custom_currencies]
        names = ["USD"] + [c for c in clean if c != "USD"]
    else:
        names = UNIVERSE_PRESETS[universe]
    return names


def engineer_nonfx_fallbacks(df: pd.DataFrame, currency_names: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    if "Global_SP500" in out.columns and "Global_SP500_ret" not in out.columns:
        out["Global_SP500_ret"] = safe_log_diff(out["Global_SP500"])
    for col in ("Global_Oil", "Global_Gold", "Global_Copper", "Global_BroadDollar", "Global_DXY"):
        ret_col = f"{col}_ret" if col != "Global_BroadDollar" else "Global_BroadDollar_ret"
        if col in out.columns and ret_col not in out.columns:
            out[ret_col] = safe_log_diff(out[col])
    if "Global_VIX" in out.columns and "Global_VIX_change" not in out.columns:
        out["Global_VIX_change"] = pd.to_numeric(out["Global_VIX"], errors="coerce").diff()
    if {"Global_US10Y", "Global_US2Y"}.issubset(out.columns):
        out["Global_US10Y_minus_US2Y"] = pd.to_numeric(out["Global_US10Y"], errors="coerce") - pd.to_numeric(out["Global_US2Y"], errors="coerce")
        out["Global_US10Y_change"] = pd.to_numeric(out["Global_US10Y"], errors="coerce").diff()
        out["Global_US2Y_change"] = pd.to_numeric(out["Global_US2Y"], errors="coerce").diff()
    for ccy in currency_names:
        stock_col = f"{ccy}_Stock"
        if stock_col in out.columns:
            out[f"{ccy}_Stock_ret"] = safe_log_diff(out[stock_col])
            out[f"{ccy}_Stock_mom_5"] = safe_log_diff(out[stock_col], 5)
            out[f"{ccy}_Stock_mom_20"] = safe_log_diff(out[stock_col], 20)
            out[f"{ccy}_Stock_vol_20"] = safe_log_diff(out[stock_col]).rolling(20).std()
            if "Global_SP500_ret" in out.columns:
                out[f"{ccy}_Stock_ret_minus_SP500_ret"] = out[f"{ccy}_Stock_ret"] - out["Global_SP500_ret"]
        y_col = f"{ccy}_Yield10Y"
        if y_col in out.columns:
            out[f"{ccy}_Yield10Y_change"] = pd.to_numeric(out[y_col], errors="coerce").diff()
            if "Global_US10Y" in out.columns:
                out[f"{ccy}_Yield10Y_minus_US10Y"] = pd.to_numeric(out[y_col], errors="coerce") - pd.to_numeric(out["Global_US10Y"], errors="coerce")
            out[f"{ccy}_Yield10Y_Available"] = pd.to_numeric(out[y_col], errors="coerce").notna().astype(float)
        if f"{ccy}_GDP_growth" not in out.columns:
            out[f"{ccy}_GDP_growth"] = 0.0
        if f"{ccy}_PPIInfl" not in out.columns:
            out[f"{ccy}_PPIInfl"] = 0.0
    if "USD_CPIInfl" in out.columns:
        for ccy in currency_names:
            for feat in ("GDP_growth", "CPIInfl", "PPIInfl"):
                col = f"{ccy}_{feat}"
                usd_col = f"USD_{feat}"
                diff_col = f"{ccy}_{feat}_minus_USD"
                if col in out.columns and usd_col in out.columns:
                    out[diff_col] = pd.to_numeric(out[col], errors="coerce") - pd.to_numeric(out[usd_col], errors="coerce")
                elif diff_col not in out.columns:
                    out[diff_col] = 0.0
    return out


def engineer_fx_local_features(df: pd.DataFrame, currency_names: Sequence[str]) -> Tuple[pd.DataFrame, List[dict]]:
    out = df.copy()
    rows: List[dict] = []
    for ccy in currency_names:
        if ccy == "USD":
            out[f"{ccy}_FX_ret_1d"] = 0.0
            out[f"{ccy}_FX_mom_5"] = 0.0
            out[f"{ccy}_FX_mom_20"] = 0.0
            out[f"{ccy}_FX_vol_5"] = 0.0
            out[f"{ccy}_FX_vol_20"] = 0.0
            out[f"{ccy}_FX_zscore_20"] = 0.0
            continue
        fx_col = f"{ccy}_FX"
        if fx_col not in out.columns:
            continue
        log_fx = fx_to_log(out[fx_col], ccy)
        out[f"{ccy}_FX_ret_1d"] = log_fx.diff()
        out[f"{ccy}_FX_mom_5"] = log_fx.diff(5)
        out[f"{ccy}_FX_mom_20"] = log_fx.diff(20)
        out[f"{ccy}_FX_vol_5"] = log_fx.diff().rolling(5).std()
        out[f"{ccy}_FX_vol_20"] = log_fx.diff().rolling(20).std()
        out[f"{ccy}_FX_zscore_20"] = rolling_zscore(log_fx, 20)
        for col, transform in [
            (f"{ccy}_FX_ret_1d", "diff1_log_local_appreciation"),
            (f"{ccy}_FX_mom_5", "diff5_log_local_appreciation"),
            (f"{ccy}_FX_mom_20", "diff20_log_local_appreciation"),
            (f"{ccy}_FX_vol_5", "rolling_std_5"),
            (f"{ccy}_FX_vol_20", "rolling_std_20"),
            (f"{ccy}_FX_zscore_20", "rolling_zscore_20"),
        ]:
            rows.append({"component": "local", "currency": ccy, "used_column": col, "source_column": fx_col, "transformation": transform, "is_global": False})
    return out, rows


def build_targets(df: pd.DataFrame, currency_names: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    target_cols = []
    for ccy in currency_names:
        col = f"TargetRet_{ccy}"
        if ccy == "USD":
            out[col] = 0.0
        else:
            pref = f"TARGET_{ccy}_FX_RET_FWD1"
            if pref in out.columns:
                out[col] = pd.to_numeric(out[pref], errors="coerce")
            else:
                out[col] = fx_to_log(out[f"Target_{ccy}_FX"], ccy) - fx_to_log(out[f"{ccy}_FX"], ccy)
        target_cols.append(col)
    return out, target_cols


def build_group_arrays(df: pd.DataFrame, currency_names: Sequence[str], templates: List[str], broadcast_global: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    n = len(currency_names)
    t = len(df)
    d = len(templates) + len(broadcast_global or [])
    arr = np.zeros((t, n, max(1, d)), dtype=np.float32)
    feature_names = []
    for j, template in enumerate(templates):
        feature_names.append(template.replace("{ccy}_", ""))
        for i, ccy in enumerate(currency_names):
            col = template.format(ccy=ccy)
            if col in df.columns:
                arr[:, i, j] = pd.to_numeric(df[col], errors="coerce").astype(float).values
    offset = len(templates)
    for g_idx, col in enumerate(broadcast_global or []):
        feature_names.append(f"broadcast::{col}")
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").astype(float).values
            arr[:, :, offset + g_idx] = vals[:, None]
    return arr, feature_names


def build_global_array(df: pd.DataFrame, columns: List[str]) -> Tuple[np.ndarray, List[str]]:
    t = len(df)
    arr = np.zeros((t, max(1, len(columns))), dtype=np.float32)
    for j, col in enumerate(columns):
        if col in df.columns:
            arr[:, j] = pd.to_numeric(df[col], errors="coerce").astype(float).values
    return arr, list(columns)


def normalize_train_only(arr: np.ndarray, train_end: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = arr[:train_end]
    valid = np.isfinite(train)
    if arr.ndim == 3:
        count = valid.sum(axis=(0, 1), keepdims=True)
        safe_train = np.where(valid, train, 0.0)
        mean = safe_train.sum(axis=(0, 1), keepdims=True) / np.maximum(count, 1)
        sq = np.where(valid, (train - mean) ** 2, 0.0)
        std = np.sqrt(sq.sum(axis=(0, 1), keepdims=True) / np.maximum(count, 1)) + 1e-6
    else:
        count = valid.sum(axis=0, keepdims=True)
        safe_train = np.where(valid, train, 0.0)
        mean = safe_train.sum(axis=0, keepdims=True) / np.maximum(count, 1)
        sq = np.where(valid, (train - mean) ** 2, 0.0)
        std = np.sqrt(sq.sum(axis=0, keepdims=True) / np.maximum(count, 1)) + 1e-6
    out = np.nan_to_num((arr - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out, mean.squeeze(), std.squeeze()


def normalize_targets_train_only(y: np.ndarray, train_end: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = y[:train_end]
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0) + 1e-6
    out = np.nan_to_num((y - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out, mean.astype(np.float32), std.astype(np.float32)


@dataclass
class PreparedData:
    merged: pd.DataFrame
    currency_names: List[str]
    x_local: np.ndarray
    x_rate: np.ndarray
    x_equity: np.ndarray
    x_countrymacro: np.ndarray
    x_global: np.ndarray
    y_raw: np.ndarray
    y_norm: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    q80_abs_y_train: float
    dims: dict


def add_regime_onehot_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    created: List[str] = []
    proxy_cols = [c for c in out.columns if c.startswith("TARGET_") and c.endswith("_FX_RET_FWD1")]
    if proxy_cols:
        abs_target_proxy = out[proxy_cols].abs().mean(axis=1)
    else:
        fx_ret_cols = [c for c in out.columns if c.endswith("_FX_ret_1d")]
        abs_target_proxy = out[fx_ret_cols].abs().mean(axis=1) if fx_ret_cols else pd.Series(0.0, index=out.index)
    roll_abs = abs_target_proxy.rolling(20, min_periods=5).mean()
    q1 = roll_abs.quantile(0.33)
    q2 = roll_abs.quantile(0.67)
    out["Regime_LowVol"] = (roll_abs <= q1).astype(float)
    out["Regime_MidVol"] = ((roll_abs > q1) & (roll_abs < q2)).astype(float)
    out["Regime_HighVol"] = (roll_abs >= q2).astype(float)
    out["Regime_RiskOff"] = (
        (pd.to_numeric(out.get("Global_VIX_change", 0.0), errors="coerce") >= pd.to_numeric(out.get("Global_VIX_change", 0.0), errors="coerce").quantile(0.8))
        | (pd.to_numeric(out.get("Global_SP500_ret", 0.0), errors="coerce") <= pd.to_numeric(out.get("Global_SP500_ret", 0.0), errors="coerce").quantile(0.2))
    ).astype(float)
    dollar_shock = pd.Series(False, index=out.index)
    for col in ("Global_BroadDollar_ret", "Global_DXY_ret"):
        if col in out.columns:
            x = pd.to_numeric(out[col], errors="coerce").abs()
            dollar_shock |= x >= x.quantile(0.8)
    out["Regime_DollarShock"] = dollar_shock.astype(float)
    commodity_shock = pd.Series(False, index=out.index)
    for col in ("Global_Oil_ret", "Global_Copper_ret"):
        if col in out.columns:
            x = pd.to_numeric(out[col], errors="coerce").abs()
            commodity_shock |= x >= x.quantile(0.8)
    out["Regime_CommodityShock"] = commodity_shock.astype(float)
    created.extend(["Regime_LowVol", "Regime_MidVol", "Regime_HighVol", "Regime_RiskOff", "Regime_DollarShock", "Regime_CommodityShock"])
    return out, created


def prepare_data(
    fx_data_path: str,
    nonfx_data_path: str,
    currency_names: Sequence[str],
    lookback: int,
    split: Sequence[float],
    include_regime_onehot: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
) -> PreparedData:
    fx_df = normalize_date_col(pd.read_csv(fx_data_path))
    nonfx_df = normalize_date_col(pd.read_csv(nonfx_data_path))
    merged = merge_prefer_right(fx_df, nonfx_df, on="Date").sort_values("Date").reset_index(drop=True)
    if start_date:
        merged = merged.loc[merged["Date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    if end_date:
        merged = merged.loc[merged["Date"] <= pd.Timestamp(end_date)].reset_index(drop=True)
    merged = engineer_nonfx_fallbacks(merged, currency_names)
    merged, target_cols = build_targets(merged, currency_names)
    merged, _ = engineer_fx_local_features(merged, currency_names)
    extra_global_columns: List[str] = []
    if include_regime_onehot:
        merged, extra_global_columns = add_regime_onehot_features(merged)
    required = [f"TargetRet_{ccy}" for ccy in currency_names if ccy != "USD"]
    merged = merged.loc[merged[required].notna().all(axis=1)].reset_index(drop=True)
    y_raw = merged[target_cols].astype(float).values.astype(np.float32)
    x_local, _ = build_group_arrays(merged, currency_names, LOCAL_FEATURE_TEMPLATES)
    x_rate, _ = build_group_arrays(merged, currency_names, RATE_LOCAL_TEMPLATES, RATE_GLOBAL_COLUMNS)
    x_equity, _ = build_group_arrays(merged, currency_names, EQUITY_TEMPLATES)
    x_country, _ = build_group_arrays(merged, currency_names, COUNTRY_MACRO_TEMPLATES)
    x_global, _ = build_global_array(merged, GLOBAL_COLUMNS + extra_global_columns)
    n_total = len(merged)
    train_end = int(n_total * split[0])
    x_local, _, _ = normalize_train_only(x_local, train_end)
    x_rate, _, _ = normalize_train_only(x_rate, train_end)
    x_equity, _, _ = normalize_train_only(x_equity, train_end)
    x_country, _, _ = normalize_train_only(x_country, train_end)
    x_global, _, _ = normalize_train_only(x_global, train_end)
    y_norm, y_mean, y_std = normalize_targets_train_only(y_raw, train_end)
    non_usd_idx = np.arange(1, len(currency_names))
    q80_abs_y_train = float(np.nanquantile(np.abs(y_norm[:train_end, non_usd_idx]), 0.8)) if len(non_usd_idx) else 1.0
    return PreparedData(
        merged=merged,
        currency_names=list(currency_names),
        x_local=x_local,
        x_rate=x_rate,
        x_equity=x_equity,
        x_countrymacro=x_country,
        x_global=x_global,
        y_raw=y_raw,
        y_norm=y_norm,
        y_mean=y_mean,
        y_std=y_std,
        q80_abs_y_train=max(q80_abs_y_train, 1e-6),
        dims={"local": x_local.shape[-1], "rate": x_rate.shape[-1], "equity": x_equity.shape[-1], "countrymacro": x_country.shape[-1], "global": x_global.shape[-1]},
    )


class MultiBlockSequenceDataset(Dataset):
    def __init__(self, prepared: PreparedData, lookback: int):
        self.p = prepared
        self.lookback = int(lookback)

    def __len__(self) -> int:
        return len(self.p.merged) - self.lookback

    def __getitem__(self, idx: int):
        end = idx + self.lookback
        batch = {
            "x_local": torch.tensor(self.p.x_local[idx:end], dtype=torch.float32),
            "x_rate": torch.tensor(self.p.x_rate[idx:end], dtype=torch.float32),
            "x_equity": torch.tensor(self.p.x_equity[idx:end], dtype=torch.float32),
            "x_countrymacro": torch.tensor(self.p.x_countrymacro[idx:end], dtype=torch.float32),
            "x_global": torch.tensor(self.p.x_global[idx:end], dtype=torch.float32),
        }
        target = {
            "y_norm": torch.tensor(self.p.y_norm[end], dtype=torch.float32),
            "y_raw": torch.tensor(self.p.y_raw[end], dtype=torch.float32),
        }
        meta = {
            "input_end_date": str(self.p.merged["Date"].iloc[end - 1].date()),
            "target_date": str(self.p.merged["Date"].iloc[end].date()),
        }
        return batch, target, meta


def collate_batch(items):
    batches, targets, metas = zip(*items)
    batch_out = {k: torch.stack([b[k] for b in batches], dim=0) for k in batches[0]}
    target_out = {k: torch.stack([t[k] for t in targets], dim=0) for k in targets[0]}
    return batch_out, target_out, list(metas)


def create_splits(dataset: Dataset, split: Sequence[float]) -> Tuple[Subset, Subset, Subset]:
    n = len(dataset)
    train_end = int(n * split[0])
    val_end = train_end + int(n * split[1])
    return Subset(dataset, range(0, train_end)), Subset(dataset, range(train_end, val_end)), Subset(dataset, range(val_end, n))


def inverse_transform(pred_norm: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return pred_norm * y_std[None, :] + y_mean[None, :]
