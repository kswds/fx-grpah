import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from config import Config
from models import create_model

try:
    from exp_utils import ALL_MACRO_FEATURES, build_windows, prepare_data_split
    from stress_utils import build_stress_context, edge_turnover_rate
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(__file__))
    from exp_utils import ALL_MACRO_FEATURES, build_windows, prepare_data_split
    from stress_utils import build_stress_context, edge_turnover_rate


LOGGER = logging.getLogger("exp_graph_adaptation_analysis")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Dynamic graph adaptation analysis for MACRO-HyGraph (Ours).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default=str(ROOT / "results" / "model_ablation" / "checkpoints" / "Ours_L20_seed42.pt"))
    p.add_argument("--config", default=str(ROOT / "results" / "model_ablation" / "configs" / "Ours_L20_seed42.json"))
    p.add_argument("--data-path", default=str(ROOT / "data" / "factor_daily_legacy.csv"))
    p.add_argument("--output-dir", default=str(ROOT / "results" / "stress_regime_analysis"))
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--stress-quantile", type=float, default=0.90)
    p.add_argument("--threshold-scope", default="trainval", choices=["trainval", "test"])
    p.add_argument("--top-k-turnover", type=int, default=3)
    p.add_argument("--story-top-pct", type=float, default=0.15)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _load_cfg(config_json: str, data_path: str, lookback: int) -> Config:
    cfg = Config()
    cfg.file_path = data_path
    cfg.lookback = lookback
    p = Path(config_json)
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        base = d.get("base_config", {})
        run = d.get("run_config", {})
        for k in ("hidden", "hybrid_hidden", "top_k", "batch_size", "epochs", "lr", "dropout"):
            if k in run:
                setattr(cfg, k, run[k])
            elif k in base:
                setattr(cfg, k, base[k])
    return cfg


def _collect_adjacency(model, Xl, Xm):
    A_list = []
    for i in range(len(Xl)):
        xl = np.expand_dims(Xl[i], axis=0)
        xm = np.expand_dims(Xm[i], axis=0)
        import torch

        with torch.no_grad():
            _ = model(
                torch.tensor(xl, dtype=torch.float32),
                torch.tensor(xm, dtype=torch.float32),
                None,
            )
        a = getattr(model, "last_adj", None)
        if a is None:
            raise RuntimeError("Model does not expose last_adj.")
        a_np = a.detach().cpu().numpy()
        if a_np.ndim == 3:
            a_np = a_np[0]
        A_list.append(a_np.astype(np.float32))
    return np.stack(A_list, axis=0)


def _graph_change_df(A: np.ndarray, dates: pd.DatetimeIndex, stress_ctx, top_k: int) -> pd.DataFrame:
    T, N, _ = A.shape
    D_F = np.zeros(T, dtype=np.float64)
    D_MA = np.zeros(T, dtype=np.float64)
    turnover = np.zeros(T, dtype=np.float64)
    entropy = np.zeros(T, dtype=np.float64)
    eps = 1e-12
    for t in range(T):
        if t > 0:
            d = A[t] - A[t - 1]
            D_F[t] = float(np.linalg.norm(d, ord="fro"))
            D_MA[t] = float(np.mean(np.abs(d)))
            turnover[t] = float(edge_turnover_rate(A[t - 1], A[t], top_k=top_k))
        row_ent = -np.sum(A[t] * np.log(A[t] + eps), axis=1)
        entropy[t] = float(np.mean(row_ent))
    mean_df = float(np.mean(D_F)) + 1e-12
    dyn = D_F / mean_df
    md = stress_ctx.macro_frame.reindex(dates)
    masks = stress_ctx.stress_masks.reindex(dates).fillna(False)
    out = pd.DataFrame(
        {
            "date": dates,
            "D_F": D_F,
            "D_MA": D_MA,
            "edge_turnover": turnover,
            "graph_entropy": entropy,
            "dynamic_intensity": dyn,
            "VIX": md["VIX"].values,
            "Delta_US2Y_abs": md["Delta_US2Y_abs"].values,
            "Oil_return_abs": md["Oil_return_abs"].values,
            "ShockScore": md["ShockScore"].values,
            "is_vix_stress": masks["VIX_top10"].values,
            "is_us2y_shock": masks["US2Y_shock_top10"].values,
            "is_oil_shock": masks["Oil_shock_top10"].values,
            "is_composite_stress": masks["composite_macro_shock_top10"].values,
            "is_normal": masks["normal"].values,
        }
    )
    return out


def _regime_stats(ts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    regimes = {
        "normal": ts["is_normal"].astype(bool),
        "VIX_top10": ts["is_vix_stress"].astype(bool),
        "US2Y_shock_top10": ts["is_us2y_shock"].astype(bool),
        "Oil_shock_top10": ts["is_oil_shock"].astype(bool),
        "composite_macro_shock_top10": ts["is_composite_stress"].astype(bool),
    }
    metrics = ["D_F", "D_MA", "edge_turnover", "graph_entropy", "dynamic_intensity"]
    rows = []
    for name, m in regimes.items():
        sub = ts[m]
        row = {"regime": name, "n_obs": int(len(sub))}
        for c in metrics:
            row[f"{c}_mean"] = float(sub[c].mean())
            row[f"{c}_std"] = float(sub[c].std(ddof=0))
        rows.append(row)
    by_regime = pd.DataFrame(rows)

    normal = by_regime[by_regime["regime"] == "normal"].iloc[0]
    ratio_rows = []
    for _, r in by_regime.iterrows():
        if r["regime"] == "normal":
            continue
        ratio_rows.append(
            {
                "regime": r["regime"],
                "D_F_stress_over_normal": float(r["D_F_mean"] / (normal["D_F_mean"] + 1e-12)),
                "Turnover_stress_over_normal": float(r["edge_turnover_mean"] / (normal["edge_turnover_mean"] + 1e-12)),
                "Entropy_minus_normal": float(r["graph_entropy_mean"] - normal["graph_entropy_mean"]),
            }
        )
    return by_regime, pd.DataFrame(ratio_rows)


def _compute_extended_regime_masks(
    data_bundle: dict,
    dates_test: pd.DatetimeIndex,
    y_test: np.ndarray,
    data_path: str,
    threshold_scope: str = "trainval",
) -> pd.DataFrame:
    """
    Build additional stress-regime masks requested for extension.
    Thresholds are computed on train+val or test scope and applied to test dates.
    """
    raw = pd.read_csv(data_path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values("Date").set_index("Date")

    feat_dates = pd.DatetimeIndex(pd.to_datetime(data_bundle["feature_dates"]))
    L = int(data_bundle["L"])
    n = int(data_bundle["n"])
    val_end = int(data_bundle["val_end"])
    sample_target_dates = pd.DatetimeIndex(feat_dates[np.arange(n) + L])
    trainval_dates = pd.DatetimeIndex(sample_target_dates[:val_end])
    scope_dates = trainval_dates if threshold_scope == "trainval" else dates_test

    usd_idx = int(data_bundle["usd_idx"])
    ccy_mask = np.ones(int(data_bundle["n_ccy"]), dtype=bool)
    ccy_mask[usd_idx] = False

    # Build full-sample FX-derived series at sample target dates so trainval thresholds are valid.
    y_full = np.asarray(data_bundle["Y"])[L : L + n]
    y_full_nonusd = y_full[:, ccy_mask]
    s_dates_full = sample_target_dates
    s_dollar_full = pd.Series(np.mean(y_full_nonusd, axis=1), index=s_dates_full)
    s_fxdisp_full = pd.Series(np.std(y_full_nonusd, axis=1), index=s_dates_full)
    s_fxabs_full = pd.Series(np.mean(np.abs(y_full_nonusd), axis=1), index=s_dates_full)
    s_dollar = s_dollar_full.reindex(dates_test)
    s_fxdisp = s_fxdisp_full.reindex(dates_test)
    s_fxabs = s_fxabs_full.reindex(dates_test)

    # Macro series
    sp500 = raw["Global_SP500"] if "Global_SP500" in raw.columns else pd.Series(index=raw.index, dtype=float)
    us2y = raw["Global_US2Y"] if "Global_US2Y" in raw.columns else pd.Series(index=raw.index, dtype=float)
    us10y = raw["Global_US10Y"] if "Global_US10Y" in raw.columns else pd.Series(index=raw.index, dtype=float)
    oil = raw["Global_Oil"] if "Global_Oil" in raw.columns else pd.Series(index=raw.index, dtype=float)
    copper = raw["Global_Copper"] if "Global_Copper" in raw.columns else pd.Series(index=raw.index, dtype=float)
    iron = raw["Global_IronOre"] if "Global_IronOre" in raw.columns else pd.Series(index=raw.index, dtype=float)

    def _safe_return(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        pos = s[s > 0]
        if len(pos) >= max(20, int(0.7 * s.notna().sum())):
            return np.log(s.where(s > 0)).diff()
        return s.diff()

    s_sp500_ret_full = _safe_return(sp500)
    s_du2_full = pd.to_numeric(us2y, errors="coerce").diff()
    s_du10_full = pd.to_numeric(us10y, errors="coerce").diff()
    s_sp500_ret = s_sp500_ret_full.reindex(dates_test)
    s_du2 = s_du2_full.reindex(dates_test)
    s_du10 = s_du10_full.reindex(dates_test)

    # Commodity basket shock
    comm_parts = []
    for s in [oil, copper, iron]:
        if len(s) == 0:
            continue
        r = _safe_return(s)
        std = float(np.nanstd(np.abs(r.reindex(scope_dates).values))) + 1e-12
        z = np.abs(r) / std
        comm_parts.append(z)
    if comm_parts:
        comm_df = pd.concat(comm_parts, axis=1)
        s_comm_full = comm_df.mean(axis=1)
        s_comm = s_comm_full.reindex(dates_test)
    else:
        s_comm_full = pd.Series(index=raw.index, dtype=float)
        s_comm = pd.Series(index=dates_test, dtype=float)

    # Scope helper
    def _q(series: pd.Series, q: float):
        return float(series.reindex(scope_dates).dropna().quantile(q))

    # Thresholds
    th_sp500_lo = _q(s_sp500_ret_full, 0.10) if s_sp500_ret_full.notna().any() else np.nan
    th_dollar = _q(np.abs(s_dollar_full), 0.90) if s_dollar_full.notna().any() else np.nan
    th_fxdisp = _q(s_fxdisp_full, 0.90) if s_fxdisp_full.notna().any() else np.nan
    th_fxabs = _q(s_fxabs_full, 0.90) if s_fxabs_full.notna().any() else np.nan
    th_comm = _q(s_comm_full, 0.90) if s_comm_full.notna().any() else np.nan
    th_u2_up = _q(s_du2_full, 0.90) if s_du2_full.notna().any() else np.nan
    th_u2_dn = _q(s_du2_full, 0.10) if s_du2_full.notna().any() else np.nan
    th_u10_up = _q(s_du10_full, 0.90) if s_du10_full.notna().any() else np.nan
    th_u10_dn = _q(s_du10_full, 0.10) if s_du10_full.notna().any() else np.nan
    s_curve_full = (s_du10_full - s_du2_full).abs() if s_du10_full.notna().any() and s_du2_full.notna().any() else pd.Series(index=raw.index, dtype=float)
    s_curve = s_curve_full.reindex(dates_test)
    th_curve = _q(s_curve_full, 0.90) if s_curve_full.notna().any() else np.nan

    out = pd.DataFrame(index=dates_test)
    out["SP500_drawdown_shock"] = s_sp500_ret <= th_sp500_lo
    out["Dollar_factor_shock"] = np.abs(s_dollar) >= th_dollar
    out["FX_dispersion_shock"] = s_fxdisp >= th_fxdisp
    out["FX_abs_move_shock"] = s_fxabs >= th_fxabs
    out["Commodity_basket_shock"] = s_comm >= th_comm
    out["US2Y_up_shock"] = s_du2 >= th_u2_up
    out["US2Y_down_shock"] = s_du2 <= th_u2_dn
    if s_du10.notna().any():
        out["US10Y_up_shock"] = s_du10 >= th_u10_up
        out["US10Y_down_shock"] = s_du10 <= th_u10_dn
    else:
        out["US10Y_up_shock"] = False
        out["US10Y_down_shock"] = False
    out["Curve_shock"] = s_curve >= th_curve if s_curve.notna().any() else False
    return out.fillna(False).astype(bool)


def _extended_regime_change_table(ts: pd.DataFrame, ext_masks: pd.DataFrame, out_graph: Path) -> pd.DataFrame:
    # keep original masks too
    regimes = {
        "normal": ts["is_normal"].astype(bool).values,
        "VIX_top10": ts["is_vix_stress"].astype(bool).values,
        "US2Y_shock_top10": ts["is_us2y_shock"].astype(bool).values,
        "Oil_shock_top10": ts["is_oil_shock"].astype(bool).values,
        "composite_macro_shock_top10": ts["is_composite_stress"].astype(bool).values,
    }
    for c in ext_masks.columns:
        regimes[c] = ext_masks[c].reindex(pd.to_datetime(ts["date"])).fillna(False).values.astype(bool)

    n_mask = regimes["normal"]
    n_df_mean = float(np.nanmean(ts.loc[n_mask, "D_F"])) if np.any(n_mask) else np.nan
    n_dma_mean = float(np.nanmean(ts.loc[n_mask, "D_MA"])) if np.any(n_mask) else np.nan
    n_to_mean = float(np.nanmean(ts.loc[n_mask, "edge_turnover"])) if np.any(n_mask) else np.nan

    rows = []
    for reg, mask in regimes.items():
        sub = ts.loc[mask]
        if len(sub) == 0:
            rows.append(
                {
                    "regime": reg,
                    "n_obs": 0,
                    "D_F_mean": np.nan,
                    "D_F_median": np.nan,
                    "D_MA_mean": np.nan,
                    "D_MA_median": np.nan,
                    "edge_turnover_mean": np.nan,
                    "edge_turnover_median": np.nan,
                    "D_F_ratio_over_normal_mean": np.nan,
                    "D_MA_ratio_over_normal_mean": np.nan,
                    "edge_turnover_ratio_over_normal_mean": np.nan,
                }
            )
            continue
        d_f_mean = float(np.nanmean(sub["D_F"]))
        d_ma_mean = float(np.nanmean(sub["D_MA"]))
        to_mean = float(np.nanmean(sub["edge_turnover"]))
        rows.append(
            {
                "regime": reg,
                "n_obs": int(len(sub)),
                "D_F_mean": d_f_mean,
                "D_F_median": float(np.nanmedian(sub["D_F"])),
                "D_MA_mean": d_ma_mean,
                "D_MA_median": float(np.nanmedian(sub["D_MA"])),
                "edge_turnover_mean": to_mean,
                "edge_turnover_median": float(np.nanmedian(sub["edge_turnover"])),
                "D_F_ratio_over_normal_mean": float(d_f_mean / (n_df_mean + 1e-12)) if np.isfinite(n_df_mean) else np.nan,
                "D_MA_ratio_over_normal_mean": float(d_ma_mean / (n_dma_mean + 1e-12)) if np.isfinite(n_dma_mean) else np.nan,
                "edge_turnover_ratio_over_normal_mean": float(to_mean / (n_to_mean + 1e-12)) if np.isfinite(n_to_mean) else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)
    out.to_csv(out_graph / "extended_graph_change_by_regime.csv", index=False)
    return out


def _plot_extended_ratio_bars(ext_df: pd.DataFrame, fig_dir: Path) -> None:
    d = ext_df[ext_df["regime"] != "normal"].copy()
    if d.empty:
        return
    regs = list(d["regime"])
    x = np.arange(len(regs))
    w = 0.25
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w, d["D_F_ratio_over_normal_mean"].values, width=w, label="D_F ratio")
    ax.bar(x, d["D_MA_ratio_over_normal_mean"].values, width=w, label="D_MA ratio")
    ax.bar(x + w, d["edge_turnover_ratio_over_normal_mean"].values, width=w, label="Turnover ratio")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(regs, rotation=25, ha="right")
    ax.set_ylabel("Stress / Normal ratio")
    ax.set_title("Extended Graph-Change Ratio by Regime")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "extended_graph_change_ratio_by_regime.png", dpi=150)
    plt.close(fig)


def _regime_mean_median_ratio_table(ts: pd.DataFrame, out_graph: Path) -> pd.DataFrame:
    regimes = {
        "Normal": ts["is_normal"].astype(bool),
        "VIX": ts["is_vix_stress"].astype(bool),
        "US2Y": ts["is_us2y_shock"].astype(bool),
        "Oil": ts["is_oil_shock"].astype(bool),
        "Composite": ts["is_composite_stress"].astype(bool),
    }
    metrics = ["D_F", "D_MA", "edge_turnover"]
    rows = []
    for m in metrics:
        row = {"Metric": m}
        normal_vals = ts.loc[regimes["Normal"], m].values.astype(float)
        n_mean = float(np.nanmean(normal_vals)) if len(normal_vals) else np.nan
        n_med = float(np.nanmedian(normal_vals)) if len(normal_vals) else np.nan
        row["Normal_mean"] = n_mean
        row["Normal_median"] = n_med
        for r in ["VIX", "US2Y", "Oil", "Composite"]:
            vals = ts.loc[regimes[r], m].values.astype(float)
            r_mean = float(np.nanmean(vals)) if len(vals) else np.nan
            r_med = float(np.nanmedian(vals)) if len(vals) else np.nan
            row[f"{r}_mean"] = r_mean
            row[f"{r}_median"] = r_med
            row[f"{r}_over_Normal_mean_ratio"] = float(r_mean / (n_mean + 1e-12)) if np.isfinite(r_mean) and np.isfinite(n_mean) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_graph / "graph_change_regime_mean_median_ratio.csv", index=False)
    return out


def _permutation_pvalue(x: np.ndarray, y: np.ndarray, n_perm: int = 2000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    obs = abs(np.mean(x) - np.mean(y))
    z = np.concatenate([x, y])
    nx = len(x)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(z)
        d = abs(np.mean(z[:nx]) - np.mean(z[nx:]))
        if d >= obs:
            cnt += 1
    return float((cnt + 1) / (n_perm + 1))


def _bootstrap_ci_mean_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs[i] = np.mean(xb) - np.mean(yb)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi)


def _regime_stat_tests(ts: pd.DataFrame, out_graph: Path) -> pd.DataFrame:
    stress = {
        "VIX": ts["is_vix_stress"].astype(bool),
        "US2Y": ts["is_us2y_shock"].astype(bool),
        "Oil": ts["is_oil_shock"].astype(bool),
        "Composite": ts["is_composite_stress"].astype(bool),
    }
    normal_mask = ts["is_normal"].astype(bool)
    metrics = ["D_F", "D_MA", "edge_turnover"]
    rows = []
    for m in metrics:
        normal_vals = ts.loc[normal_mask, m].values.astype(float)
        for reg, mask in stress.items():
            stress_vals = ts.loc[mask, m].values.astype(float)
            if len(normal_vals) == 0 or len(stress_vals) == 0:
                rows.append(
                    {
                        "metric": m,
                        "regime": reg,
                        "n_normal": int(len(normal_vals)),
                        "n_stress": int(len(stress_vals)),
                        "normal_mean": np.nan,
                        "stress_mean": np.nan,
                        "mean_ratio_stress_over_normal": np.nan,
                        "mannwhitney_u_pvalue": np.nan,
                        "permutation_pvalue": np.nan,
                        "bootstrap_ci_diff_low": np.nan,
                        "bootstrap_ci_diff_high": np.nan,
                    }
                )
                continue
            n_mean = float(np.nanmean(normal_vals))
            s_mean = float(np.nanmean(stress_vals))
            ratio = float(s_mean / (n_mean + 1e-12))
            try:
                mw = mannwhitneyu(stress_vals, normal_vals, alternative="two-sided")
                p_mw = float(mw.pvalue)
            except Exception:
                p_mw = np.nan
            p_perm = _permutation_pvalue(stress_vals, normal_vals, n_perm=2000, seed=42)
            ci_lo, ci_hi = _bootstrap_ci_mean_diff(stress_vals, normal_vals, n_boot=2000, seed=42)
            rows.append(
                {
                    "metric": m,
                    "regime": reg,
                    "n_normal": int(len(normal_vals)),
                    "n_stress": int(len(stress_vals)),
                    "normal_mean": n_mean,
                    "stress_mean": s_mean,
                    "mean_ratio_stress_over_normal": ratio,
                    "mannwhitney_u_pvalue": p_mw,
                    "permutation_pvalue": p_perm,
                    "bootstrap_ci_diff_low": ci_lo,
                    "bootstrap_ci_diff_high": ci_hi,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_graph / "graph_change_regime_stat_tests.csv", index=False)
    return out


def _efficiency(ts: pd.DataFrame) -> pd.DataFrame:
    d = ts["D_F"].values
    q75, q90, q95, q99 = np.quantile(d, [0.75, 0.90, 0.95, 0.99])
    tau = q90
    is_update = d >= tau
    is_stress = ts[["is_vix_stress", "is_us2y_shock", "is_oil_shock", "is_composite_stress"]].any(axis=1).values
    is_normal = ts["is_normal"].values
    out = pd.DataFrame(
        [
            {
                "mean_D_F": float(np.mean(d)),
                "median_D_F": float(np.median(d)),
                "Q75_D_F": float(q75),
                "Q90_D_F": float(q90),
                "Q95_D_F": float(q95),
                "Q99_D_F": float(q99),
                "frac_update_days": float(np.mean(is_update)),
                "stress_share_among_update_days": float(np.mean(is_stress[is_update])) if np.any(is_update) else np.nan,
                "stress_days_with_large_update": float(np.mean(is_update[is_stress])) if np.any(is_stress) else np.nan,
                "normal_days_with_large_update": float(np.mean(is_update[is_normal])) if np.any(is_normal) else np.nan,
            }
        ]
    )
    return out


def _shock_alignment_summary(ts: pd.DataFrame, out_graph: Path, fig_dir: Path) -> None:
    """Quantify and visualize that graph updates concentrate in shock regimes."""
    stress_union = ts[["is_vix_stress", "is_us2y_shock", "is_oil_shock", "is_composite_stress"]].any(axis=1).values
    normal_mask = ts["is_normal"].values.astype(bool)

    d = ts["D_F"].values.astype(float)
    q90 = float(np.nanquantile(d, 0.90))
    is_large = d >= q90

    stress_mean = float(np.nanmean(d[stress_union])) if np.any(stress_union) else np.nan
    normal_mean = float(np.nanmean(d[normal_mask])) if np.any(normal_mask) else np.nan
    ratio = float(stress_mean / (normal_mean + 1e-12)) if np.isfinite(stress_mean) and np.isfinite(normal_mean) else np.nan

    share_large_in_stress = float(np.mean(is_large[stress_union])) if np.any(stress_union) else np.nan
    share_large_in_normal = float(np.mean(is_large[normal_mask])) if np.any(normal_mask) else np.nan
    stress_share_among_large = float(np.mean(stress_union[is_large])) if np.any(is_large) else np.nan

    out = pd.DataFrame(
        [
            {
                "D_F_mean_stress": stress_mean,
                "D_F_mean_normal": normal_mean,
                "D_F_stress_over_normal": ratio,
                "large_update_threshold_q90": q90,
                "large_update_share_in_stress_days": share_large_in_stress,
                "large_update_share_in_normal_days": share_large_in_normal,
                "stress_share_among_large_updates": stress_share_among_large,
            }
        ]
    )
    out.to_csv(out_graph / "shock_alignment_summary.csv", index=False)

    # Compact visual for paper claim
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["mean D_F\n(normal)", "mean D_F\n(stress)", "large-update\nshare normal", "large-update\nshare stress"]
    vals = [
        normal_mean,
        stress_mean,
        share_large_in_normal,
        share_large_in_stress,
    ]
    ax.bar(labels, vals, color=["#4c78a8", "#e45756", "#72b7b2", "#f58518"])
    ax.set_title("Graph Change Concentration in Shock Regimes")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "graph_change_shock_alignment.png", dpi=150)
    plt.close(fig)


def _graph_update_ratio_and_overlap(ts: pd.DataFrame, out_graph: Path, fig_dir: Path, rolling_window: int = 20, top_n: int = 30):
    d = ts["D_F"].values
    tau = float(np.quantile(d, 0.90))
    is_large = d >= tau
    stress_union = ts[["is_vix_stress", "is_us2y_shock", "is_oil_shock", "is_composite_stress"]].any(axis=1).values

    ratio_df = ts[["date", "D_F", "VIX", "Delta_US2Y_abs", "Oil_return_abs", "ShockScore"]].copy()
    ratio_df["is_large_update"] = is_large.astype(int)
    ratio_df["is_stress_day"] = stress_union.astype(int)
    ratio_df["rolling_update_ratio_20d"] = (
        pd.Series(is_large.astype(float), index=pd.to_datetime(ts["date"]))
        .rolling(rolling_window, min_periods=1)
        .mean()
        .values
    )
    ratio_df["rolling_stress_ratio_20d"] = (
        pd.Series(stress_union.astype(float), index=pd.to_datetime(ts["date"]))
        .rolling(rolling_window, min_periods=1)
        .mean()
        .values
    )
    ratio_df["rolling_D_F_20d"] = (
        pd.Series(d.astype(float), index=pd.to_datetime(ts["date"]))
        .rolling(rolling_window, min_periods=1)
        .mean()
        .values
    )
    stress_top_pct = 0.15
    stress_thr = float(np.nanquantile(ratio_df["rolling_stress_ratio_20d"].values, 1.0 - stress_top_pct))
    ratio_df["is_top15pct_stress_period"] = (
        ratio_df["rolling_stress_ratio_20d"].values >= stress_thr
    ).astype(int)
    ratio_df.to_csv(out_graph / "graph_update_ratio_timeseries.csv", index=False)

    # Top graph-change dates and nearest stress-date proximity
    dates = pd.to_datetime(ts["date"]).reset_index(drop=True)
    stress_dates = dates[stress_union]
    idx_top = np.argsort(d)[::-1][: min(top_n, len(d))]
    top_rows = []
    for rank, idx in enumerate(idx_top, start=1):
        dt = dates.iloc[idx]
        if len(stress_dates) > 0:
            dist_days = int(np.min(np.abs((stress_dates - dt).dt.days)))
        else:
            dist_days = np.nan
        top_rows.append(
            {
                "rank": rank,
                "date": dt,
                "D_F": float(d[idx]),
                "is_vix_stress": bool(ts.loc[idx, "is_vix_stress"]),
                "is_us2y_shock": bool(ts.loc[idx, "is_us2y_shock"]),
                "is_oil_shock": bool(ts.loc[idx, "is_oil_shock"]),
                "is_composite_stress": bool(ts.loc[idx, "is_composite_stress"]),
                "is_any_stress": bool(stress_union[idx]),
                "days_to_nearest_stress": dist_days,
            }
        )
    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(out_graph / "top_graph_change_event_overlap.csv", index=False)

    # Figure 1: rolling update ratio over test period
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = pd.to_datetime(ratio_df["date"])
    top_stress = ratio_df["is_top15pct_stress_period"].astype(bool).values
    ax.fill_between(
        x,
        0,
        1,
        where=top_stress,
        transform=ax.get_xaxis_transform(),
        color="#f4b400",
        alpha=0.30,
        label="top 15% stress period",
    )
    ax.plot(x, ratio_df["rolling_update_ratio_20d"], label="rolling update ratio (20d)", color="#d62728")
    ax.plot(x, ratio_df["rolling_stress_ratio_20d"], label="rolling stress ratio (20d)", color="#1f77b4", alpha=0.85)
    ax.axhline(np.mean(is_large), color="gray", linestyle="--", linewidth=1, label="avg update ratio")
    ax.set_ylabel("ratio")
    ax.set_title("Graph Reconfiguration Intensity Across Stress Regimes", fontweight="bold")
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        ratio_df["rolling_D_F_20d"],
        color="#2ca02c",
        linewidth=1.8,
        alpha=0.95,
        label="rolling avg D_F (20d)",
    )
    ax2.set_ylabel("rolling D_F")

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(fig_dir / "graph_update_ratio_over_time.png", dpi=150)
    plt.close(fig)

    # Figure 2: top graph-change dates vs macro shock events
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(dates, d, color="black", linewidth=1.2, label="D_F")
    ax.scatter(
        pd.to_datetime(top_df["date"]),
        top_df["D_F"],
        color="#d62728",
        s=32,
        label=f"top-{len(top_df)} graph-change dates",
        zorder=3,
    )
    for col, color, lbl in [
        ("is_vix_stress", "#9467bd", "VIX stress"),
        ("is_us2y_shock", "#2ca02c", "US2Y shock"),
        ("is_oil_shock", "#ff7f0e", "Oil shock"),
        ("is_composite_stress", "#17becf", "Composite stress"),
    ]:
        mask = ts[col].values.astype(bool)
        ax.vlines(dates[mask], ymin=np.nanmin(d), ymax=np.nanmin(d) + (np.nanmax(d) - np.nanmin(d)) * 0.08, color=color, alpha=0.35, linewidth=1)
    ax.set_title("Top graph changes and macro shock event timing")
    ax.set_ylabel("D_F")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "top_graph_change_vs_macro_shocks.png", dpi=150)
    plt.close(fig)


def _plot_story_style_dynamic_reconfiguration(
    ts: pd.DataFrame,
    fig_dir: Path,
    top_pct: float = 0.15,
    event_dates: list[str] | None = None,
    event_labels: list[str] | None = None,
):
    x = pd.to_datetime(ts["date"])
    d = ts["dynamic_intensity"].values.astype(float)
    roll = pd.Series(d, index=x).rolling(20, min_periods=1).mean().values
    q = float(np.nanquantile(d, 1.0 - top_pct))
    high = d >= q

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#060b1a")
    ax.set_facecolor("#0b1226")

    ax.bar(x[high], d[high], width=2.0, color="#ff5a5f", alpha=0.28, label=f"Top {int(top_pct*100)}% change days")
    ax.plot(x, d, color="#4fc3ff", linewidth=1.0, alpha=0.9, label="Daily Δ_norm")
    ax.plot(x, roll, color="#ffd54f", linewidth=2.2, label="20 day rolling avg")
    ax.axhline(q, color="#ff6e6e", linestyle="--", linewidth=1.2, alpha=0.85, label=f"{int((1-top_pct)*100)}th percentile ({q:.3f})")

    if event_dates and event_labels and len(event_dates) == len(event_labels):
        y_top = np.nanquantile(d, 0.94)
        y_bot = np.nanquantile(d, 0.14)
        for i, (ds, lbl) in enumerate(zip(event_dates, event_labels)):
            dt = pd.to_datetime(ds)
            if dt < x.min() or dt > x.max():
                continue
            ax.axvline(dt, color="#ff4d4d", linestyle=":", linewidth=1.2, alpha=0.75)
            yv = y_top if i % 2 == 0 else y_bot
            ax.scatter([dt], [yv], s=52, color="#ff4d4d", edgecolors="white", linewidths=0.8, zorder=5)
            ax.annotate(
                lbl,
                xy=(dt, yv),
                xytext=(0, 18 if i % 2 == 0 else -26),
                textcoords="offset points",
                ha="center",
                va="bottom" if i % 2 == 0 else "top",
                fontsize=9,
                fontweight="bold",
                color="#e8ecff",
                bbox=dict(boxstyle="round,pad=0.3", fc="#101a35", ec="#ff4d4d", lw=1.2, alpha=0.95),
                arrowprops=dict(arrowstyle="-", color="#ff4d4d", lw=1.0, alpha=0.85),
            )

    info = f"Test period: {x.min().date()} - {x.max().date()}\nTop {int(top_pct*100)}% high-change days: {int(high.sum())} / {len(x)}"
    ax.text(
        0.995,
        0.97,
        info,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#d0d7ff",
        bbox=dict(boxstyle="round,pad=0.35", fc="#182447", ec="#40508a", alpha=0.9),
    )

    ax.set_title("MACRO-HyGraph — Dynamic Graph Reconfiguration & Macro Regime Changes", fontsize=17, fontweight="bold", pad=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Δ_norm (Relative adjacency change)", fontsize=12)
    ax.grid(alpha=0.14, color="#8aa0ff")
    for s in ax.spines.values():
        s.set_color("#3a4571")
        s.set_linewidth(1.2)

    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=True, fontsize=10)
    leg.get_frame().set_facecolor("#101a35")
    leg.get_frame().set_edgecolor("#4b5b95")

    fig.tight_layout()
    fig.savefig(fig_dir / "graph_reconfiguration_regime_story.png", dpi=160)
    plt.close(fig)


def _plot_rolling_top_change_focus(
    ts: pd.DataFrame,
    fig_dir: Path,
    rolling_window: int = 20,
    top_n: int = 6,
    min_gap_days: int = 90,
):
    x = pd.to_datetime(ts["date"])
    y = pd.Series(ts["dynamic_intensity"].values.astype(float), index=x)
    r = y.rolling(rolling_window, min_periods=1).mean()

    # Pick top peaks with minimum time gap to avoid crowded labels.
    cand_idx = np.argsort(r.values)[::-1]
    chosen = []
    for idx in cand_idx:
        dt = r.index[idx]
        if len(chosen) == 0:
            chosen.append(idx)
        else:
            dists = [abs((dt - r.index[j]).days) for j in chosen]
            if min(dists) >= min_gap_days:
                chosen.append(idx)
        if len(chosen) >= min(top_n, len(r)):
            break
    top_idx = np.array(chosen, dtype=int)
    top_dates = r.index[top_idx]
    top_vals = r.values[top_idx]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor("#060b1a")
    ax.set_facecolor("#0b1226")

    ax.plot(r.index, r.values, color="#ffd54f", linewidth=2.2, label=f"{rolling_window}d rolling avg Δ_norm")
    ax.scatter(
        top_dates,
        top_vals,
        color="#ff4d4d",
        s=54,
        edgecolors="white",
        linewidths=0.9,
        zorder=5,
        label=f"Top {len(top_dates)} peaks",
    )
    # Keep guides but remove text labels to avoid clutter.
    for dt in top_dates:
        ax.axvline(dt, color="#ff4d4d", alpha=0.22, linewidth=0.9)

    ax.set_title("Rolling Graph-Change Intensity with Top Change Points Highlighted", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Δ_norm")
    ax.grid(alpha=0.14, color="#8aa0ff")
    for s in ax.spines.values():
        s.set_color("#3a4571")
        s.set_linewidth(1.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "graph_change_rolling_top_highlights.png", dpi=160)
    plt.close(fig)


def _corr_reg(ts: pd.DataFrame, out_dir: Path):
    cols = ["VIX", "Delta_US2Y_abs", "Oil_return_abs", "ShockScore"]
    rows = []
    for c in cols:
        cc = np.corrcoef(ts["D_F"].values, ts[c].values)[0, 1]
        rows.append({"var": c, "corr_with_D_F": float(cc)})
    pd.DataFrame(rows).to_csv(out_dir / "graph_change_correlation.csv", index=False)

    txt = []
    try:
        import statsmodels.api as sm

        X = ts[["is_vix_stress", "Delta_US2Y_abs", "Oil_return_abs", "ShockScore"]].astype(float)
        y = ts["D_F"].astype(float)
        Xc = sm.add_constant(X)
        m = sm.OLS(y, Xc, missing="drop").fit()
        txt.append("Regression: D_F(t)\n")
        txt.append(m.summary().as_text())

        y2 = ts["D_F"].shift(-1)
        X2 = ts[["ShockScore", "VIX", "Delta_US2Y_abs", "Oil_return_abs"]].astype(float)
        m2 = sm.OLS(y2, sm.add_constant(X2), missing="drop").fit()
        txt.append("\n\nLagged Regression: D_F(t+1)\n")
        txt.append(m2.summary().as_text())
    except Exception as e:
        txt.append("statsmodels unavailable or regression failed.\n")
        txt.append(str(e))
    (out_dir / "graph_change_regression.txt").write_text("\n".join(txt), encoding="utf-8")


def _plot_timeseries(ts: pd.DataFrame, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pd.to_datetime(ts["date"]), ts["D_F"], label="D_F")
    stress = ts[["is_vix_stress", "is_us2y_shock", "is_oil_shock", "is_composite_stress"]].any(axis=1)
    ax.scatter(pd.to_datetime(ts.loc[stress, "date"]), ts.loc[stress, "D_F"], s=8, color="red", label="stress")
    ax.set_title("Graph change over time")
    ax.set_ylabel("D_F")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "graph_change_timeseries_with_stress.png", dpi=150)
    plt.close(fig)


def _plot_distributions(ts: pd.DataFrame, fig_dir: Path):
    stress = ts[["is_vix_stress", "is_us2y_shock", "is_oil_shock", "is_composite_stress"]].any(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ts.loc[~stress, "D_F"], bins=30, alpha=0.6, label="normal", density=True)
    ax.hist(ts.loc[stress, "D_F"], bins=30, alpha=0.6, label="stress", density=True)
    ax.legend()
    ax.set_title("D_F distribution: normal vs stress")
    fig.tight_layout()
    fig.savefig(fig_dir / "graph_change_distribution_normal_vs_stress.png", dpi=150)
    plt.close(fig)

    mcols = ["D_F", "D_MA", "edge_turnover", "graph_entropy"]
    regimes = ["normal", "is_vix_stress", "is_us2y_shock", "is_oil_shock", "is_composite_stress"]
    labels = ["normal", "vix", "us2y", "oil", "composite"]
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    axs = axs.ravel()
    for i, c in enumerate(mcols):
        data = []
        for r in regimes:
            if r == "normal":
                data.append(ts.loc[ts["is_normal"], c].values)
            else:
                data.append(ts.loc[ts[r], c].values)
        axs[i].boxplot(data, labels=labels, showfliers=False)
        axs[i].set_title(c)
        axs[i].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(fig_dir / "graph_change_boxplot_by_regime.png", dpi=150)
    plt.close(fig)

    for xcol, fname in [
        ("ShockScore", "shockscore_vs_graph_change.png"),
        ("Delta_US2Y_abs", "us2yshock_vs_graph_change.png"),
        ("Oil_return_abs", "oilshock_vs_graph_change.png"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(ts[xcol], ts["D_F"], s=10, alpha=0.6)
        ax.set_xlabel(xcol)
        ax.set_ylabel("D_F")
        ax.set_title(f"{xcol} vs D_F")
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)

    def _bar(metric, fname):
        vals = {
            "normal": ts.loc[ts["is_normal"], metric].mean(),
            "vix": ts.loc[ts["is_vix_stress"], metric].mean(),
            "us2y": ts.loc[ts["is_us2y_shock"], metric].mean(),
            "oil": ts.loc[ts["is_oil_shock"], metric].mean(),
            "composite": ts.loc[ts["is_composite_stress"], metric].mean(),
        }
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(list(vals.keys()), list(vals.values()))
        ax.set_title(metric)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)

    _bar("graph_entropy", "graph_entropy_by_regime.png")
    _bar("edge_turnover", "edge_turnover_by_regime.png")


def _avg_adj_and_heatmaps(A, dates, ts, ccys, out_graph: Path, fig_dir: Path):
    masks = {
        "normal": ts["is_normal"].values,
        "vix": ts["is_vix_stress"].values,
        "us2y": ts["is_us2y_shock"].values,
        "oil": ts["is_oil_shock"].values,
        "composite": ts["is_composite_stress"].values,
    }

    def _avg(mask):
        if np.sum(mask) == 0:
            return np.full((A.shape[1], A.shape[2]), np.nan, dtype=np.float32)
        return np.nanmean(A[mask], axis=0)

    mats = {k: _avg(v) for k, v in masks.items()}
    np.savez(
        out_graph / "average_adjacency_by_regime.npz",
        A_normal=mats["normal"],
        A_vix=mats["vix"],
        A_us2y=mats["us2y"],
        A_oil=mats["oil"],
        A_composite=mats["composite"],
        Delta_A_vix=mats["vix"] - mats["normal"],
        Delta_A_us2y=mats["us2y"] - mats["normal"],
        Delta_A_oil=mats["oil"] - mats["normal"],
        Delta_A_composite=mats["composite"] - mats["normal"],
        currency_names=np.array(ccys, dtype=object),
    )

    def _heat(mat, title, fname, vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap="coolwarm")
        ax.set_yticks(np.arange(len(ccys)))
        ax.set_yticklabels(ccys)
        ax.set_xticks(np.arange(len(ccys)))
        ax.set_xticklabels(ccys, rotation=45, ha="right")
        ax.set_ylabel("target currency i")
        ax.set_xlabel("source currency j")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)

    _heat(mats["normal"], "A_normal", "adjacency_normal.png")
    _heat(mats["vix"], "A_vix", "adjacency_vix_stress.png")
    _heat(mats["vix"] - mats["normal"], "Delta_A_vix", "adjacency_diff_vix_minus_normal.png")
    _heat(mats["us2y"] - mats["normal"], "Delta_A_us2y", "adjacency_diff_us2y_minus_normal.png")
    _heat(mats["oil"] - mats["normal"], "Delta_A_oil", "adjacency_diff_oil_minus_normal.png")
    _heat(mats["composite"] - mats["normal"], "Delta_A_composite", "adjacency_diff_composite_minus_normal.png")
    return mats


def _centrality_tables(A, ts, ccys, table_dir: Path, fig_dir: Path):
    regimes = {
        "normal": ts["is_normal"].values,
        "VIX_top10": ts["is_vix_stress"].values,
        "US2Y_shock_top10": ts["is_us2y_shock"].values,
        "Oil_shock_top10": ts["is_oil_shock"].values,
        "composite_macro_shock_top10": ts["is_composite_stress"].values,
    }
    rows = []
    top_rows = []
    for reg, mask in regimes.items():
        if np.sum(mask) == 0:
            continue
        As = A[mask]
        incoming = As.sum(axis=1).mean(axis=0)  # sum_i A_ij then mean_t
        out_conc = As.max(axis=2).mean(axis=0)  # mean_t max_j A_ij
        ent = float(np.mean(-np.sum(As * np.log(As + 1e-12), axis=2)))
        rank_idx = np.argsort(incoming)[::-1]
        for r, j in enumerate(rank_idx, start=1):
            rows.append(
                {
                    "regime": reg,
                    "currency": ccys[j],
                    "incoming_centrality": float(incoming[j]),
                    "outgoing_concentration": float(out_conc[j]),
                    "mean_entropy": ent,
                    "rank_incoming": int(r),
                }
            )
            if r <= 3:
                top_rows.append({"regime": reg, "rank": r, "currency": ccys[j], "incoming_centrality": float(incoming[j])})
    df = pd.DataFrame(rows)
    top = pd.DataFrame(top_rows)
    df.to_csv(table_dir / "graph_centrality_by_regime.csv", index=False)
    top.to_csv(table_dir / "top_central_currencies_by_regime.csv", index=False)

    if not top.empty:
        pivot = top.pivot(index="regime", columns="rank", values="currency")
        fig, ax = plt.subplots(figsize=(8, 4))
        reg = list(top["regime"].drop_duplicates())
        x = np.arange(len(reg))
        for rk in sorted(top["rank"].unique()):
            vals = []
            for rr in reg:
                sub = top[(top["regime"] == rr) & (top["rank"] == rk)]
                vals.append(float(sub["incoming_centrality"].iloc[0]) if len(sub) else np.nan)
            ax.plot(x, vals, marker="o", label=f"rank{rk}")
        ax.set_xticks(x)
        ax.set_xticklabels(reg, rotation=20)
        ax.set_title("Top central currencies by regime (incoming centrality)")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "top_central_currencies_by_regime.png", dpi=150)
        plt.close(fig)


def _write_report(base_dir: Path):
    tab = base_dir / "tables" / "ours_stress_improvement_vs_baselines.csv"
    ratio = base_dir / "graph_adaptation" / "graph_change_stress_ratio.csv"
    ratio_tbl = base_dir / "graph_adaptation" / "graph_change_regime_mean_median_ratio.csv"
    stat_tbl = base_dir / "graph_adaptation" / "graph_change_regime_stat_tests.csv"
    eff = base_dir / "graph_adaptation" / "graph_update_efficiency.csv"
    align = base_dir / "graph_adaptation" / "shock_alignment_summary.csv"
    cen = base_dir / "tables" / "top_central_currencies_by_regime.csv"
    txt = ["# GRAPH_ADAPTATION_REPORT", ""]
    txt.append("## 1) Experiment purpose")
    txt.append("Stress robustness and dynamic graph adaptation analysis for MACRO-HyGraph (Ours), excluding FiLMHyGraph.")
    txt.append("")
    txt.append("## 2) Key computed summaries")
    if tab.exists():
        d = pd.read_csv(tab)
        if len(d):
            best = d.sort_values("improvement_rmse", ascending=False).iloc[0]
            txt.append(
                f"- Largest Ours RMSE improvement: {best['improvement_rmse']:.2f}% vs {best['baseline_model']} in {best['regime']}."
            )
    if ratio.exists():
        r = pd.read_csv(ratio)
        if len(r):
            rr = r.sort_values("D_F_stress_over_normal", ascending=False).iloc[0]
            txt.append(
                f"- Highest graph-change ratio: {rr['D_F_stress_over_normal']:.2f}x in {rr['regime']} vs normal."
            )
    if ratio_tbl.exists():
        rt = pd.read_csv(ratio_tbl)
        dfr = rt[rt["Metric"] == "D_F"]
        if len(dfr):
            vix_row = dfr.iloc[0]
            txt.append(f"- D_F mean ratio example (VIX/Normal): {vix_row['VIX_over_Normal_mean_ratio']:.2f}x.")
    if stat_tbl.exists():
        st = pd.read_csv(stat_tbl)
        s = st[(st["metric"] == "D_F") & (st["regime"] == "VIX")]
        if len(s):
            sv = s.iloc[0]
            txt.append(
                f"- D_F VIX vs Normal p-values: Mann-Whitney={sv['mannwhitney_u_pvalue']:.4g}, "
                f"Permutation={sv['permutation_pvalue']:.4g}."
            )
    if eff.exists():
        e = pd.read_csv(eff).iloc[0]
        txt.append(f"- Large-update day fraction: {e['frac_update_days']:.2%}.")
        txt.append(f"- Stress share among update days: {e['stress_share_among_update_days']:.2%}.")
    if align.exists():
        a = pd.read_csv(align).iloc[0]
        txt.append(f"- Mean graph change ratio (stress/normal): {a['D_F_stress_over_normal']:.2f}x.")
        txt.append(
            f"- Large-update share normal vs stress: "
            f"{a['large_update_share_in_normal_days']:.2%} vs {a['large_update_share_in_stress_days']:.2%}."
        )
    if cen.exists():
        c = pd.read_csv(cen)
        if len(c):
            top = c[c["rank"] == 1]
            txt.append("- Top incoming-central currency by regime:")
            for _, row in top.iterrows():
                txt.append(f"  - {row['regime']}: {row['currency']} ({row['incoming_centrality']:.4f})")
    txt.append("")
    txt.append("## 3) Output files")
    txt.append("- metrics/stress_metrics_raw.csv")
    txt.append("- tables/stress_metrics_summary.csv")
    txt.append("- tables/ours_stress_improvement_vs_baselines.csv")
    txt.append("- graph_adaptation/*.csv, *.npz")
    txt.append("- figures/*.png")
    (base_dir / "GRAPH_ADAPTATION_REPORT.md").write_text("\n".join(txt), encoding="utf-8")


def main():
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    base_out = Path(args.output_dir)
    graph_out = base_out / "graph_adaptation"
    fig_out = base_out / "figures"
    tab_out = base_out / "tables"
    for p in [graph_out, fig_out, tab_out]:
        p.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg(args.config, args.data_path, args.lookback)
    data_bundle = prepare_data_split(cfg, split_mode="602020", data_path=args.data_path, macro_features=ALL_MACRO_FEATURES)
    stress_ctx = build_stress_context(
        cfg,
        data_path=args.data_path,
        lookback=args.lookback,
        quantile=args.stress_quantile,
        threshold_scope=args.threshold_scope,
    )

    Xl_te, Xm_te, y_te, dates_df = build_windows(data_bundle, args.lookback, "test", add_rv=True, return_dates=True)
    dates = pd.DatetimeIndex(pd.to_datetime(dates_df["target_date"]))
    input_end_dates = pd.DatetimeIndex(pd.to_datetime(dates_df["input_end_date"]))

    import torch

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    # Infer hidden sizes from checkpoint to avoid mismatch with default Config values.
    if "local_gru.weight_hh_l0" in state:
        h = int(state["local_gru.weight_hh_l0"].shape[1])
        cfg.hidden = h
        cfg.hybrid_hidden = h
    model = create_model("Ours", cfg)
    model.load_state_dict(state, strict=False)
    model.eval()

    A = _collect_adjacency(model, Xl_te, Xm_te)
    ccys = data_bundle["currency_names"]
    np.savez(
        graph_out / "ours_adjacency_test.npz",
        A=A,
        dates=np.array(dates.astype(str)),
        input_end_dates=np.array(input_end_dates.astype(str)),
        target_dates=np.array(dates.astype(str)),
        currency_names=np.array(ccys, dtype=object),
    )

    ts = _graph_change_df(A, dates, stress_ctx, top_k=args.top_k_turnover)
    ts.to_csv(graph_out / "graph_change_timeseries.csv", index=False)
    by_reg, ratio = _regime_stats(ts)
    by_reg.to_csv(graph_out / "graph_change_by_regime.csv", index=False)
    ratio.to_csv(graph_out / "graph_change_stress_ratio.csv", index=False)
    ext_masks = _compute_extended_regime_masks(
        data_bundle=data_bundle,
        dates_test=dates,
        y_test=y_te,
        data_path=args.data_path,
        threshold_scope=args.threshold_scope,
    )
    ext_df = _extended_regime_change_table(ts, ext_masks, graph_out)
    _plot_extended_ratio_bars(ext_df, fig_out)
    _regime_mean_median_ratio_table(ts, graph_out)
    _regime_stat_tests(ts, graph_out)
    _efficiency(ts).to_csv(graph_out / "graph_update_efficiency.csv", index=False)
    _corr_reg(ts, graph_out)
    _graph_update_ratio_and_overlap(ts, graph_out, fig_out, rolling_window=20, top_n=30)
    _shock_alignment_summary(ts, graph_out, fig_out)
    _plot_story_style_dynamic_reconfiguration(
        ts,
        fig_out,
        top_pct=args.story_top_pct,
        event_dates=[
            "2022-03-16",
            "2022-09-21",
            "2023-10-19",
            "2024-03-21",
            "2025-04-02",
        ],
        event_labels=[
            "Russia-Ukraine\n+ Fed hiking",
            "Fed +75bps\nUSD peak",
            "Israel-Hamas war\n+ UST yield surge",
            "Risk-on surge",
            "Tariff/trade uncertainty\nUS high-change day",
        ],
    )
    _plot_rolling_top_change_focus(ts, fig_out, rolling_window=20, top_n=6)

    _plot_timeseries(ts, fig_out)
    _plot_distributions(ts, fig_out)
    _avg_adj_and_heatmaps(A, dates, ts, ccys, graph_out, fig_out)
    _centrality_tables(A, ts, ccys, tab_out, fig_out)
    _write_report(base_out)
    LOGGER.info("graph adaptation analysis saved under: %s", base_out)


if __name__ == "__main__":
    main()
