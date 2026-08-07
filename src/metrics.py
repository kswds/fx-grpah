from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def pairwise_hit(pred: np.ndarray, target: np.ndarray, non_usd_mask: np.ndarray) -> float:
    pred = pred[:, non_usd_mask]
    target = target[:, non_usd_mask]
    hits = []
    for i in range(pred.shape[1]):
        for j in range(i + 1, pred.shape[1]):
            hits.append(np.sign(pred[:, i] - pred[:, j]) == np.sign(target[:, i] - target[:, j]))
    return float(np.mean(np.concatenate(hits))) if hits else 0.0


def information_coefficient(pred: np.ndarray, target: np.ndarray, non_usd_mask: np.ndarray) -> float:
    pred = pred[:, non_usd_mask]
    target = target[:, non_usd_mask]
    vals = []
    for t in range(len(pred)):
        if np.std(pred[t]) < 1e-10 or np.std(target[t]) < 1e-10:
            continue
        vals.append(pd.Series(pred[t]).corr(pd.Series(target[t]), method="spearman"))
    return float(np.nanmean(vals)) if vals else 0.0


def long_short_returns(pred: np.ndarray, target: np.ndarray, non_usd_mask: np.ndarray) -> np.ndarray:
    pred = pred[:, non_usd_mask]
    target = target[:, non_usd_mask]
    if pred.shape[1] == 0:
        return np.array([], dtype=float)
    k = max(1, pred.shape[1] // 3)
    rets = []
    for t in range(len(pred)):
        order = np.argsort(pred[t])
        shorts = order[:k]
        longs = order[-k:]
        rets.append(target[t, longs].mean() - target[t, shorts].mean())
    return np.asarray(rets, dtype=float)


def sharpe_ratio(x: np.ndarray) -> float:
    if len(x) < 2 or np.std(x, ddof=1) < 1e-12:
        return 0.0
    return float((x.mean() / (x.std(ddof=1) + 1e-12)) * np.sqrt(252))


def sortino_ratio(x: np.ndarray) -> float:
    downside = x[x < 0]
    if len(x) < 2:
        return 0.0
    denom = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if denom < 1e-12:
        return 0.0
    return float((x.mean() / (denom + 1e-12)) * np.sqrt(252))


def max_drawdown(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    curve = np.cumsum(x)
    peak = np.maximum.accumulate(curve)
    dd = curve - peak
    return float(dd.min())


def compute_metrics(pred: np.ndarray, target: np.ndarray, non_usd_mask: np.ndarray) -> Dict[str, float]:
    pred_nu = pred[:, non_usd_mask]
    tgt_nu = target[:, non_usd_mask]
    rmse = float(np.sqrt(np.mean((pred_nu - tgt_nu) ** 2)))
    mae = float(np.mean(np.abs(pred_nu - tgt_nu)))
    hit = float(np.mean(np.sign(pred_nu) == np.sign(tgt_nu)))
    abs_target = np.abs(tgt_nu)
    non_tiny_thresh = float(np.nanmedian(abs_target))
    extreme_thresh = float(np.nanquantile(abs_target, 0.8))
    non_tiny_mask = abs_target >= non_tiny_thresh
    extreme_mask = abs_target >= extreme_thresh
    non_tiny_hit = float(np.mean((np.sign(pred_nu) == np.sign(tgt_nu))[non_tiny_mask])) if non_tiny_mask.any() else 0.0
    extreme_hit = float(np.mean((np.sign(pred_nu) == np.sign(tgt_nu))[extreme_mask])) if extreme_mask.any() else 0.0
    ls = long_short_returns(pred, target, non_usd_mask)
    return {
        "rmse": rmse,
        "mae": mae,
        "hit_ratio": hit,
        "non_tiny_hit_ratio": non_tiny_hit,
        "extreme_hit_ratio": extreme_hit,
        "pairwise_hit": pairwise_hit(pred, target, non_usd_mask),
        "ic": information_coefficient(pred, target, non_usd_mask),
        "long_short_sharpe": sharpe_ratio(ls),
        "long_short_sortino": sortino_ratio(ls),
        "max_drawdown": max_drawdown(ls),
        "cumulative_return": float(ls.sum()) if len(ls) else 0.0,
    }


def compute_single_currency_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    valid = np.isfinite(pred) & np.isfinite(target)
    pred = pred[valid]
    target = target[valid]
    if len(pred) == 0:
        return {
            "n_obs": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "hit_ratio": 0.0,
            "non_tiny_hit_ratio": 0.0,
            "extreme_hit_ratio": 0.0,
            "pearson_corr": 0.0,
            "spearman_corr": 0.0,
        }
    abs_target = np.abs(target)
    non_tiny_thresh = float(np.nanmedian(abs_target))
    extreme_thresh = float(np.nanquantile(abs_target, 0.8))
    sign_hit = np.sign(pred) == np.sign(target)
    non_tiny_mask = abs_target >= non_tiny_thresh
    extreme_mask = abs_target >= extreme_thresh
    pearson_corr = 0.0
    if len(pred) >= 2 and np.std(pred) > 1e-12 and np.std(target) > 1e-12:
        pearson_corr = float(np.corrcoef(pred, target)[0, 1])
    spearman_corr = 0.0
    if len(pred) >= 2:
        pred_s = pd.Series(pred)
        target_s = pd.Series(target)
        if pred_s.nunique() > 1 and target_s.nunique() > 1:
            spearman_corr = float(pred_s.corr(target_s, method="spearman"))
    return {
        "n_obs": float(len(pred)),
        "rmse": float(np.sqrt(np.mean((pred - target) ** 2))),
        "mae": float(np.mean(np.abs(pred - target))),
        "hit_ratio": float(np.mean(sign_hit)),
        "non_tiny_hit_ratio": float(np.mean(sign_hit[non_tiny_mask])) if non_tiny_mask.any() else 0.0,
        "extreme_hit_ratio": float(np.mean(sign_hit[extreme_mask])) if extreme_mask.any() else 0.0,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
    }


def build_currency_metrics_from_predictions(prediction_df: pd.DataFrame, group_cols: List[str] | None = None) -> pd.DataFrame:
    required_cols = {"model", "currency", "pred", "target"}
    missing = required_cols.difference(prediction_df.columns)
    if missing:
        raise ValueError(f"prediction_df is missing required columns: {sorted(missing)}")
    group_cols = group_cols or ["model", "currency", "seed"]
    rows = []
    for keys, grp in prediction_df.groupby(group_cols, dropna=False):
        key_vals = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_vals))
        row.update(compute_single_currency_metrics(grp["pred"].to_numpy(), grp["target"].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_currency_metrics(raw_df: pd.DataFrame, out_dir: Path) -> None:
    if raw_df.empty:
        return
    metric_cols = [
        "n_obs",
        "rmse",
        "mae",
        "hit_ratio",
        "non_tiny_hit_ratio",
        "extreme_hit_ratio",
        "pearson_corr",
        "spearman_corr",
    ]
    agg = raw_df.groupby(["model", "currency"], as_index=False)[metric_cols].agg(["mean", "std", "min", "max"])
    agg.columns = ["_".join([c for c in col if c]).strip("_") for col in agg.columns.to_flat_index()]
    agg = agg.rename(columns={"model_": "model", "currency_": "currency"})
    save_dataframe(raw_df, out_dir / "currency_metrics_detail.csv")
    save_dataframe(agg, out_dir / "currency_metrics_aggregate.csv")
    md_lines = [
        "# Currency-Level Predictive Comparison",
        "",
        "| Model | Currency | Mean Hit | Mean RMSE | Mean MAE | Mean Spearman |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in agg.sort_values(["currency", "hit_ratio_mean", "rmse_mean"], ascending=[True, False, True]).iterrows():
        md_lines.append(
            "| {model} | {currency} | {hit_ratio_mean:.4f} | {rmse_mean:.6f} | {mae_mean:.6f} | {spearman_corr_mean:.4f} |".format(
                **row.to_dict()
            )
        )
    (out_dir / "currency_metrics_summary.md").write_text("\n".join(md_lines), encoding="utf-8")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def summarize_metrics(raw_df: pd.DataFrame, out_dir: Path) -> None:
    if raw_df.empty:
        return
    metric_cols = [
        "rmse",
        "mae",
        "hit_ratio",
        "non_tiny_hit_ratio",
        "extreme_hit_ratio",
        "pairwise_hit",
        "ic",
        "long_short_sharpe",
        "long_short_sortino",
        "max_drawdown",
        "cumulative_return",
    ]
    agg = raw_df.groupby(["model", "universe", "lookback"], as_index=False)[metric_cols].agg(["mean", "std", "min", "max"])
    agg.columns = ["_".join([c for c in col if c]).strip("_") for col in agg.columns.to_flat_index()]
    agg = agg.rename(columns={"model_": "model", "universe_": "universe", "lookback_": "lookback"})
    save_dataframe(raw_df, out_dir / "seed_metrics_detail.csv")
    save_dataframe(agg, out_dir / "seed_metrics_aggregate.csv")
    md_lines = [
        "# Predictive Comparison",
        "",
        "| Model | Universe | Lookback | Mean Hit | Mean Extreme Hit | Mean RMSE | Mean MAE | Mean Pairwise Hit | Mean IC | Mean LS Sharpe |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in agg.sort_values(["hit_ratio_mean", "rmse_mean"], ascending=[False, True]).iterrows():
        md_lines.append(
            "| {model} | {universe} | {lookback:.0f} | {hit_ratio_mean:.4f} | {extreme_hit_ratio_mean:.4f} | {rmse_mean:.6f} | {mae_mean:.6f} | {pairwise_hit_mean:.4f} | {ic_mean:.4f} | {long_short_sharpe_mean:.4f} |".format(
                **row.to_dict()
            )
        )
    (out_dir / "metrics_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
