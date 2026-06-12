import argparse
import logging
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from exp_utils import compute_metrics, load_or_create_results_dir, save_csv
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(__file__))
    from exp_utils import compute_metrics, load_or_create_results_dir, save_csv

LOGGER = logging.getLogger("exp_stress_regime_macro_hygraph")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Stress regime evaluation from saved full-ablation predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pred-dir", default=str(ROOT / "results" / "model_ablation" / "predictions"))
    p.add_argument("--data-path", default=str(ROOT / "data" / "factor_daily_legacy.csv"))
    p.add_argument("--stress-quantile", type=float, default=0.90)
    p.add_argument("--output-dir", default=str(ROOT / "results" / "stress_regime_analysis"))
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _load_prediction_files(pred_dir: Path) -> list[Path]:
    files = []
    for p in pred_dir.glob("*.parquet"):
        if p.name.endswith("_matrix.parquet"):
            continue
        files.append(p)
    return sorted(files)


def _build_test_macro_frame(data_path: str, test_dates: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if "Date" not in df.columns:
        raise ValueError("data-path CSV must contain 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date")
    df = df.set_index("Date")

    req = ["Global_VIX", "Global_US2Y", "Global_Oil"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required macro columns: {missing}. Available: {list(df.columns)}")

    out = pd.DataFrame(index=test_dates.unique().sort_values())
    out["vix"] = df["Global_VIX"].reindex(out.index)
    out["us2y_abs_diff"] = df["Global_US2Y"].diff().abs().reindex(out.index)
    out["oil_abs_logret"] = np.log(df["Global_Oil"]).diff().abs().reindex(out.index)
    return out.dropna()


def _stress_dates_from_macro(macro_test: pd.DataFrame, q: float) -> dict[str, pd.DatetimeIndex]:
    vix_th = macro_test["vix"].quantile(q)
    us2y_th = macro_test["us2y_abs_diff"].quantile(q)
    oil_th = macro_test["oil_abs_logret"].quantile(q)
    return {
        "VIX_top10": macro_test.index[macro_test["vix"] >= vix_th],
        "US2Y_shock_top10": macro_test.index[macro_test["us2y_abs_diff"] >= us2y_th],
        "Oil_shock_top10": macro_test.index[macro_test["oil_abs_logret"] >= oil_th],
    }


def _to_matrix(pred_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    pred_df = pred_df.copy()
    pred_df["target_date"] = pd.to_datetime(pred_df["target_date"])
    ccy_order = list(pred_df["currency"].drop_duplicates())
    pred_mat = (
        pred_df.pivot_table(index="target_date", columns="currency", values="pred", aggfunc="first")
        .reindex(columns=ccy_order)
        .sort_index()
    )
    tgt_mat = (
        pred_df.pivot_table(index="target_date", columns="currency", values="target", aggfunc="first")
        .reindex(columns=ccy_order)
        .sort_index()
    )
    common_idx = pred_mat.index.intersection(tgt_mat.index)
    pred_mat = pred_mat.reindex(common_idx).fillna(0.0)
    tgt_mat = tgt_mat.reindex(common_idx).fillna(0.0)
    return pred_mat.values.astype(np.float32), tgt_mat.values.astype(np.float32), ccy_order, common_idx


def _evaluate_subset(
    pred: np.ndarray,
    target: np.ndarray,
    ccy_order: list[str],
    dates: pd.DatetimeIndex,
    selected_dates: pd.DatetimeIndex,
) -> dict:
    mask_date = dates.isin(selected_dates)
    if mask_date.sum() == 0:
        return {
            "n_obs": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "hit_ccy": np.nan,
            "hit_pair": np.nan,
            "extreme_rmse": np.nan,
            "extreme_hit": np.nan,
        }
    p = pred[mask_date]
    t = target[mask_date]
    usd_idx = ccy_order.index("USD") if "USD" in ccy_order else 0
    m = compute_metrics(p, t, n_ccy=len(ccy_order), usd_idx=usd_idx, q=0.90)
    m["n_obs"] = int(mask_date.sum())
    return m


def _evaluate_fx_move_top10(pred_df: pd.DataFrame, q: float) -> pd.DatetimeIndex:
    fx_score = (
        pred_df.assign(abs_target=lambda x: x["target"].abs())
        .groupby("target_date", as_index=True)["abs_target"]
        .max()
        .sort_index()
    )
    th = fx_score.quantile(q)
    return pd.DatetimeIndex(fx_score.index[fx_score >= th])


def _make_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grp_cols = ["model", "lookback", "stress_type"]
    metric_cols = ["rmse", "hit_ccy", "hit_pair"]
    for keys, g in df.groupby(grp_cols):
        out = {"model": keys[0], "lookback": keys[1], "stress_type": keys[2]}
        for c in metric_cols:
            out[f"{c}_mean"] = float(g[c].mean())
            out[f"{c}_std"] = float(g[c].std(ddof=0))
        rows.append(out)
    return pd.DataFrame(rows).sort_values(grp_cols).reset_index(drop=True)


def _make_ours_improvement(summary_df: pd.DataFrame) -> pd.DataFrame:
    baselines = ["FiLMHyGraph", "StaticGraph", "NoMacro", "LSTM", "MLP"]
    rows = []
    for (lb, st), g in summary_df.groupby(["lookback", "stress_type"]):
        sub = {r["model"]: r for _, r in g.iterrows()}
        if "Ours" not in sub:
            continue
        ours = sub["Ours"]
        for b in baselines:
            if b not in sub:
                continue
            bb = sub[b]
            rmse_imp = (bb["rmse_mean"] - v5["rmse_mean"]) / bb["rmse_mean"] * 100.0 if bb["rmse_mean"] != 0 else np.nan
            rows.append(
                {
                    "lookback": lb,
                    "stress_type": st,
                    "baseline_model": b,
                    "rmse_improvement_pct": float(rmse_imp),
                    "hit_ccy_improvement": float(ours["hit_ccy_mean"] - bb["hit_ccy_mean"]),
                    "hit_pair_improvement": float(ours["hit_pair_mean"] - bb["hit_pair_mean"]),
                }
            )
    return pd.DataFrame(rows)


def _plot_metric_bar(summary_df: pd.DataFrame, metric: str, out_path: Path):
    stress_types = list(summary_df["stress_type"].drop_duplicates())
    models = list(summary_df["model"].drop_duplicates())
    x = np.arange(len(stress_types))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, m in enumerate(models):
        vals = []
        for s in stress_types:
            sub = summary_df[(summary_df["model"] == m) & (summary_df["stress_type"] == s)]
            vals.append(float(sub[f"{metric}_mean"].iloc[0]) if len(sub) else np.nan)
        color = "#d62728" if m == "Ours" else None
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=m, color=color, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(stress_types, rotation=15)
    ax.set_ylabel(metric)
    ax.set_title(f"Stress Regime {metric.upper()} by Model")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_ours_improvement(imp_df: pd.DataFrame, out_path: Path):
    if imp_df.empty:
        return
    stress_types = list(imp_df["stress_type"].drop_duplicates())
    baselines = list(imp_df["baseline_model"].drop_duplicates())
    x = np.arange(len(stress_types))
    width = 0.8 / max(len(baselines), 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, b in enumerate(baselines):
        vals = []
        for s in stress_types:
            sub = imp_df[(imp_df["baseline_model"] == b) & (imp_df["stress_type"] == s)]
            vals.append(float(sub["rmse_improvement_pct"].iloc[0]) if len(sub) else np.nan)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=b, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(stress_types, rotation=15)
    ax.set_ylabel("RMSE Improvement of Ours vs Baseline (%)")
    ax.set_title("Ours Improvement by Stress Regime")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dirs = load_or_create_results_dir(args.output_dir, ["metrics", "tables", "figures"])
    pred_files = _load_prediction_files(Path(args.pred_dir))
    if not pred_files:
        raise FileNotFoundError(f"No prediction parquet files found under: {args.pred_dir}")

    rows = []
    for p in pred_files:
        pred_df = pd.read_parquet(p)
        if "target_date" not in pred_df.columns:
            raise ValueError(f"Missing target_date in prediction file: {p}")
        pred_df["target_date"] = pd.to_datetime(pred_df["target_date"])
        model = str(pred_df["model"].iloc[0])
        lookback = str(pred_df["lookback"].iloc[0])
        seed = int(pred_df["seed"].iloc[0])

        pred, target, ccy_order, dates = _to_matrix(pred_df)
        macro_test = _build_test_macro_frame(args.data_path, dates)
        stress_dates = _stress_dates_from_macro(macro_test, args.stress_quantile)
        stress_dates["FX_move_top10"] = _evaluate_fx_move_top10(pred_df, args.stress_quantile)

        for stype, sdates in stress_dates.items():
            m = _evaluate_subset(pred, target, ccy_order, dates, sdates)
            rows.append(
                {
                    "model": model,
                    "lookback": lookback,
                    "seed": seed,
                    "stress_type": stype,
                    "n_obs": m["n_obs"],
                    "rmse": m["rmse"],
                    "mae": m["mae"],
                    "hit_ccy": m["hit_ccy"],
                    "hit_pair": m["hit_pair"],
                    "extreme_rmse": m["extreme_rmse"],
                    "extreme_hit": m["extreme_hit"],
                }
            )
        LOGGER.info("processed %s", p.name)

    stress_df = pd.DataFrame(rows).sort_values(["model", "lookback", "seed", "stress_type"]).reset_index(drop=True)
    save_csv(out_dirs["metrics"] / "stress_metrics.csv", stress_df)

    summary_df = _make_summary(stress_df)
    save_csv(out_dirs["tables"] / "stress_metrics_summary.csv", summary_df)

    imp_df = _make_ours_improvement(summary_df)
    save_csv(out_dirs["tables"] / "ours_stress_improvement.csv", imp_df)

    _plot_metric_bar(summary_df, "hit_ccy", out_dirs["figures"] / "stress_hit_bar.png")
    _plot_metric_bar(summary_df, "rmse", out_dirs["figures"] / "stress_rmse_bar.png")
    _plot_ours_improvement(imp_df, out_dirs["figures"] / "ours_improvement_by_stress.png")

    LOGGER.info("saved: %s", out_dirs["metrics"] / "stress_metrics.csv")
    LOGGER.info("saved: %s", out_dirs["tables"] / "stress_metrics_summary.csv")
    LOGGER.info("saved: %s", out_dirs["tables"] / "ours_stress_improvement.csv")


if __name__ == "__main__":
    main()
