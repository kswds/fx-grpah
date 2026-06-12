import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from config import Config

try:
    from exp_utils import (
        ALL_MACRO_FEATURES,
        DEFAULT_DISPLAY_NAMES,
        compute_metrics,
        load_or_create_results_dir,
        predict_model,
        prepare_data_split,
        save_csv,
        save_json,
        save_predictions,
        train_model,
    )
    from stress_utils import build_stress_context
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(__file__))
    from exp_utils import (
        ALL_MACRO_FEATURES,
        DEFAULT_DISPLAY_NAMES,
        compute_metrics,
        load_or_create_results_dir,
        predict_model,
        prepare_data_split,
        save_csv,
        save_json,
        save_predictions,
        train_model,
    )
    from stress_utils import build_stress_context


LOGGER = logging.getLogger("exp_stress_regime_performance")
METRIC_COLS = ["rmse", "mae", "hit_ccy", "hit_pair", "extreme_rmse", "extreme_hit"]
BASELINE_COMPARE = ["StaticGraph", "NoMacro", "NoGraph", "PureGraphFX", "LSTM", "MLP", "GAT"]


def _parse_args():
    p = argparse.ArgumentParser(
        description="Stress regime performance evaluation (exclude FiLMHyGraph).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-path", default=str(ROOT / "data" / "factor_daily_legacy.csv"))
    p.add_argument("--models", nargs="+", default=["MLP", "LSTM", "GAT", "NoGraph", "NoMacro", "StaticGraph", "PureGraphFX", "Ours"])
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--stress-quantile", type=float, default=0.90)
    p.add_argument("--threshold-scope", default="trainval", choices=["trainval", "test"])
    p.add_argument("--output-dir", default=str(ROOT / "results" / "stress_regime_analysis"))
    p.add_argument("--pred-dir", default="")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _pred_file(pred_dir: Path, model: str, lookback: int, seed: int) -> Path:
    return pred_dir / f"{model}_L{lookback}_seed{seed}.parquet"


def _to_long(pred, target, dates_df, model, lookback, seed, ccys):
    rows = []
    for t in range(pred.shape[0]):
        for i, ccy in enumerate(ccys):
            rows.append(
                {
                    "date": dates_df.iloc[t]["target_date"],
                    "input_end_date": dates_df.iloc[t]["input_end_date"],
                    "target_date": dates_df.iloc[t]["target_date"],
                    "model": model,
                    "lookback": str(lookback),
                    "seed": seed,
                    "currency": ccy,
                    "pred": float(pred[t, i]),
                    "target": float(target[t, i]),
                }
            )
    return pd.DataFrame(rows)


def _matrix_from_long(df: pd.DataFrame):
    df = df.copy()
    df["target_date"] = pd.to_datetime(df["target_date"])
    ccys = list(df["currency"].drop_duplicates())
    pred = df.pivot_table(index="target_date", columns="currency", values="pred", aggfunc="first").reindex(columns=ccys).sort_index()
    tgt = df.pivot_table(index="target_date", columns="currency", values="target", aggfunc="first").reindex(columns=ccys).sort_index()
    idx = pred.index.intersection(tgt.index)
    return pred.reindex(idx).fillna(0.0).values, tgt.reindex(idx).fillna(0.0).values, ccys, pd.DatetimeIndex(idx)


def _eval_regimes(pred, target, ccys, dates, stress_masks):
    usd_idx = ccys.index("USD") if "USD" in ccys else 0
    rows = []
    for regime in stress_masks.columns:
        mask = stress_masks[regime].reindex(dates).fillna(False).values.astype(bool)
        p = pred[mask]
        t = target[mask]
        if len(p) == 0:
            m = {k: np.nan for k in METRIC_COLS}
            n_obs = 0
        else:
            m = compute_metrics(p, t, n_ccy=len(ccys), usd_idx=usd_idx, q=0.90)
            n_obs = int(len(p))
        row = {"regime": regime, "n_obs": n_obs}
        row.update({k: m[k] for k in METRIC_COLS})
        rows.append(row)
    return rows


def _summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, regime), g in raw_df.groupby(["model", "regime"]):
        r = {"model": model, "regime": regime, "n_obs_mean": float(g["n_obs"].mean())}
        for c in METRIC_COLS:
            r[f"{c}_mean"] = float(g[c].mean())
            r[f"{c}_std"] = float(g[c].std(ddof=0))
        out.append(r)
    return pd.DataFrame(out).sort_values(["regime", "model"]).reset_index(drop=True)


def _summary_md(df: pd.DataFrame, path: Path):
    cols = [
        "model",
        "regime",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "hit_ccy_mean",
        "hit_ccy_std",
        "hit_pair_mean",
        "hit_pair_std",
        "extreme_rmse_mean",
        "extreme_rmse_std",
        "extreme_hit_mean",
        "extreme_hit_std",
        "n_obs_mean",
    ]
    path.write_text(df[cols].to_markdown(index=False), encoding="utf-8")


def _improvement(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, g in summary_df.groupby("regime"):
        d = {r["model"]: r for _, r in g.iterrows()}
        if "Ours" not in d:
            continue
        ours = d["Ours"]
        for b in BASELINE_COMPARE:
            if b not in d:
                continue
            bb = d[b]
            rmse_imp = (bb["rmse_mean"] - ours["rmse_mean"]) / bb["rmse_mean"] * 100.0 if bb["rmse_mean"] != 0 else np.nan
            rows.append(
                {
                    "regime": regime,
                    "baseline_model": b,
                    "ours_rmse_mean": float(ours["rmse_mean"]),
                    "baseline_rmse_mean": float(bb["rmse_mean"]),
                    "improvement_rmse": float(rmse_imp),
                    "improvement_hit": float(ours["hit_ccy_mean"] - bb["hit_ccy_mean"]),
                    "improvement_pair_hit": float(ours["hit_pair_mean"] - bb["hit_pair_mean"]),
                }
            )
    return pd.DataFrame(rows)


def _plot_bar(summary_df: pd.DataFrame, metric: str, path: Path):
    regimes = [r for r in summary_df["regime"].drop_duplicates() if r != "all_test"]
    models = list(summary_df["model"].drop_duplicates())
    x = np.arange(len(regimes))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, m in enumerate(models):
        vals = []
        for r in regimes:
            s = summary_df[(summary_df["model"] == m) & (summary_df["regime"] == r)]
            vals.append(float(s[f"{metric}_mean"].iloc[0]) if len(s) else np.nan)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=m, color=("#d62728" if m == "Ours" else None))
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=20)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric.upper()} by stress regime")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_ours_improvement(imp_df: pd.DataFrame, path: Path):
    if imp_df.empty:
        return
    regimes = [r for r in imp_df["regime"].drop_duplicates() if r != "all_test"]
    baselines = list(imp_df["baseline_model"].drop_duplicates())
    x = np.arange(len(regimes))
    width = 0.8 / max(len(baselines), 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, b in enumerate(baselines):
        vals = []
        for r in regimes:
            s = imp_df[(imp_df["baseline_model"] == b) & (imp_df["regime"] == r)]
            vals.append(float(s["improvement_rmse"].iloc[0]) if len(s) else np.nan)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=b, alpha=0.9)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=20)
    ax.set_ylabel("RMSE improvement of Ours (%)")
    ax.set_title("Ours improvement over baselines by stress regime")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_normal_vs_stress_gap(summary_df: pd.DataFrame, path: Path):
    regimes = ["VIX_top10", "US2Y_shock_top10", "Oil_shock_top10", "composite_macro_shock_top10", "FX_move_top10"]
    rows = []
    for m, g in summary_df.groupby("model"):
        normal = g[g["regime"] == "normal"]
        if normal.empty:
            continue
        nrm = float(normal["rmse_mean"].iloc[0])
        for r in regimes:
            rr = g[g["regime"] == r]
            if rr.empty:
                continue
            rows.append({"model": m, "regime": r, "rmse_gap": float(rr["rmse_mean"].iloc[0] - nrm)})
    gdf = pd.DataFrame(rows)
    if gdf.empty:
        return
    pivot = gdf.pivot(index="model", columns="regime", values="rmse_gap")
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20)
    ax.set_title("RMSE gap: stress - normal")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _print_key_results(summary_df: pd.DataFrame, imp_df: pd.DataFrame) -> None:
    LOGGER.info("=" * 88)
    LOGGER.info("Stress-Regime Performance Summary (key rows)")
    key_regimes = ["all_test", "normal", "FX_move_top10", "VIX_top10", "US2Y_shock_top10", "Oil_shock_top10", "composite_macro_shock_top10"]
    key_models = ["Ours", "StaticGraph", "PureGraphFX", "NoGraph", "NoMacro", "LSTM", "MLP", "GAT"]
    cols = ["model", "regime", "n_obs_mean", "rmse_mean", "hit_ccy_mean", "hit_pair_mean"]
    view = summary_df[summary_df["regime"].isin(key_regimes) & summary_df["model"].isin(key_models)][cols].copy()
    if not view.empty:
        view = view.sort_values(["regime", "model"]).reset_index(drop=True)
        LOGGER.info("\n%s", view.to_string(index=False))

    zero_obs = summary_df[summary_df["n_obs_mean"].fillna(0) <= 0][["model", "regime", "n_obs_mean"]]
    if not zero_obs.empty:
        LOGGER.warning("Regimes with zero observations detected:")
        LOGGER.warning("\n%s", zero_obs.sort_values(["regime", "model"]).to_string(index=False))

    if not imp_df.empty:
        top = imp_df.sort_values("improvement_rmse", ascending=False).head(10)
        LOGGER.info("Top Ours RMSE improvements vs baselines:")
        LOGGER.info(
            "\n%s",
            top[["regime", "baseline_model", "improvement_rmse", "improvement_hit", "improvement_pair_hit"]].to_string(index=False),
        )
    LOGGER.info("=" * 88)


def main():
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if "FiLMHyGraph" in args.models:
        args.models = [m for m in args.models if m != "FiLMHyGraph"]
        LOGGER.info("Excluded FiLMHyGraph as requested.")

    out = load_or_create_results_dir(args.output_dir, ["metrics", "tables", "figures", "predictions", "configs"])
    pred_dir = Path(args.pred_dir) if args.pred_dir else out["predictions"]
    pred_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    cfg.file_path = args.data_path
    cfg.lookback = args.lookback
    data_bundle = prepare_data_split(cfg, split_mode="602020", data_path=args.data_path, macro_features=ALL_MACRO_FEATURES)
    stress_ctx = build_stress_context(
        cfg,
        data_path=args.data_path,
        lookback=args.lookback,
        quantile=args.stress_quantile,
        threshold_scope=args.threshold_scope,
    )
    save_json(out["tables"] / "stress_thresholds.json", stress_ctx.threshold_info)

    rows = []
    for model in args.models:
        for seed in args.seeds:
            pfile = _pred_file(pred_dir, model, args.lookback, seed)
            if pfile.exists():
                df_pred = pd.read_parquet(pfile)
            else:
                LOGGER.info("prediction missing -> training model=%s seed=%d", model, seed)
                train_params = {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "hidden": args.hidden,
                    "top_k": args.top_k,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "patience": args.patience,
                }
                model_obj, art = train_model(model, data_bundle, args.lookback, seed, train_params)
                pred, target, ddf = predict_model(model_obj, art, data_bundle, args.lookback)
                df_pred = _to_long(pred, target, ddf, model, args.lookback, seed, data_bundle["currency_names"])
                save_predictions(pfile, df_pred)
            pred, target, ccys, dates = _matrix_from_long(df_pred)
            metrics_rows = _eval_regimes(pred, target, ccys, dates, stress_ctx.stress_masks)
            for mr in metrics_rows:
                rows.append(
                    {
                        "model": model,
                        "display_name": DEFAULT_DISPLAY_NAMES.get(model, model),
                        "lookback": args.lookback,
                        "seed": seed,
                        **mr,
                    }
                )
            LOGGER.info("evaluated model=%s seed=%d", model, seed)

    raw_df = pd.DataFrame(rows).sort_values(["model", "seed", "regime"]).reset_index(drop=True)
    save_csv(out["metrics"] / "stress_metrics_raw.csv", raw_df)
    summary_df = _summarize(raw_df)
    save_csv(out["tables"] / "stress_metrics_summary.csv", summary_df)
    _summary_md(summary_df, out["tables"] / "stress_metrics_summary.md")
    imp_df = _improvement(summary_df)
    save_csv(out["tables"] / "ours_stress_improvement_vs_baselines.csv", imp_df)

    _plot_bar(summary_df, "rmse", out["figures"] / "stress_rmse_by_model.png")
    _plot_bar(summary_df, "hit_ccy", out["figures"] / "stress_hit_by_model.png")
    _plot_ours_improvement(imp_df, out["figures"] / "ours_improvement_by_stress.png")
    _plot_normal_vs_stress_gap(summary_df, out["figures"] / "normal_vs_stress_gap.png")
    _print_key_results(summary_df, imp_df)

    LOGGER.info("saved stress performance outputs under: %s", out["base"])


if __name__ == "__main__":
    main()
