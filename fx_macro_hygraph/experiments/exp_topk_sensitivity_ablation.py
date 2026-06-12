import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


LOGGER = logging.getLogger("exp_topk_sensitivity_ablation")


def _args():
    p = argparse.ArgumentParser(description="Top-k sensitivity ablation for dynamic graph models.")
    p.add_argument("--data-path", default=str(ROOT / "data" / "factor_daily_legacy.csv"))
    p.add_argument("--models", nargs="+", default=["Ours"])
    p.add_argument("--topk-values", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--output-dir", default=str(ROOT / "results" / "topk_sensitivity_ablation"))
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _to_long(pred, target, dates_df, model, lookback, seed, top_k, ccys):
    rows = []
    for t in range(pred.shape[0]):
        for i, c in enumerate(ccys):
            rows.append(
                {
                    "target_date": dates_df.iloc[t]["target_date"],
                    "input_end_date": dates_df.iloc[t]["input_end_date"],
                    "model": model,
                    "lookback": lookback,
                    "seed": seed,
                    "top_k": top_k,
                    "currency": c,
                    "pred": float(pred[t, i]),
                    "target": float(target[t, i]),
                }
            )
    return pd.DataFrame(rows)


def _plot_topk(summary: pd.DataFrame, out_dir: Path):
    if summary.empty:
        return
    for metric in ["rmse_mean", "hit_ccy_mean", "hit_pair_mean", "sharpe_mean"]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for model, sub in summary.groupby("model"):
            s = sub.sort_values("top_k")
            ax.plot(s["top_k"], s[metric], marker="o", label=model)
        ax.set_xlabel("top_k")
        ax.set_ylabel(metric.replace("_mean", ""))
        ax.set_title(f"Top-k sensitivity: {metric.replace('_mean', '')}")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"topk_sensitivity_{metric.replace('_mean', '')}.png", dpi=150)
        plt.close(fig)


def main():
    args = _args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    dirs = load_or_create_results_dir(
        args.output_dir,
        ["metrics", "tables", "predictions", "configs", "figures"],
    )

    cfg = Config()
    cfg.file_path = args.data_path
    cfg.lookback = args.lookback
    cfg.hidden = args.hidden
    cfg.hybrid_hidden = args.hidden

    data_bundle = prepare_data_split(cfg, split_mode="602020", data_path=args.data_path, macro_features=ALL_MACRO_FEATURES)
    max_valid_topk = max(1, data_bundle["n_ccy"] - 1)
    topk_values = [k for k in args.topk_values if 1 <= k <= max_valid_topk]
    if len(topk_values) < len(args.topk_values):
        LOGGER.warning("Some top_k values were out of range and ignored. valid range: [1, %d]", max_valid_topk)

    raw_rows = []
    for model in args.models:
        for top_k in topk_values:
            train_params = {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "hidden": args.hidden,
                "top_k": top_k,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "patience": args.patience,
            }
            for seed in args.seeds:
                LOGGER.info("run model=%s top_k=%d seed=%d", model, top_k, seed)
                mdl, art = train_model(model, data_bundle, args.lookback, seed, train_params)
                pred, target, ddf = predict_model(mdl, art, data_bundle, args.lookback)
                m = compute_metrics(pred, target, data_bundle["n_ccy"], data_bundle["usd_idx"])
                row = {
                    "model": model,
                    "display_name": DEFAULT_DISPLAY_NAMES.get(model, model),
                    "lookback": args.lookback,
                    "seed": seed,
                    "top_k": top_k,
                    "best_epoch": int(art["best_epoch"]),
                    "best_val_mse": float(art["best_val_mse"]),
                    **m,
                }
                raw_rows.append(row)
                save_predictions(
                    dirs["predictions"] / f"{model}_L{args.lookback}_K{top_k}_seed{seed}.parquet",
                    _to_long(pred, target, ddf, model, args.lookback, seed, top_k, data_bundle["currency_names"]),
                )
                save_json(
                    dirs["configs"] / f"{model}_L{args.lookback}_K{top_k}_seed{seed}.json",
                    {"train_params": train_params, "best_epoch": art["best_epoch"], "best_val_mse": art["best_val_mse"]},
                )

    raw = pd.DataFrame(raw_rows)
    save_csv(dirs["metrics"] / "topk_sensitivity_raw.csv", raw)

    summary = (
        raw.groupby(["model", "display_name", "lookback", "top_k"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            hit_ccy_mean=("hit_ccy", "mean"),
            hit_ccy_std=("hit_ccy", "std"),
            hit_pair_mean=("hit_pair", "mean"),
            hit_pair_std=("hit_pair", "std"),
            extreme_rmse_mean=("extreme_rmse", "mean"),
            extreme_rmse_std=("extreme_rmse", "std"),
            extreme_hit_mean=("extreme_hit", "mean"),
            extreme_hit_std=("extreme_hit", "std"),
            ic_mean=("ic", "mean"),
            ic_std=("ic", "std"),
            sharpe_mean=("sharpe", "mean"),
            sharpe_std=("sharpe", "std"),
        )
        .fillna(0.0)
        .sort_values(["model", "top_k"])
    )
    save_csv(dirs["tables"] / "topk_sensitivity_summary.csv", summary)
    (dirs["tables"] / "topk_sensitivity_summary.md").write_text(summary.to_markdown(index=False), encoding="utf-8")

    # Best top_k per model (primary metric: rmse_mean; tie-breaker: hit_ccy_mean)
    best_rows = []
    for model, sub in summary.groupby("model"):
        s = sub.sort_values(["rmse_mean", "hit_ccy_mean"], ascending=[True, False]).iloc[0]
        best_rows.append(dict(s))
    best_df = pd.DataFrame(best_rows).sort_values("rmse_mean")
    save_csv(dirs["tables"] / "topk_best_by_model.csv", best_df)

    # Improvement table relative to a reference top_k (default 6 if available else smallest)
    imp_rows = []
    for model, sub in summary.groupby("model"):
        ref_k = 6 if (sub["top_k"] == 6).any() else int(sub["top_k"].min())
        ref = sub[sub["top_k"] == ref_k].iloc[0]
        for _, r in sub.iterrows():
            imp_rows.append(
                {
                    "model": model,
                    "top_k": int(r["top_k"]),
                    "ref_top_k": int(ref_k),
                    "rmse_improvement_pct_vs_ref": float((ref["rmse_mean"] - r["rmse_mean"]) / (ref["rmse_mean"] + 1e-12) * 100.0),
                    "hit_ccy_delta_vs_ref": float(r["hit_ccy_mean"] - ref["hit_ccy_mean"]),
                    "hit_pair_delta_vs_ref": float(r["hit_pair_mean"] - ref["hit_pair_mean"]),
                    "sharpe_delta_vs_ref": float(r["sharpe_mean"] - ref["sharpe_mean"]),
                }
            )
    imp_df = pd.DataFrame(imp_rows).sort_values(["model", "top_k"])
    save_csv(dirs["tables"] / "topk_improvement_vs_reference.csv", imp_df)

    _plot_topk(summary, dirs["figures"])
    LOGGER.info("Saved top-k sensitivity results to: %s", args.output_dir)


if __name__ == "__main__":
    main()

