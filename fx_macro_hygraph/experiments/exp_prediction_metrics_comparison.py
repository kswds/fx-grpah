import argparse
import logging
import os
import sys
from pathlib import Path

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

LOGGER = logging.getLogger("exp_prediction_metrics_comparison")


def _args():
    p = argparse.ArgumentParser(description="Prediction metrics comparison across models.")
    p.add_argument("--data-path", default=str(ROOT / "data" / "factor_daily_legacy.csv"))
    p.add_argument("--models", nargs="+", default=["MLP", "LSTM", "GRU", "Transformer", "GAT", "Ours"])
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--output-dir", default=str(ROOT / "results" / "model_prediction_comparison"))
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _to_long(pred, target, dates_df, model, lookback, seed, ccys):
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
                    "currency": c,
                    "pred": float(pred[t, i]),
                    "target": float(target[t, i]),
                }
            )
    return pd.DataFrame(rows)


def main():
    args = _args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    dirs = load_or_create_results_dir(args.output_dir, ["metrics", "tables", "predictions", "configs"])

    cfg = Config()
    cfg.file_path = args.data_path
    cfg.lookback = args.lookback
    data_bundle = prepare_data_split(cfg, data_path=args.data_path, macro_features=ALL_MACRO_FEATURES)
    train_params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "top_k": args.top_k,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
    }

    raw_rows = []
    for model in args.models:
        for seed in args.seeds:
            mdl, art = train_model(model, data_bundle, args.lookback, seed, train_params)
            pred, target, ddf = predict_model(mdl, art, data_bundle, args.lookback)
            m = compute_metrics(pred, target, data_bundle["n_ccy"], data_bundle["usd_idx"], q=0.90)
            mse = float(np.mean((pred - target) ** 2))
            row = {
                "model": model,
                "display_name": DEFAULT_DISPLAY_NAMES.get(model, model),
                "seed": seed,
                "lookback": args.lookback,
                "rmse": m["rmse"],
                "mse": mse,
                "mae": m["mae"],
                "extreme_rmse": m["extreme_rmse"],
                "hit_ratio": m["hit_ccy"],
                "extreme_hit_ratio": m["extreme_hit"],
                "hit_pair": m["hit_pair"],
            }
            raw_rows.append(row)
            save_predictions(dirs["predictions"] / f"{model}_L{args.lookback}_seed{seed}.parquet", _to_long(pred, target, ddf, model, args.lookback, seed, data_bundle["currency_names"]))
            save_json(dirs["configs"] / f"{model}_L{args.lookback}_seed{seed}.json", {"train_params": train_params, "best_epoch": art["best_epoch"], "best_val_mse": art["best_val_mse"]})
            LOGGER.info("done model=%s seed=%d rmse=%.6f hit=%.4f", model, seed, row["rmse"], row["hit_ratio"])

    raw_df = pd.DataFrame(raw_rows)
    save_csv(dirs["metrics"] / "prediction_metrics_raw.csv", raw_df)
    summary = raw_df.groupby(["model", "display_name", "lookback"], as_index=False).agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        extreme_rmse_mean=("extreme_rmse", "mean"),
        extreme_rmse_std=("extreme_rmse", "std"),
        hit_ratio_mean=("hit_ratio", "mean"),
        hit_ratio_std=("hit_ratio", "std"),
        extreme_hit_ratio_mean=("extreme_hit_ratio", "mean"),
        extreme_hit_ratio_std=("extreme_hit_ratio", "std"),
        hit_pair_mean=("hit_pair", "mean"),
        hit_pair_std=("hit_pair", "std"),
    )
    summary = summary.fillna(0.0).sort_values(["rmse_mean", "hit_ratio_mean"], ascending=[True, False])
    save_csv(dirs["tables"] / "prediction_metrics_summary.csv", summary)
    (dirs["tables"] / "prediction_metrics_summary.md").write_text(summary.to_markdown(index=False), encoding="utf-8")


if __name__ == "__main__":
    main()
