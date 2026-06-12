import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from config import Config

try:
    from exp_utils import (
        ALL_MACRO_FEATURES,
        compute_metrics,
        config_to_dict,
        get_device,
        predict_model,
        prepare_data_split,
        save_json,
        save_predictions,
        set_seed,
        train_model,
    )
    from ablation_utils import (
        COMPONENT_MODELS,
        COMPONENT_TO_INTERNAL,
        DISPLAY_NAMES,
        component_improvement,
        ensure_component_dirs,
        long_prediction_df,
        save_component_outputs,
        summarize_component_metrics,
    )
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(__file__))
    from exp_utils import (
        ALL_MACRO_FEATURES,
        compute_metrics,
        config_to_dict,
        get_device,
        predict_model,
        prepare_data_split,
        save_json,
        save_predictions,
        set_seed,
        train_model,
    )
    from ablation_utils import (
        COMPONENT_MODELS,
        COMPONENT_TO_INTERNAL,
        DISPLAY_NAMES,
        component_improvement,
        ensure_component_dirs,
        long_prediction_df,
        save_component_outputs,
        summarize_component_metrics,
    )


LOGGER = logging.getLogger("exp_component_ablation")


def _args():
    p = argparse.ArgumentParser(
        description="Component ablation for MACRO-HyGraph: NoMacro/NoGraph/StaticGraph/NoDirect/Ours",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-path", default=str(ROOT / "data" / "factor_daily_legacy.csv"))
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--models", nargs="+", default=COMPONENT_MODELS)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--output-dir", default=str(ROOT / "results" / "model_ablation"))
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main():
    args = _args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    set_seed(42)
    device = get_device()
    LOGGER.info("Device: %s", device)

    dirs = ensure_component_dirs(args.output_dir)

    cfg = Config()
    cfg.file_path = args.data_path
    cfg.lookback = args.lookback
    cfg.hidden = args.hidden
    cfg.hybrid_hidden = args.hidden
    cfg.top_k = args.top_k
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.early_stopping_patience = args.patience

    data_bundle = prepare_data_split(cfg, split_mode="602020", data_path=args.data_path, macro_features=ALL_MACRO_FEATURES)
    LOGGER.info(
        "Samples train/val/test = %d/%d/%d",
        data_bundle["train_end"],
        data_bundle["val_end"] - data_bundle["train_end"],
        data_bundle["n"] - data_bundle["val_end"],
    )

    train_params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "top_k": args.top_k,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
    }

    run_rows = []
    total = len(args.models) * len(args.seeds)
    done = 0
    for c_model in args.models:
        if c_model not in COMPONENT_TO_INTERNAL:
            raise ValueError(f"Unsupported component model: {c_model}. allowed={list(COMPONENT_TO_INTERNAL.keys())}")
        i_model = COMPONENT_TO_INTERNAL[c_model]
        for seed in args.seeds:
            done += 1
            LOGGER.info("[%d/%d] component_model=%s internal=%s seed=%d", done, total, c_model, i_model, seed)
            model_obj, art = train_model(i_model, data_bundle, args.lookback, seed, train_params, device=device)
            pred, target, dates_df = predict_model(model_obj, art, data_bundle, args.lookback, device=device)
            metrics = compute_metrics(pred, target, data_bundle["n_ccy"], data_bundle["usd_idx"])
            run_rows.append(
                {
                    "model": c_model,
                    "display_name": DISPLAY_NAMES[c_model],
                    "internal_model": i_model,
                    "lookback": str(args.lookback),
                    "seed": int(seed),
                    "best_val_mse": float(art["best_val_mse"]),
                    "best_epoch": int(art["best_epoch"]),
                    **metrics,
                }
            )
            pred_df = long_prediction_df(pred, target, dates_df, c_model, str(args.lookback), seed, data_bundle["currency_names"])
            save_predictions(dirs["predictions"] / f"{c_model}_L{args.lookback}_seed{seed}.parquet", pred_df)
            save_json(
                dirs["configs"] / f"{c_model}_L{args.lookback}_seed{seed}.json",
                {
                    "component_model": c_model,
                    "internal_model": i_model,
                    "seed": int(seed),
                    "lookback": int(args.lookback),
                    "train_params": train_params,
                    "best_epoch": int(art["best_epoch"]),
                    "best_val_mse": float(art["best_val_mse"]),
                    "base_config": config_to_dict(cfg),
                },
            )

            # Save checkpoints for all component runs for reproducibility.
            if hasattr(model_obj, "state_dict"):
                torch.save(
                    {
                        "component_model": c_model,
                        "internal_model": i_model,
                        "lookback": int(args.lookback),
                        "seed": int(seed),
                        "state_dict": model_obj.state_dict(),
                    },
                    dirs["checkpoints"] / f"{c_model}_L{args.lookback}_seed{seed}.pt",
                )

            LOGGER.info(
                "test rmse=%.6f mae=%.6f hit=%.4f pair=%.4f extreme_rmse=%.6f extreme_hit=%.4f",
                metrics["rmse"],
                metrics["mae"],
                metrics["hit_ccy"],
                metrics["hit_pair"],
                metrics["extreme_rmse"],
                metrics["extreme_hit"],
            )

    raw_df = pd.DataFrame(run_rows).sort_values(["model", "seed"]).reset_index(drop=True)
    summary_df = summarize_component_metrics(raw_df)
    imp_df = component_improvement(summary_df)
    save_component_outputs(dirs, raw_df, summary_df, imp_df)

    LOGGER.info("saved: %s", dirs["metrics"] / "component_ablation_raw.csv")
    LOGGER.info("saved: %s", dirs["tables"] / "component_ablation_summary.csv")
    LOGGER.info("saved: %s", dirs["tables"] / "ours_component_improvement.csv")


if __name__ == "__main__":
    main()

