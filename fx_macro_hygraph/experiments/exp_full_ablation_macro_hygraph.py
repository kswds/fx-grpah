import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from config import Config
try:
    from exp_utils import (
        ALL_MACRO_FEATURES,
        DEFAULT_DISPLAY_NAMES,
        aggregate_metrics,
        compute_metrics,
        config_to_dict,
        get_device,
        load_or_create_results_dir,
        predict_model,
        prepare_data_split,
        save_csv,
        save_json,
        save_predictions,
        set_seed,
        significance_tests,
        train_model,
    )
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(__file__))
    from exp_utils import (
        ALL_MACRO_FEATURES,
        DEFAULT_DISPLAY_NAMES,
        aggregate_metrics,
        compute_metrics,
        config_to_dict,
        get_device,
        load_or_create_results_dir,
        predict_model,
        prepare_data_split,
        save_csv,
        save_json,
        save_predictions,
        set_seed,
        significance_tests,
        train_model,
    )


LOGGER = logging.getLogger("exp_full_ablation_macro_hygraph")
METRIC_COLS = ["rmse", "mae", "hit_ccy", "hit_pair", "extreme_rmse", "extreme_hit", "ic", "sharpe"]


def _parse_args():
    p = argparse.ArgumentParser(
        description="Full ablation for MACRO-HyGraph (60/20/20 split).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-path", default=str(ROOT / "data" / "factor_daily_legacy.csv"))
    p.add_argument("--split", default="602020", choices=["602020"])
    p.add_argument("--lookbacks", nargs="+", default=["1", "2", "5", "20"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument(
        "--models",
        nargs="+",
        default=["MLP", "LSTM", "GAT", "NoGraph", "NoMacro", "StaticGraph", "PureGraphFX", "FiLMHyGraph", "Ours"],
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--output-dir", default=str(ROOT / "results" / "model_ablation"))
    p.add_argument("--save-pdf", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _as_lookback_token(lb):
    s = str(lb)
    return int(s) if s.isdigit() else s


def _long_predictions(pred, target, dates_df, model, lookback, seed, ccys):
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
                    "seed": int(seed),
                    "currency": ccy,
                    "pred": float(pred[t, i]),
                    "target": float(target[t, i]),
                }
            )
    return pd.DataFrame(rows)


def _summary_to_markdown(df: pd.DataFrame, out_path: Path):
    view = df.copy()
    for m in METRIC_COLS:
        view[m] = view.apply(lambda r: f"{r[f'{m}_mean']:.6f} ± {r[f'{m}_std']:.6f}", axis=1)
    cols = ["model", "display_name", "lookback"] + METRIC_COLS
    view = view[cols].sort_values(["lookback", "model"]).reset_index(drop=True)
    out_path.write_text(view.to_markdown(index=False), encoding="utf-8")


def _ours_improvement(summary_df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    by_lb = {}
    for _, r in summary_df.iterrows():
        by_lb.setdefault(str(r["lookback"]), {})[r["model"]] = r
    for lb, d in by_lb.items():
        if "Ours" not in d:
            continue
        ours = d["Ours"]
        for model, row in d.items():
            if model == "Ours":
                continue
            base_rmse = float(row["rmse_mean"])
            if abs(base_rmse) < 1e-12:
                imp = np.nan
            else:
                imp = (base_rmse - float(v5["rmse_mean"])) / base_rmse * 100.0
            recs.append(
                {
                    "lookback": lb,
                    "baseline_model": model,
                    "baseline_display_name": row["display_name"],
                    "ours_rmse_mean": float(ours["rmse_mean"]),
                    "baseline_rmse_mean": base_rmse,
                    "improvement_rmse_vs_ours": float(imp),
                }
            )
    return pd.DataFrame(recs)


def main():
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    set_seed(42)
    device = get_device()
    LOGGER.info("Device: %s", device)

    dirs = load_or_create_results_dir(
        args.output_dir,
        [
            "metrics",
            "predictions",
            "configs",
            "tables",
            "figures",
            "diagnostics",
            "checkpoints",
        ],
    )

    cfg = Config()
    cfg.file_path = args.data_path
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.hidden = args.hidden
    cfg.hybrid_hidden = args.hidden
    cfg.top_k = args.top_k
    cfg.lr = args.lr
    cfg.early_stopping_patience = args.patience

    data_bundle = prepare_data_split(cfg, split_mode=args.split, data_path=args.data_path, macro_features=ALL_MACRO_FEATURES)
    LOGGER.info(
        "Samples train/val/test = %d/%d/%d",
        data_bundle["train_end"],
        data_bundle["val_end"] - data_bundle["train_end"],
        data_bundle["n"] - data_bundle["val_end"],
    )

    run_rows = []
    pred_records = {}
    train_params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "top_k": args.top_k,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
    }

    total = len(args.models) * len(args.lookbacks) * len(args.seeds)
    done = 0
    for model in args.models:
        for lb in args.lookbacks:
            lb_token = _as_lookback_token(lb)
            for seed in args.seeds:
                done += 1
                LOGGER.info("[%d/%d] model=%s lookback=%s seed=%s", done, total, model, lb, seed)
                model_obj, art = train_model(model, data_bundle, lb_token, seed, train_params, device=device)
                pred, target, dates_df = predict_model(model_obj, art, data_bundle, lb_token, device=device)
                metrics = compute_metrics(pred, target, data_bundle["n_ccy"], data_bundle["usd_idx"])

                row = {
                    "model": model,
                    "display_name": DEFAULT_DISPLAY_NAMES.get(model, model),
                    "lookback": str(lb),
                    "seed": int(seed),
                    "best_val_mse": float(art["best_val_mse"]),
                    "best_epoch": int(art["best_epoch"]),
                }
                row.update(metrics)
                run_rows.append(row)
                pred_records[(model, str(lb), int(seed))] = (pred.copy(), target.copy())

                pred_df = _long_predictions(pred, target, dates_df, model, lb, seed, data_bundle["currency_names"])
                save_predictions(dirs["predictions"] / f"{model}_L{lb}_seed{seed}.parquet", pred_df)

                pred_matrix = pd.DataFrame(pred, columns=[f"pred_{c}" for c in data_bundle["currency_names"]])
                tgt_matrix = pd.DataFrame(target, columns=[f"target_{c}" for c in data_bundle["currency_names"]])
                meta = dates_df.reset_index(drop=True).copy()
                save_predictions(
                    dirs["predictions"] / f"{model}_L{lb}_seed{seed}_matrix.parquet",
                    pd.concat([meta, pred_matrix, tgt_matrix], axis=1),
                )

                cfg_payload = {
                    "run_config": {
                        "model": model,
                        "lookback": str(lb),
                        "seed": int(seed),
                        **train_params,
                        "split": args.split,
                        "data_path": args.data_path,
                    },
                    "data_meta": {
                        "n_ccy": data_bundle["n_ccy"],
                        "usd_idx": data_bundle["usd_idx"],
                        "currencies": data_bundle["currency_names"],
                        "macro_features": data_bundle["macro_features"],
                        "train_samples": data_bundle["train_end"],
                        "val_samples": data_bundle["val_end"] - data_bundle["train_end"],
                        "test_samples": data_bundle["n"] - data_bundle["val_end"],
                    },
                    "model_selection": {"best_val_mse": float(art["best_val_mse"]), "best_epoch": int(art["best_epoch"])},
                    "metrics": metrics,
                    "base_config": config_to_dict(cfg),
                }
                save_json(dirs["configs"] / f"{model}_L{lb}_seed{seed}.json", cfg_payload)

                if model == "Ours" and hasattr(model_obj, "state_dict"):
                    ckpt_payload = {
                        "model_name": model,
                        "lookback": str(lb),
                        "seed": int(seed),
                        "state_dict": model_obj.state_dict(),
                        "run_config": cfg_payload["run_config"],
                    }
                    torch.save(ckpt_payload, dirs["checkpoints"] / f"Ours_L{lb}_seed{seed}.pt")

                LOGGER.info(
                    "test rmse=%.6f mae=%.6f hit_ccy=%.4f hit_pair=%.4f ic=%.4f sharpe=%.4f",
                    metrics["rmse"],
                    metrics["mae"],
                    metrics["hit_ccy"],
                    metrics["hit_pair"],
                    metrics["ic"],
                    metrics["sharpe"],
                )

    raw_df = pd.DataFrame(run_rows)
    save_csv(dirs["metrics"] / "full_ablation_raw.csv", raw_df)

    summary = aggregate_metrics(raw_df, METRIC_COLS, ["model", "display_name", "lookback"])
    save_csv(dirs["tables"] / "full_ablation_summary.csv", summary)
    _summary_to_markdown(summary, dirs["tables"] / "full_ablation_summary.md")

    imp = _ours_improvement(summary)
    save_csv(dirs["tables"] / "ours_improvement_vs_baselines.csv", imp)

    sig_df = significance_tests(
        pred_records,
        comparisons=[("Ours", "FiLMHyGraph"), ("Ours", "StaticGraph"), ("Ours", "NoMacro"), ("Ours", "LSTM"), ("Ours", "MLP")],
    )
    save_csv(dirs["tables"] / "significance_tests.csv", sig_df)

    LOGGER.info("Saved raw metrics: %s", dirs["metrics"] / "full_ablation_raw.csv")
    LOGGER.info("Saved summary: %s", dirs["tables"] / "full_ablation_summary.csv")
    LOGGER.info("Saved improvements: %s", dirs["tables"] / "ours_improvement_vs_baselines.csv")
    LOGGER.info("Saved significance tests: %s", dirs["tables"] / "significance_tests.csv")


if __name__ == "__main__":
    main()
