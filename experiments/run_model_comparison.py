from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import DEFAULT_FX_DATA, DEFAULT_NONFX_DATA, OURSMAIN_DEFAULTS
from data_pipeline import prepare_data, resolve_currency_names, set_seed
from metrics import save_dataframe, summarize_metrics
from training import train_baseline_model, train_relational_model


RELATIONAL_MODELS = {"oursmain", "foundation_relational", "foundation_nograph", "foundation_static"}
BASELINE_MODELS = {"mlp", "lstm", "gru", "gnn"}


def parse_args():
    parser = argparse.ArgumentParser(description="Shared FX model comparison pipeline")
    parser.add_argument("--fx-data-path", default=str(DEFAULT_FX_DATA))
    parser.add_argument("--nonfx-data-path", default=str(DEFAULT_NONFX_DATA))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "model_comparison"))
    parser.add_argument("--universe", default="core6", choices=["major3", "core6", "krw7"])
    parser.add_argument("--custom-currencies", nargs="*")
    parser.add_argument("--models", nargs="+", default=["oursmain", "foundation_relational", "mlp", "lstm", "gru", "gnn"])
    parser.add_argument("--lookback", type=int, default=OURSMAIN_DEFAULTS["lookback"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--split", nargs=3, type=float, default=[0.6, 0.2, 0.2])
    parser.add_argument("--hidden", type=int, default=OURSMAIN_DEFAULTS["hidden"])
    parser.add_argument("--top-k", type=int, default=OURSMAIN_DEFAULTS["top_k"])
    parser.add_argument("--graph-rank", type=int, default=OURSMAIN_DEFAULTS["graph_rank"])
    parser.add_argument("--dropout", type=float, default=OURSMAIN_DEFAULTS["dropout"])
    parser.add_argument("--edge-dropout", type=float, default=OURSMAIN_DEFAULTS["edge_dropout"])
    parser.add_argument("--spectral-bound", type=float, default=OURSMAIN_DEFAULTS["spectral_bound"])
    parser.add_argument("--lr", type=float, default=OURSMAIN_DEFAULTS["lr"])
    parser.add_argument("--weight-decay", type=float, default=OURSMAIN_DEFAULTS["weight_decay"])
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lambda-dir", type=float, default=OURSMAIN_DEFAULTS["lambda_dir"])
    parser.add_argument("--lambda-rank", type=float, default=OURSMAIN_DEFAULTS["lambda_rank"])
    parser.add_argument("--lambda-component", type=float, default=OURSMAIN_DEFAULTS["lambda_component"])
    parser.add_argument("--lambda-smooth", type=float, default=OURSMAIN_DEFAULTS["lambda_smooth"])
    parser.add_argument("--lambda-static", type=float, default=OURSMAIN_DEFAULTS["lambda_static"])
    parser.add_argument("--lambda-sparse", type=float, default=OURSMAIN_DEFAULTS["lambda_sparse"])
    parser.add_argument("--lambda-spectral", type=float, default=OURSMAIN_DEFAULTS["lambda_spectral"])
    parser.add_argument("--small-return-quantile", type=float, default=OURSMAIN_DEFAULTS["small_return_quantile"])
    parser.add_argument("--component-gate-type", default=OURSMAIN_DEFAULTS["component_gate_type"], choices=["sigmoid", "softmax"])
    parser.add_argument("--selection-metric", default="mse_hit", choices=["mse", "hit", "mse_hit", "sharpe"])
    parser.add_argument("--hit-alpha", type=float, default=0.01)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    currency_names = resolve_currency_names(args.universe, args.custom_currencies)
    include_regime = any(m == "oursmain" for m in args.models)
    prepared = prepare_data(args.fx_data_path, args.nonfx_data_path, currency_names, args.lookback, args.split, include_regime_onehot=include_regime, start_date=args.start_date, end_date=args.end_date)
    raw_rows = []
    for seed in args.seeds:
        args.seed = seed
        set_seed(seed)
        for model_name in args.models:
            run_dir = output_dir / f"{model_name}_seed{seed}"
            if model_name in RELATIONAL_MODELS:
                result = train_relational_model(model_name, prepared, args, run_dir, args.device)
            elif model_name in BASELINE_MODELS:
                result = train_baseline_model(model_name, prepared, args, run_dir, args.device)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            raw_rows.append(result.raw_metrics)
            print(f"[done] model={model_name} seed={seed} hit={result.raw_metrics['hit_ratio']:.4f} rmse={result.raw_metrics['rmse']:.6f}")
    raw_df = pd.DataFrame(raw_rows)
    save_dataframe(raw_df, output_dir / "tables" / "metrics_summary.csv")
    summarize_metrics(raw_df, output_dir / "tables")
    print(f"saved to {output_dir}")


if __name__ == "__main__":
    main()
