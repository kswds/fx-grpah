from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
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
from training import train_relational_model


ABLATION_MODELS = [
    ("foundation_nograph", "no_graph"),
    ("foundation_static", "static_graph"),
    ("foundation_relational", "static_plus_dynamic_graph"),
    ("oursmain", "oursmain"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Graph ablation study with oursmain reference: No Graph -> Static Graph -> Static + Dynamic Graph -> oursmain")
    parser.add_argument("--fx-data-path", default=str(DEFAULT_FX_DATA))
    parser.add_argument("--nonfx-data-path", default=str(DEFAULT_NONFX_DATA))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "graph_ablation_study"))
    parser.add_argument("--universe", default="core6", choices=["major3", "core6", "krw7"])
    parser.add_argument("--custom-currencies", nargs="*")
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


def build_prediction_comparison(prediction_rows: List[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(prediction_rows, ignore_index=True)
    wide = combined.pivot_table(
        index=["seed", "target_date", "input_end_date", "currency"],
        columns="ablation_stage",
        values=["pred", "target"],
        aggfunc="first",
    )
    wide.columns = [f"{left}_{right}" for left, right in wide.columns.to_flat_index()]
    wide = wide.reset_index()

    if "target_no_graph" in wide.columns:
        wide = wide.rename(columns={"target_no_graph": "target"})
    for col in ["target_static_graph", "target_static_plus_dynamic_graph", "target_oursmain"]:
        if col in wide.columns:
            wide = wide.drop(columns=col)

    stage_cols = ["pred_no_graph", "pred_static_graph", "pred_static_plus_dynamic_graph", "pred_oursmain"]
    for col in stage_cols:
        if col not in wide.columns:
            wide[col] = np.nan

    wide["delta_static_minus_nograph"] = wide["pred_static_graph"] - wide["pred_no_graph"]
    wide["delta_dynamic_minus_static"] = wide["pred_static_plus_dynamic_graph"] - wide["pred_static_graph"]
    wide["delta_dynamic_minus_nograph"] = wide["pred_static_plus_dynamic_graph"] - wide["pred_no_graph"]
    wide["delta_oursmain_minus_dynamic"] = wide["pred_oursmain"] - wide["pred_static_plus_dynamic_graph"]
    wide["delta_oursmain_minus_static"] = wide["pred_oursmain"] - wide["pred_static_graph"]
    wide["delta_oursmain_minus_nograph"] = wide["pred_oursmain"] - wide["pred_no_graph"]
    wide["abs_error_no_graph"] = np.abs(wide["pred_no_graph"] - wide["target"])
    wide["abs_error_static_graph"] = np.abs(wide["pred_static_graph"] - wide["target"])
    wide["abs_error_static_plus_dynamic_graph"] = np.abs(wide["pred_static_plus_dynamic_graph"] - wide["target"])
    wide["abs_error_oursmain"] = np.abs(wide["pred_oursmain"] - wide["target"])
    return wide.sort_values(["seed", "target_date", "currency"]).reset_index(drop=True)


def summarize_stage_transitions(pred_comp: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    transitions = [
        ("no_graph", "static_graph", "static_minus_nograph"),
        ("static_graph", "static_plus_dynamic_graph", "dynamic_minus_static"),
        ("no_graph", "static_plus_dynamic_graph", "dynamic_minus_nograph"),
        ("static_plus_dynamic_graph", "oursmain", "oursmain_minus_dynamic"),
        ("static_graph", "oursmain", "oursmain_minus_static"),
        ("no_graph", "oursmain", "oursmain_minus_nograph"),
    ]
    for seed, group in pred_comp.groupby("seed"):
        target = pd.to_numeric(group["target"], errors="coerce").to_numpy(dtype=float)
        for left, right, label in transitions:
            left_pred = pd.to_numeric(group[f"pred_{left}"], errors="coerce").to_numpy(dtype=float)
            right_pred = pd.to_numeric(group[f"pred_{right}"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(left_pred) & np.isfinite(right_pred) & np.isfinite(target)
            if not valid.any():
                continue
            left_err = np.abs(left_pred[valid] - target[valid])
            right_err = np.abs(right_pred[valid] - target[valid])
            rows.append(
                {
                    "seed": seed,
                    "transition": label,
                    "mean_abs_prediction_change": float(np.mean(np.abs(right_pred[valid] - left_pred[valid]))),
                    "prediction_correlation": float(np.corrcoef(left_pred[valid], right_pred[valid])[0, 1]) if valid.sum() > 1 else 1.0,
                    "mae_delta": float(np.mean(right_err - left_err)),
                    "improved_obs_ratio": float(np.mean(right_err < left_err)),
                    "worsened_obs_ratio": float(np.mean(right_err > left_err)),
                    "same_sign_ratio": float(np.mean(np.sign(left_pred[valid]) == np.sign(right_pred[valid]))),
                }
            )
    return pd.DataFrame(rows)


def summarize_transition_aggregate(transition_df: pd.DataFrame) -> pd.DataFrame:
    if transition_df.empty:
        return transition_df
    metric_cols = [
        "mean_abs_prediction_change",
        "prediction_correlation",
        "mae_delta",
        "improved_obs_ratio",
        "worsened_obs_ratio",
        "same_sign_ratio",
    ]
    agg = transition_df.groupby("transition", as_index=False)[metric_cols].agg(["mean", "std", "min", "max"])
    agg.columns = ["_".join([c for c in col if c]).strip("_") for col in agg.columns.to_flat_index()]
    return agg.rename(columns={"transition_": "transition"})


def write_ablation_markdown(metrics_agg: pd.DataFrame, transition_agg: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# Graph Ablation Study",
        "",
        "Stages:",
        "- `no_graph`: relational branch removed",
        "- `static_graph`: static graph only",
        "- `static_plus_dynamic_graph`: static + dynamic graph",
        "- `oursmain`: same full graph backbone as `static_plus_dynamic_graph`, but trained with the oursmain loss",
        "",
        "## Predictive Metrics",
        "",
        "| Stage | Mean Hit | Mean RMSE | Mean MAE | Mean Pairwise Hit | Mean IC | Mean LS Sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in metrics_agg.sort_values(["hit_ratio_mean", "rmse_mean"], ascending=[False, True]).iterrows():
        lines.append(
            "| {model} | {hit_ratio_mean:.4f} | {rmse_mean:.6f} | {mae_mean:.6f} | {pairwise_hit_mean:.4f} | {ic_mean:.4f} | {long_short_sharpe_mean:.4f} |".format(
                **row.to_dict()
            )
        )
    lines.extend(
        [
            "",
            "## Prediction Transitions",
            "",
            "| Transition | Mean Abs Prediction Change | Prediction Corr | MAE Delta | Improved Obs Ratio | Worsened Obs Ratio | Same Sign Ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in transition_agg.iterrows():
        lines.append(
            "| {transition} | {mean_abs_prediction_change_mean:.6f} | {prediction_correlation_mean:.4f} | {mae_delta_mean:.6f} | {improved_obs_ratio_mean:.4f} | {worsened_obs_ratio_mean:.4f} | {same_sign_ratio_mean:.4f} |".format(
                **row.to_dict()
            )
        )
    (out_dir / "ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    currency_names = resolve_currency_names(args.universe, args.custom_currencies)
    prepared = prepare_data(
        args.fx_data_path,
        args.nonfx_data_path,
        currency_names,
        args.lookback,
        args.split,
        include_regime_onehot=True,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    raw_rows: List[Dict[str, object]] = []
    prediction_rows: List[pd.DataFrame] = []

    for seed in args.seeds:
        args.seed = seed
        set_seed(seed)
        for model_name, stage_name in ABLATION_MODELS:
            run_dir = output_dir / f"{stage_name}_seed{seed}"
            result = train_relational_model(model_name, prepared, args, run_dir, args.device)
            metrics_row = dict(result.raw_metrics)
            metrics_row["base_model"] = metrics_row["model"]
            metrics_row["model"] = stage_name
            metrics_row["ablation_stage"] = stage_name
            raw_rows.append(metrics_row)

            pred_df = result.prediction_df.copy()
            pred_df["ablation_stage"] = stage_name
            prediction_rows.append(pred_df)
            print(f"[done] stage={stage_name} seed={seed} hit={metrics_row['hit_ratio']:.4f} rmse={metrics_row['rmse']:.6f}")

    raw_df = pd.DataFrame(raw_rows)
    save_dataframe(raw_df, output_dir / "tables" / "metrics_summary.csv")
    summarize_metrics(raw_df, output_dir / "tables")

    pred_comp = build_prediction_comparison(prediction_rows)
    save_dataframe(pred_comp, output_dir / "tables" / "prediction_comparison.csv")

    transition_detail = summarize_stage_transitions(pred_comp)
    transition_agg = summarize_transition_aggregate(transition_detail)
    save_dataframe(transition_detail, output_dir / "tables" / "transition_detail.csv")
    save_dataframe(transition_agg, output_dir / "tables" / "transition_aggregate.csv")

    metrics_agg = pd.read_csv(output_dir / "tables" / "seed_metrics_aggregate.csv")
    write_ablation_markdown(metrics_agg, transition_agg, output_dir / "tables")
    print(f"saved to {output_dir}")


if __name__ == "__main__":
    main()
