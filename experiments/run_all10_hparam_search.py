from __future__ import annotations

import argparse
import itertools
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import DEFAULT_FX_DATA, DEFAULT_NONFX_DATA, OURSMAIN_DEFAULTS
from data_pipeline import prepare_data, resolve_currency_names, set_seed
from metrics import save_dataframe
from training import train_relational_model


RELATIONAL_MODELS = {"oursmain", "foundation_relational", "foundation_nograph", "foundation_static"}
DEFAULT_ALL10 = ["EUR", "JPY", "GBP", "CAD", "AUD", "KRW", "CHF", "NZD", "SEK", "NOK"]
MAXIMIZE_METRICS = {
    "hit_ratio",
    "non_tiny_hit_ratio",
    "extreme_hit_ratio",
    "pairwise_hit",
    "ic",
    "long_short_sharpe",
    "long_short_sortino",
    "cumulative_return",
}
MINIMIZE_METRICS = {"rmse", "mae", "max_drawdown"}


def parse_args():
    parser = argparse.ArgumentParser(description="All-10 FX hyperparameter search for relational models")
    parser.add_argument("--fx-data-path", default=str(DEFAULT_FX_DATA))
    parser.add_argument("--nonfx-data-path", default=str(DEFAULT_NONFX_DATA))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "all10_hparam_search"))
    parser.add_argument("--universe", default="core6", choices=["major3", "core6", "krw7"])
    parser.add_argument("--custom-currencies", nargs="*", default=DEFAULT_ALL10)
    parser.add_argument("--models", nargs="+", default=["oursmain", "foundation_relational"])
    parser.add_argument("--lookback", type=int, default=OURSMAIN_DEFAULTS["lookback"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split", nargs=3, type=float, default=[0.6, 0.2, 0.2])
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--selection-metric", default="mse_hit", choices=["mse", "hit", "mse_hit", "sharpe"])
    parser.add_argument("--hit-alpha", type=float, default=0.01)
    parser.add_argument("--component-gate-type", default=OURSMAIN_DEFAULTS["component_gate_type"], choices=["sigmoid", "softmax"])
    parser.add_argument("--lambda-component", type=float, default=OURSMAIN_DEFAULTS["lambda_component"])

    parser.add_argument("--hidden-grid", nargs="+", type=int, default=[OURSMAIN_DEFAULTS["hidden"]])
    parser.add_argument("--top-k-grid", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7])
    parser.add_argument("--graph-rank-grid", nargs="+", type=int, default=[OURSMAIN_DEFAULTS["graph_rank"]])
    parser.add_argument("--dropout-grid", nargs="+", type=float, default=[OURSMAIN_DEFAULTS["dropout"]])
    parser.add_argument("--edge-dropout-grid", nargs="+", type=float, default=[OURSMAIN_DEFAULTS["edge_dropout"]])
    parser.add_argument("--spectral-bound-grid", nargs="+", type=float, default=[1.0])
    parser.add_argument("--lr-grid", nargs="+", type=float, default=[OURSMAIN_DEFAULTS["lr"]])
    parser.add_argument("--weight-decay-grid", nargs="+", type=float, default=[OURSMAIN_DEFAULTS["weight_decay"]])
    parser.add_argument("--lambda-dir-grid", nargs="+", type=float, default=[OURSMAIN_DEFAULTS["lambda_dir"]])
    parser.add_argument("--lambda-rank-grid", nargs="+", type=float, default=[OURSMAIN_DEFAULTS["lambda_rank"]])
    parser.add_argument("--lambda-smooth-grid", nargs="+", type=float, default=[OURSMAIN_DEFAULTS["lambda_smooth"]])
    parser.add_argument("--lambda-static-grid", nargs="+", type=float, default=[OURSMAIN_DEFAULTS["lambda_static"]])
    parser.add_argument("--lambda-sparse-grid", nargs="+", type=float, default=[OURSMAIN_DEFAULTS["lambda_sparse"]])
    parser.add_argument("--lambda-spectral-grid", nargs="+", type=float, default=[OURSMAIN_DEFAULTS["lambda_spectral"]])
    parser.add_argument("--small-return-quantile-grid", nargs="+", type=float, default=[OURSMAIN_DEFAULTS["small_return_quantile"]])

    parser.add_argument("--rank-metric", default="hit_ratio", choices=sorted(MAXIMIZE_METRICS | MINIMIZE_METRICS))
    parser.add_argument("--top-n-report", type=int, default=10)
    return parser.parse_args()


def iter_trial_configs(args) -> Iterable[Dict[str, float]]:
    keys = [
        "hidden",
        "top_k",
        "graph_rank",
        "dropout",
        "edge_dropout",
        "spectral_bound",
        "lr",
        "weight_decay",
        "lambda_dir",
        "lambda_rank",
        "lambda_smooth",
        "lambda_static",
        "lambda_sparse",
        "lambda_spectral",
        "small_return_quantile",
    ]
    values = [
        args.hidden_grid,
        args.top_k_grid,
        args.graph_rank_grid,
        args.dropout_grid,
        args.edge_dropout_grid,
        args.spectral_bound_grid,
        args.lr_grid,
        args.weight_decay_grid,
        args.lambda_dir_grid,
        args.lambda_rank_grid,
        args.lambda_smooth_grid,
        args.lambda_static_grid,
        args.lambda_sparse_grid,
        args.lambda_spectral_grid,
        args.small_return_quantile_grid,
    ]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def build_trial_args(base_args, model_name: str, seed: int, trial_config: Dict[str, float]):
    params = deepcopy(vars(base_args))
    params.update(trial_config)
    params["seed"] = seed
    params["model_name"] = model_name
    return SimpleNamespace(**params)


def trial_score(metric_name: str, value: float) -> float:
    if metric_name in MAXIMIZE_METRICS:
        return -float(value)
    if metric_name in MINIMIZE_METRICS:
        return float(value)
    raise ValueError(f"Unsupported rank metric: {metric_name}")


def aggregate_trials(raw_df: pd.DataFrame) -> pd.DataFrame:
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
    hp_cols = [
        "model",
        "trial_id",
        "hidden",
        "top_k",
        "graph_rank",
        "dropout",
        "edge_dropout",
        "spectral_bound",
        "lr",
        "weight_decay",
        "lambda_dir",
        "lambda_rank",
        "lambda_smooth",
        "lambda_static",
        "lambda_sparse",
        "lambda_spectral",
        "small_return_quantile",
        "universe",
        "lookback",
    ]
    agg = raw_df.groupby(hp_cols, as_index=False)[metric_cols].agg(["mean", "std", "min", "max"])
    agg.columns = ["_".join([c for c in col if c]).strip("_") for col in agg.columns.to_flat_index()]
    return agg.rename(
        columns={
            "model_": "model",
            "trial_id_": "trial_id",
            "hidden_": "hidden",
            "top_k_": "top_k",
            "graph_rank_": "graph_rank",
            "dropout_": "dropout",
            "edge_dropout_": "edge_dropout",
            "spectral_bound_": "spectral_bound",
            "lr_": "lr",
            "weight_decay_": "weight_decay",
            "lambda_dir_": "lambda_dir",
            "lambda_rank_": "lambda_rank",
            "lambda_smooth_": "lambda_smooth",
            "lambda_static_": "lambda_static",
            "lambda_sparse_": "lambda_sparse",
            "lambda_spectral_": "lambda_spectral",
            "small_return_quantile_": "small_return_quantile",
            "universe_": "universe",
            "lookback_": "lookback",
        }
    )


def write_markdown_report(agg_df: pd.DataFrame, args, out_dir: Path) -> None:
    lines = [
        "# All-10 Hyperparameter Search",
        "",
        "Custom currencies:",
        f"- `{', '.join(args.custom_currencies)}`",
        "",
    ]
    for model_name, group in agg_df.groupby("model"):
        rank_col = f"{args.rank_metric}_mean"
        ascending = args.rank_metric in MINIMIZE_METRICS
        ranked = group.sort_values([rank_col, "rmse_mean"], ascending=[ascending, True]).head(args.top_n_report)
        lines.extend(
            [
                f"## {model_name}",
                "",
                f"Rank metric: `{args.rank_metric}`",
                "",
                "| Trial | Hit | RMSE | Pairwise Hit | IC | LS Sharpe | top_k | graph_rank | hidden | dropout | edge_dropout | spectral_bound | lambda_dir | lambda_rank |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in ranked.iterrows():
            lines.append(
                "| {trial_id} | {hit_ratio_mean:.4f} | {rmse_mean:.6f} | {pairwise_hit_mean:.4f} | {ic_mean:.4f} | {long_short_sharpe_mean:.4f} | {top_k:.0f} | {graph_rank:.0f} | {hidden:.0f} | {dropout:.2f} | {edge_dropout:.2f} | {spectral_bound:.2f} | {lambda_dir:.3f} | {lambda_rank:.3f} |".format(
                    **row.to_dict()
                )
            )
        lines.append("")
    (out_dir / "hparam_search_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    currency_names = resolve_currency_names(args.universe, args.custom_currencies)

    prepared_cache: Dict[bool, object] = {}
    raw_rows: List[Dict[str, object]] = []
    trial_counter = 0

    for model_name in args.models:
        if model_name not in RELATIONAL_MODELS:
            raise ValueError(f"Only relational models are supported in this search script: {model_name}")
        include_regime = model_name == "oursmain"
        if include_regime not in prepared_cache:
            prepared_cache[include_regime] = prepare_data(
                args.fx_data_path,
                args.nonfx_data_path,
                currency_names,
                args.lookback,
                args.split,
                include_regime_onehot=include_regime,
                start_date=args.start_date,
                end_date=args.end_date,
            )
        prepared = prepared_cache[include_regime]

        for trial_config in iter_trial_configs(args):
            trial_counter += 1
            trial_id = f"{model_name}_trial{trial_counter:04d}"
            for seed in args.seeds:
                set_seed(seed)
                trial_args = build_trial_args(args, model_name, seed, trial_config)
                run_dir = output_dir / "runs" / trial_id / f"seed{seed}"
                result = train_relational_model(model_name, prepared, trial_args, run_dir, args.device)
                row = dict(result.raw_metrics)
                row.update(trial_config)
                row.update({"model": model_name, "trial_id": trial_id, "seed": seed, "universe": args.universe, "lookback": args.lookback})
                raw_rows.append(row)
            print(f"[done] {trial_id} model={model_name} top_k={trial_config['top_k']} graph_rank={trial_config['graph_rank']} hidden={trial_config['hidden']}")

    raw_df = pd.DataFrame(raw_rows)
    agg_df = aggregate_trials(raw_df)
    rank_col = f"{args.rank_metric}_mean"
    ascending = args.rank_metric in MINIMIZE_METRICS
    agg_df["rank_score"] = agg_df[rank_col].apply(lambda x: trial_score(args.rank_metric, x))
    agg_df = agg_df.sort_values(["model", "rank_score", "rmse_mean"], ascending=[True, True, True]).reset_index(drop=True)

    save_dataframe(raw_df, output_dir / "tables" / "trial_metrics_detail.csv")
    save_dataframe(agg_df, output_dir / "tables" / "trial_metrics_aggregate.csv")
    best_df = agg_df.groupby("model", as_index=False).head(1).reset_index(drop=True)
    save_dataframe(best_df, output_dir / "tables" / "best_trials.csv")
    write_markdown_report(agg_df, args, output_dir / "tables")
    print(f"saved to {output_dir}")


if __name__ == "__main__":
    main()
