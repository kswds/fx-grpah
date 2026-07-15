from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import DEFAULT_FX_DATA, DEFAULT_NONFX_DATA, OURSMAIN_DEFAULTS
from data_pipeline import add_regime_onehot_features, prepare_data, resolve_currency_names, set_seed
from metrics import compute_metrics, save_dataframe, summarize_metrics
from training import train_baseline_model, train_relational_model


RELATIONAL_MODELS = {"oursmain", "foundation_relational", "foundation_nograph", "foundation_static"}
BASELINE_MODELS = {"mlp", "lstm", "gru", "gnn"}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare model performance during extreme FX and macro regimes")
    parser.add_argument("--fx-data-path", default=str(DEFAULT_FX_DATA))
    parser.add_argument("--nonfx-data-path", default=str(DEFAULT_NONFX_DATA))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "extreme_regime_comparison"))
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
    parser.add_argument("--fx-vol-quantile", type=float, default=0.8, help="Quantile for high realized cross-sectional FX volatility regime")
    parser.add_argument("--target-move-quantile", type=float, default=0.9, help="Quantile for large realized absolute FX move regime")
    parser.add_argument("--yield-shock-quantile", type=float, default=0.85, help="Quantile for US yield shock regime")
    parser.add_argument("--vix-shock-quantile", type=float, default=0.85, help="Quantile for VIX spike regime")
    parser.add_argument("--min-scenario-dates", type=int, default=8, help="Skip scenario summaries with fewer than this many test dates")
    return parser.parse_args()


def safe_quantile(series: pd.Series, q: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, q)) if len(vals) else float("nan")


def build_extreme_scenarios(prepared, min_scenario_dates: int, fx_vol_quantile: float, target_move_quantile: float, yield_shock_quantile: float, vix_shock_quantile: float) -> pd.DataFrame:
    merged, _ = add_regime_onehot_features(prepared.merged)
    non_usd_targets = [f"TargetRet_{ccy}" for ccy in prepared.currency_names if ccy != "USD"]
    target_frame = merged[non_usd_targets].apply(pd.to_numeric, errors="coerce")
    abs_target = target_frame.abs()

    fx_vol_proxy = abs_target.mean(axis=1)
    max_abs_move = abs_target.max(axis=1)
    cross_section_dispersion = target_frame.std(axis=1)
    us10y_change = pd.to_numeric(merged.get("Global_US10Y_change", 0.0), errors="coerce").abs()
    vix_change = pd.to_numeric(merged.get("Global_VIX_change", 0.0), errors="coerce").abs()

    fx_vol_threshold = safe_quantile(fx_vol_proxy, fx_vol_quantile)
    target_move_threshold = safe_quantile(max_abs_move, target_move_quantile)
    yield_shock_threshold = safe_quantile(us10y_change, yield_shock_quantile)
    vix_shock_threshold = safe_quantile(vix_change, vix_shock_quantile)

    scenario_df = pd.DataFrame(
        {
            "target_date": pd.to_datetime(merged["Date"]).dt.normalize(),
            "all_test": True,
            "high_fx_vol": fx_vol_proxy >= fx_vol_threshold if np.isfinite(fx_vol_threshold) else False,
            "large_abs_move": max_abs_move >= target_move_threshold if np.isfinite(target_move_threshold) else False,
            "cross_section_dispersion": cross_section_dispersion >= safe_quantile(cross_section_dispersion, fx_vol_quantile),
            "risk_off": pd.to_numeric(merged.get("Regime_RiskOff", 0.0), errors="coerce").fillna(0.0) >= 0.5,
            "dollar_shock": pd.to_numeric(merged.get("Regime_DollarShock", 0.0), errors="coerce").fillna(0.0) >= 0.5,
            "commodity_shock": pd.to_numeric(merged.get("Regime_CommodityShock", 0.0), errors="coerce").fillna(0.0) >= 0.5,
            "yield_shock": us10y_change >= yield_shock_threshold if np.isfinite(yield_shock_threshold) else False,
            "vix_spike": vix_change >= vix_shock_threshold if np.isfinite(vix_shock_threshold) else False,
        }
    )
    scenario_df["macro_shock_union"] = scenario_df[["risk_off", "dollar_shock", "commodity_shock", "yield_shock", "vix_spike"]].any(axis=1)
    scenario_df["extreme_union"] = scenario_df[["high_fx_vol", "large_abs_move", "macro_shock_union"]].any(axis=1)

    scenario_cols = [c for c in scenario_df.columns if c != "target_date"]
    for col in scenario_cols:
        scenario_df[col] = scenario_df[col].astype(bool)

    counts = scenario_df[scenario_cols].sum(axis=0)
    valid_cols = ["target_date"] + [col for col in scenario_cols if col == "all_test" or counts[col] >= min_scenario_dates]
    return scenario_df.loc[:, valid_cols]


def prediction_panel(pred_df: pd.DataFrame, currency_names: List[str]) -> tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    tidy = pred_df.copy()
    tidy["target_date"] = pd.to_datetime(tidy["target_date"]).dt.normalize()
    date_index = sorted(tidy["target_date"].unique())
    pred_panel = tidy.pivot(index="target_date", columns="currency", values="pred").reindex(index=date_index, columns=currency_names)
    target_panel = tidy.pivot(index="target_date", columns="currency", values="target").reindex(index=date_index, columns=currency_names)
    valid = pred_panel.notna().all(axis=1) & target_panel.notna().all(axis=1)
    pred_panel = pred_panel.loc[valid]
    target_panel = target_panel.loc[valid]
    return pred_panel.to_numpy(dtype=float), target_panel.to_numpy(dtype=float), list(pred_panel.index)


def iter_scenarios(scenario_df: pd.DataFrame) -> Iterable[str]:
    for col in scenario_df.columns:
        if col != "target_date":
            yield col


def evaluate_extreme_scenarios(pred_df: pd.DataFrame, scenario_df: pd.DataFrame, currency_names: List[str], metadata: Dict[str, object]) -> List[Dict[str, object]]:
    pred_mat, target_mat, dates = prediction_panel(pred_df, currency_names)
    if len(dates) == 0:
        return []
    panel_dates = pd.Index(pd.to_datetime(dates).normalize())
    scenario_lookup = scenario_df.copy()
    scenario_lookup["target_date"] = pd.to_datetime(scenario_lookup["target_date"]).dt.normalize()
    scenario_lookup = scenario_lookup.drop_duplicates("target_date").set_index("target_date")
    scenario_lookup = scenario_lookup.reindex(panel_dates).fillna(False)
    non_usd_mask = np.array([c != "USD" for c in currency_names], dtype=bool)

    rows: List[Dict[str, object]] = []
    for scenario_name in scenario_lookup.columns:
        date_mask = scenario_lookup[scenario_name].to_numpy(dtype=bool)
        if not date_mask.any():
            continue
        metrics = compute_metrics(pred_mat[date_mask], target_mat[date_mask], non_usd_mask)
        row: Dict[str, object] = dict(metadata)
        row.update(metrics)
        row["scenario"] = scenario_name
        row["n_dates"] = int(date_mask.sum())
        row["n_currency_obs"] = int(date_mask.sum() * non_usd_mask.sum())
        rows.append(row)
    return rows


def summarize_extreme_metrics(raw_df: pd.DataFrame, out_dir: Path) -> None:
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
        "n_dates",
        "n_currency_obs",
    ]
    agg = raw_df.groupby(["scenario", "model", "universe", "lookback"], as_index=False)[metric_cols].agg(["mean", "std", "min", "max"])
    agg.columns = ["_".join([c for c in col if c]).strip("_") for col in agg.columns.to_flat_index()]
    agg = agg.rename(columns={"scenario_": "scenario", "model_": "model", "universe_": "universe", "lookback_": "lookback"})
    save_dataframe(raw_df, out_dir / "seed_extreme_metrics_detail.csv")
    save_dataframe(agg, out_dir / "seed_extreme_metrics_aggregate.csv")

    lines = [
        "# Extreme Regime Comparison",
        "",
    ]
    for scenario, group in agg.groupby("scenario"):
        lines.append(f"## {scenario}")
        lines.append("")
        lines.append("| Model | Dates | Mean Hit | Extreme Hit | Mean RMSE | Pairwise Hit | IC | LS Sharpe |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in group.sort_values(["hit_ratio_mean", "rmse_mean"], ascending=[False, True]).iterrows():
            lines.append(
                "| {model} | {n_dates_mean:.1f} | {hit_ratio_mean:.4f} | {extreme_hit_ratio_mean:.4f} | {rmse_mean:.6f} | {pairwise_hit_mean:.4f} | {ic_mean:.4f} | {long_short_sharpe_mean:.4f} |".format(
                    **row.to_dict()
                )
            )
        lines.append("")
    (out_dir / "extreme_metrics_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    currency_names = resolve_currency_names(args.universe, args.custom_currencies)
    include_regime = any(m == "oursmain" for m in args.models)
    prepared = prepare_data(
        args.fx_data_path,
        args.nonfx_data_path,
        currency_names,
        args.lookback,
        args.split,
        include_regime_onehot=include_regime,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    scenario_df = build_extreme_scenarios(
        prepared,
        min_scenario_dates=args.min_scenario_dates,
        fx_vol_quantile=args.fx_vol_quantile,
        target_move_quantile=args.target_move_quantile,
        yield_shock_quantile=args.yield_shock_quantile,
        vix_shock_quantile=args.vix_shock_quantile,
    )
    save_dataframe(scenario_df, output_dir / "tables" / "scenario_flags.csv")

    raw_rows: List[Dict[str, object]] = []
    scenario_rows: List[Dict[str, object]] = []
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
            scenario_rows.extend(
                evaluate_extreme_scenarios(
                    result.prediction_df,
                    scenario_df,
                    currency_names,
                    {"model": model_name, "universe": args.universe, "lookback": args.lookback, "seed": seed},
                )
            )
            print(f"[done] model={model_name} seed={seed} hit={result.raw_metrics['hit_ratio']:.4f} rmse={result.raw_metrics['rmse']:.6f}")

    raw_df = pd.DataFrame(raw_rows)
    save_dataframe(raw_df, output_dir / "tables" / "metrics_summary.csv")
    summarize_metrics(raw_df, output_dir / "tables")
    scenario_metrics_df = pd.DataFrame(scenario_rows)
    summarize_extreme_metrics(scenario_metrics_df, output_dir / "tables")
    print(f"saved to {output_dir}")


if __name__ == "__main__":
    main()
