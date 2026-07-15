from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

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
from metrics import save_dataframe
from training import train_baseline_model, train_relational_model


RELATIONAL_MODELS = {"oursmain", "foundation_relational", "foundation_nograph", "foundation_static"}
BASELINE_MODELS = {"mlp", "lstm", "gru", "gnn"}
TRADING_DAYS = 252


def parse_args():
    parser = argparse.ArgumentParser(description="Compare long-short portfolio performance across FX models")
    parser.add_argument("--fx-data-path", default=str(DEFAULT_FX_DATA))
    parser.add_argument("--nonfx-data-path", default=str(DEFAULT_NONFX_DATA))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "long_short_backtest_comparison"))
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
    parser.add_argument("--transaction-cost-bps", nargs="+", type=float, default=[0.0, 2.0, 5.0])
    parser.add_argument("--portfolio-third", type=float, default=0.3333333333, help="Fraction of non-USD currencies to long and short each date")
    return parser.parse_args()


def prediction_panel(pred_df: pd.DataFrame, currency_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    tidy = pred_df.copy()
    tidy["target_date"] = pd.to_datetime(tidy["target_date"]).dt.normalize()
    dates = sorted(tidy["target_date"].unique())
    pred_panel = tidy.pivot(index="target_date", columns="currency", values="pred").reindex(index=dates, columns=currency_names)
    target_panel = tidy.pivot(index="target_date", columns="currency", values="target").reindex(index=dates, columns=currency_names)
    valid = pred_panel.notna().all(axis=1) & target_panel.notna().all(axis=1)
    pred_panel = pred_panel.loc[valid]
    target_panel = target_panel.loc[valid]
    return pred_panel.to_numpy(dtype=float), target_panel.to_numpy(dtype=float), list(pred_panel.index)


def build_long_short_weights(pred: np.ndarray, non_usd_mask: np.ndarray, portfolio_third: float) -> np.ndarray:
    pred_nu = pred[:, non_usd_mask]
    n_dates, n_assets = pred_nu.shape
    k = max(1, min(n_assets // 2, int(np.ceil(n_assets * portfolio_third))))
    weights_nu = np.zeros_like(pred_nu, dtype=float)
    for t in range(n_dates):
        order = np.argsort(pred_nu[t])
        shorts = order[:k]
        longs = order[-k:]
        weights_nu[t, longs] = 1.0 / k
        weights_nu[t, shorts] = -1.0 / k
    weights = np.zeros_like(pred, dtype=float)
    weights[:, non_usd_mask] = weights_nu
    return weights


def max_drawdown_from_returns(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    wealth = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / np.maximum(peak, 1e-12) - 1.0
    return float(drawdown.min())


def annualized_volatility(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    vol = np.std(returns, ddof=1)
    if vol < 1e-12:
        return 0.0
    return float((np.mean(returns) / vol) * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: np.ndarray) -> float:
    downside = returns[returns < 0]
    if len(returns) < 2 or len(downside) < 2:
        return 0.0
    denom = np.std(downside, ddof=1)
    if denom < 1e-12:
        return 0.0
    return float((np.mean(returns) / denom) * np.sqrt(TRADING_DAYS))


def cagr(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    wealth = float(np.prod(1.0 + returns))
    if wealth <= 0:
        return -1.0
    years = len(returns) / TRADING_DAYS
    if years <= 0:
        return 0.0
    return float(wealth ** (1.0 / years) - 1.0)


def evaluate_long_short_portfolio(
    pred_df: pd.DataFrame,
    currency_names: Sequence[str],
    transaction_cost_bps: float,
    portfolio_third: float,
    metadata: Dict[str, object],
) -> tuple[Dict[str, object], pd.DataFrame]:
    pred_mat, target_mat, dates = prediction_panel(pred_df, currency_names)
    non_usd_mask = np.array([ccy != "USD" for ccy in currency_names], dtype=bool)
    weights = build_long_short_weights(pred_mat, non_usd_mask, portfolio_third)
    gross_returns = (weights * target_mat).sum(axis=1)
    prev_weights = np.zeros_like(weights)
    if len(weights) > 1:
        prev_weights[1:] = weights[:-1]
    turnover = 0.5 * np.abs(weights - prev_weights).sum(axis=1)
    transaction_cost = turnover * (transaction_cost_bps / 10000.0)
    net_returns = gross_returns - transaction_cost
    wealth = np.cumprod(1.0 + net_returns)

    daily_df = pd.DataFrame(
        {
            "target_date": pd.to_datetime(dates),
            "gross_return": gross_returns,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "net_return": net_returns,
            "wealth": wealth,
            "tc_bps": transaction_cost_bps,
            **metadata,
        }
    )

    summary = dict(metadata)
    summary.update(
        {
            "tc_bps": transaction_cost_bps,
            "n_dates": int(len(net_returns)),
            "avg_turnover": float(np.mean(turnover)) if len(turnover) else 0.0,
            "gross_cumulative_return": float(np.prod(1.0 + gross_returns) - 1.0) if len(gross_returns) else 0.0,
            "net_cumulative_return": float(np.prod(1.0 + net_returns) - 1.0) if len(net_returns) else 0.0,
            "cagr": cagr(net_returns),
            "annualized_vol": annualized_volatility(net_returns),
            "sharpe": sharpe_ratio(net_returns),
            "sortino": sortino_ratio(net_returns),
            "max_drawdown": max_drawdown_from_returns(net_returns),
            "hit_ratio": float(np.mean(net_returns > 0)) if len(net_returns) else 0.0,
            "avg_gross_return": float(np.mean(gross_returns)) if len(gross_returns) else 0.0,
            "avg_net_return": float(np.mean(net_returns)) if len(net_returns) else 0.0,
        }
    )
    return summary, daily_df


def summarize_portfolio_metrics(raw_df: pd.DataFrame, out_dir: Path) -> None:
    if raw_df.empty:
        return
    metric_cols = [
        "n_dates",
        "avg_turnover",
        "gross_cumulative_return",
        "net_cumulative_return",
        "cagr",
        "annualized_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "hit_ratio",
        "avg_gross_return",
        "avg_net_return",
    ]
    agg = raw_df.groupby(["tc_bps", "model", "universe", "lookback"], as_index=False)[metric_cols].agg(["mean", "std", "min", "max"])
    agg.columns = ["_".join([c for c in col if c]).strip("_") for col in agg.columns.to_flat_index()]
    agg = agg.rename(columns={"tc_bps_": "tc_bps", "model_": "model", "universe_": "universe", "lookback_": "lookback"})
    save_dataframe(raw_df, out_dir / "seed_portfolio_metrics_detail.csv")
    save_dataframe(agg, out_dir / "seed_portfolio_metrics_aggregate.csv")

    lines = [
        "# Long-Short Portfolio Comparison",
        "",
    ]
    for tc_bps, group in agg.groupby("tc_bps"):
        lines.append(f"## Transaction Cost {tc_bps:.0f}bp")
        lines.append("")
        lines.append("| Model | CAGR | Sharpe | Sortino | Cum Return | Max Drawdown | Avg Turnover |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in group.sort_values(["sharpe_mean", "cagr_mean"], ascending=[False, False]).iterrows():
            lines.append(
                "| {model} | {cagr_mean:.4f} | {sharpe_mean:.4f} | {sortino_mean:.4f} | {net_cumulative_return_mean:.4f} | {max_drawdown_mean:.4f} | {avg_turnover_mean:.4f} |".format(
                    **row.to_dict()
                )
            )
        lines.append("")
    (out_dir / "portfolio_summary.md").write_text("\n".join(lines), encoding="utf-8")


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

    portfolio_rows: List[Dict[str, object]] = []
    daily_rows: List[pd.DataFrame] = []

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

            base_meta = {"model": model_name, "universe": args.universe, "lookback": args.lookback, "seed": seed}
            for tc_bps in args.transaction_cost_bps:
                summary, daily_df = evaluate_long_short_portfolio(
                    result.prediction_df,
                    currency_names,
                    transaction_cost_bps=tc_bps,
                    portfolio_third=args.portfolio_third,
                    metadata=base_meta,
                )
                portfolio_rows.append(summary)
                daily_rows.append(daily_df)
            print(f"[done] model={model_name} seed={seed} base_hit={result.raw_metrics['hit_ratio']:.4f} base_rmse={result.raw_metrics['rmse']:.6f}")

    portfolio_df = pd.DataFrame(portfolio_rows)
    daily_df = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    save_dataframe(daily_df, output_dir / "tables" / "portfolio_daily_returns.csv")
    summarize_portfolio_metrics(portfolio_df, output_dir / "tables")
    print(f"saved to {output_dir}")


if __name__ == "__main__":
    main()
