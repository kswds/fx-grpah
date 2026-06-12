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
try:
    from exp_utils import load_or_create_results_dir, save_csv
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(__file__))
    from exp_utils import load_or_create_results_dir, save_csv

try:
    from train import long_short_sharpe
except Exception:
    long_short_sharpe = None

LOGGER = logging.getLogger("exp_portfolio_investment_comparison2")


def _args():
    p = argparse.ArgumentParser(
        description="Portfolio/trading metric comparison with turnover-based transaction costs and consistent diagnostics."
    )
    p.add_argument("--pred-dir", default=str(ROOT / "results" / "model_prediction_comparison" / "predictions"))
    p.add_argument("--output-dir", default=str(ROOT / "results" / "model_portfolio_comparison"))
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument(
        "--gross-exposures",
        nargs="+",
        type=float,
        default=[1.0],
        help="One or multiple gross exposure settings (e.g., 1.0 2.0).",
    )
    p.add_argument("--cost-bps", type=float, default=0.0, help="Transaction cost in bps applied to turnover-based daily costs.")
    p.add_argument(
        "--cost-bps-grid",
        nargs="+",
        type=float,
        default=None,
        help="Optional list of cost bps values for sensitivity (e.g., 0 5 10 20). If set, overrides --cost-bps.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _mdd(cum_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum_curve)
    dd = (cum_curve - peak) / (peak + 1e-12)
    return float(np.min(dd))


def _sortino(rets: np.ndarray) -> float:
    neg = rets[rets < 0]
    if len(neg) <= 1:
        return np.nan
    down = np.std(neg, ddof=1) + 1e-12
    return float(np.mean(rets) / down * np.sqrt(252))


def _portfolio_series_and_turnover(
    pred: np.ndarray, target: np.ndarray, k: int, gross_exposure: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    T, N = pred.shape
    out = np.zeros(T, dtype=np.float64)
    turnover = np.zeros(T, dtype=np.float64)
    weights = np.zeros((T, N), dtype=np.float64)
    k_eff = max(1, min(k, N // 2 if N >= 2 else 1))
    long_w = (gross_exposure / 2.0) / k_eff
    short_w = -(gross_exposure / 2.0) / k_eff
    for t in range(T):
        r = np.argsort(pred[t])
        longs = r[-k_eff:]
        shorts = r[:k_eff]
        w = np.zeros(N, dtype=np.float64)
        w[longs] = long_w
        w[shorts] = short_w
        weights[t] = w
        out[t] = float(np.dot(w, target[t]))
        if t == 0:
            turnover[t] = np.sum(np.abs(w))  # entry from cash (conservative)
        else:
            turnover[t] = np.sum(np.abs(w - weights[t - 1]))
    return out, turnover


def _metrics_from_rets(rets: np.ndarray) -> dict:
    ret_std = float(np.std(rets, ddof=1)) + 1e-12
    curve = np.cumprod(1.0 + rets)
    terminal_wealth = float(curve[-1]) if len(curve) else np.nan
    cumulative_return = terminal_wealth - 1.0 if np.isfinite(terminal_wealth) else np.nan
    years = max(len(rets) / 252.0, 1e-12)
    cagr = float(terminal_wealth ** (1.0 / years) - 1.0) if np.isfinite(terminal_wealth) and terminal_wealth > 0 else np.nan
    sharpe = float(np.mean(rets) / ret_std * np.sqrt(252))
    volatility = float(ret_std * np.sqrt(252))
    sortino = float(_sortino(rets))
    mdd = float(_mdd(curve))
    q = np.quantile(rets, [0.01, 0.05, 0.50, 0.95, 0.99]) if len(rets) else [np.nan] * 5
    return {
        "sharpe": sharpe,
        "cumulative_return": float(cumulative_return),
        "terminal_wealth": float(terminal_wealth),
        "cagr": cagr,
        "volatility": volatility,
        "sortino": sortino,
        "mdd": mdd,
        "ret_mean_daily": float(np.mean(rets)) if len(rets) else np.nan,
        "ret_std_daily": float(np.std(rets, ddof=1)) if len(rets) > 1 else np.nan,
        "ret_min": float(np.min(rets)) if len(rets) else np.nan,
        "ret_max": float(np.max(rets)) if len(rets) else np.nan,
        "ret_q01": float(q[0]),
        "ret_q05": float(q[1]),
        "ret_q50": float(q[2]),
        "ret_q95": float(q[3]),
        "ret_q99": float(q[4]),
    }


def main():
    args = _args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    dirs = load_or_create_results_dir(args.output_dir, ["metrics", "tables"])

    rows = []
    cost_grid = args.cost_bps_grid if args.cost_bps_grid is not None and len(args.cost_bps_grid) > 0 else [args.cost_bps]
    pred_paths = [p for p in sorted(Path(args.pred_dir).glob("*.parquet")) if not p.name.endswith("_matrix.parquet")]
    if not pred_paths:
        raise FileNotFoundError(f"No prediction parquet files found under: {args.pred_dir}")

    for p in pred_paths:
        df = pd.read_parquet(p)
        if "target_date" not in df.columns:
            raise ValueError(f"prediction file missing target_date column: {p}")
        df["target_date"] = pd.to_datetime(df["target_date"])
        ccys = list(df["currency"].drop_duplicates())
        pred = df.pivot_table(index="target_date", columns="currency", values="pred", aggfunc="first").reindex(columns=ccys).sort_index()
        tgt = df.pivot_table(index="target_date", columns="currency", values="target", aggfunc="first").reindex(columns=ccys).sort_index()
        common = pred.index.intersection(tgt.index)
        pred_al = pred.reindex(common)
        tgt_al = tgt.reindex(common)

        pred_nan_count = int(pred_al.isna().sum().sum())
        target_nan_count = int(tgt_al.isna().sum().sum())
        if pred_nan_count > 0 or target_nan_count > 0:
            LOGGER.warning(
                "NaNs detected before fillna: pred_nan=%d, target_nan=%d, file=%s",
                pred_nan_count,
                target_nan_count,
                p.name,
            )

        pred_v = pred_al.fillna(0.0).values
        tgt_v = tgt_al.fillna(0.0).values

        usd_idx = ccys.index("USD") if "USD" in ccys else 0
        mask = np.ones(len(ccys), dtype=bool)
        mask[usd_idx] = False
        pred_m = pred_v[:, mask]
        tgt_m = tgt_v[:, mask]

        # Target scale diagnostics (must be raw FX return scale for portfolio evaluation)
        target_mean = float(np.nanmean(tgt_m))
        target_std = float(np.nanstd(tgt_m))
        target_min = float(np.nanmin(tgt_m))
        target_max = float(np.nanmax(tgt_m))
        tq = np.nanquantile(tgt_m, [0.01, 0.50, 0.99])
        target_q01, target_q50, target_q99 = float(tq[0]), float(tq[1]), float(tq[2])
        if target_std > 0.1:
            LOGGER.warning("Target scale may not be raw FX return scale: std=%.4f file=%s", target_std, p.name)

        for gross_exposure in args.gross_exposures:
            rets, turnover = _portfolio_series_and_turnover(
                pred_m, tgt_m, args.top_k, gross_exposure=gross_exposure
            )
            for cost_bps in cost_grid:
                cost_rate = float(cost_bps) / 10000.0
                costs = turnover * cost_rate
                rets_after_cost = rets - costs

                m = _metrics_from_rets(rets_after_cost)

                row = {
                    "model": str(df["model"].iloc[0]),
                    "seed": int(df["seed"].iloc[0]),
                    "lookback": str(df["lookback"].iloc[0]),
                    "gross_exposure": float(gross_exposure),
                    "cost_bps": float(cost_bps),
                    "top_k": int(args.top_k),
                    "turnover_mean_daily": float(np.mean(turnover)) if len(turnover) else np.nan,
                    "turnover_std_daily": float(np.std(turnover, ddof=1)) if len(turnover) > 1 else np.nan,
                    "cost_mean_daily": float(np.mean(costs)) if len(costs) else np.nan,
                    "pred_nan_count": pred_nan_count,
                    "target_nan_count": target_nan_count,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "target_min": target_min,
                    "target_max": target_max,
                    "target_q01": target_q01,
                    "target_q50": target_q50,
                    "target_q99": target_q99,
                    **m,
                    "cumulative_return_pct": 100.0 * float(m["cumulative_return"]),
                    "volatility_pct": 100.0 * float(m["volatility"]),
                    "mdd_pct": 100.0 * float(m["mdd"]),
                    "cagr_pct": 100.0 * float(m["cagr"]) if np.isfinite(m["cagr"]) else np.nan,
                }

                # Optional diagnostic comparison against legacy helper (if available)
                if long_short_sharpe is not None:
                    try:
                        ls = long_short_sharpe(pred_m, tgt_m, k=args.top_k, mask=None)
                        row["diag_ls_sharpe"] = float(ls.get("sharpe", np.nan))
                        row["diag_ls_cumulative_return"] = float(ls.get("cumulative_return", np.nan))
                    except Exception:
                        row["diag_ls_sharpe"] = np.nan
                        row["diag_ls_cumulative_return"] = np.nan

                rows.append(row)
                LOGGER.info(
                    "model=%s seed=%s gross=%.2f cost_bps=%.2f mean=%.6f std=%.6f vol=%.4f sharpe=%.3f cumret=%.3f tw=%.3f mdd=%.3f target_std=%.6f",
                    row["model"],
                    row["seed"],
                    row["gross_exposure"],
                    row["cost_bps"],
                    row["ret_mean_daily"],
                    row["ret_std_daily"] if np.isfinite(row["ret_std_daily"]) else np.nan,
                    row["volatility"],
                    row["sharpe"],
                    row["cumulative_return"],
                    row["terminal_wealth"],
                    row["mdd"],
                    row["target_std"],
                )

    raw = pd.DataFrame(rows).sort_values(["model", "gross_exposure", "seed"])
    save_csv(dirs["metrics"] / "portfolio_metrics_raw.csv", raw)

    summary = raw.groupby(["model", "lookback", "gross_exposure", "cost_bps"], as_index=False).agg(
        sharpe_mean=("sharpe", "mean"),
        sharpe_std=("sharpe", "std"),
        cumulative_return_mean=("cumulative_return", "mean"),
        cumulative_return_std=("cumulative_return", "std"),
        terminal_wealth_mean=("terminal_wealth", "mean"),
        terminal_wealth_std=("terminal_wealth", "std"),
        cagr_mean=("cagr", "mean"),
        cagr_std=("cagr", "std"),
        sortino_mean=("sortino", "mean"),
        sortino_std=("sortino", "std"),
        volatility_mean=("volatility", "mean"),
        volatility_std=("volatility", "std"),
        mdd_mean=("mdd", "mean"),
        mdd_std=("mdd", "std"),
        ret_mean_daily_mean=("ret_mean_daily", "mean"),
        ret_mean_daily_std=("ret_mean_daily", "std"),
        ret_std_daily_mean=("ret_std_daily", "mean"),
        ret_std_daily_std=("ret_std_daily", "std"),
        ret_min_mean=("ret_min", "mean"),
        ret_min_std=("ret_min", "std"),
        ret_max_mean=("ret_max", "mean"),
        ret_max_std=("ret_max", "std"),
        turnover_mean_daily_mean=("turnover_mean_daily", "mean"),
        turnover_mean_daily_std=("turnover_mean_daily", "std"),
        cost_mean_daily_mean=("cost_mean_daily", "mean"),
        cost_mean_daily_std=("cost_mean_daily", "std"),
        pred_nan_count_mean=("pred_nan_count", "mean"),
        target_nan_count_mean=("target_nan_count", "mean"),
        target_std_mean=("target_std", "mean"),
        target_std_std=("target_std", "std"),
    ).fillna(0.0).sort_values("sharpe_mean", ascending=False)

    summary["cumulative_return_pct_mean"] = 100.0 * summary["cumulative_return_mean"]
    summary["cagr_pct_mean"] = 100.0 * summary["cagr_mean"]
    summary["volatility_pct_mean"] = 100.0 * summary["volatility_mean"]
    summary["mdd_pct_mean"] = 100.0 * summary["mdd_mean"]

    save_csv(dirs["tables"] / "portfolio_metrics_summary.csv", summary)

    presentation_summary = summary.copy()
    save_csv(dirs["tables"] / "portfolio_metrics_summary_presentation.csv", presentation_summary)
    (dirs["tables"] / "portfolio_metrics_summary.md").write_text(summary.to_markdown(index=False), encoding="utf-8")

    # Cost sensitivity pivot for quick review
    cost_pivot = summary.pivot_table(
        index=["model", "gross_exposure"],
        columns="cost_bps",
        values=["sharpe_mean", "cumulative_return_mean", "cagr_mean", "mdd_mean"],
    )
    cost_pivot.columns = [f"{m}_cost{c:g}" for m, c in cost_pivot.columns]
    cost_pivot = cost_pivot.reset_index()
    save_csv(dirs["tables"] / "portfolio_cost_sensitivity.csv", cost_pivot)


if __name__ == "__main__":
    main()
