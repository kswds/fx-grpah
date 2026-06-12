import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("plot_graph_change_shock_focus")


def _args():
    p = argparse.ArgumentParser(
        description="Visualize that dynamic-graph updates are concentrated around macro-shock regimes."
    )
    p.add_argument(
        "--timeseries-csv",
        default=str(ROOT / "results" / "stress_regime_analysis" / "graph_adaptation" / "graph_change_timeseries.csv"),
    )
    p.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "stress_regime_analysis" / "figures"),
    )
    p.add_argument("--rolling-window", type=int, default=20)
    p.add_argument("--top-pct", type=float, default=0.15)
    p.add_argument("--min-gap-days", type=int, default=90)
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _pick_spaced_peaks(series: pd.Series, top_n: int, min_gap_days: int) -> np.ndarray:
    order = np.argsort(series.values)[::-1]
    picked = []
    for idx in order:
        dt = series.index[idx]
        if not picked:
            picked.append(idx)
        else:
            if min(abs((dt - series.index[j]).days) for j in picked) >= min_gap_days:
                picked.append(idx)
        if len(picked) >= min(top_n, len(series)):
            break
    return np.array(sorted(picked), dtype=int)


def _main_figure(df: pd.DataFrame, out_dir: Path, rolling_window: int, top_pct: float, top_n: int, min_gap_days: int):
    x = pd.to_datetime(df["date"])
    d_f = df["D_F"].astype(float)
    d_roll = d_f.rolling(rolling_window, min_periods=1).mean()

    is_stress = (
        df["is_vix_stress"].astype(bool)
        | df["is_us2y_shock"].astype(bool)
        | df["is_oil_shock"].astype(bool)
        | df["is_composite_stress"].astype(bool)
    )
    stress_roll = is_stress.astype(float).rolling(rolling_window, min_periods=1).mean()

    thr = float(np.nanquantile(d_roll.values, 1.0 - top_pct))
    high_mask = d_roll >= thr
    peak_idx = _pick_spaced_peaks(pd.Series(d_roll.values, index=x), top_n=top_n, min_gap_days=min_gap_days)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#070b16")
    ax.set_facecolor("#0b1329")

    # Stress-day vertical shading
    ax.vlines(x[is_stress.values], ymin=0, ymax=np.nanmax(d_roll.values) * 1.1, color="#4da3ff", alpha=0.10, linewidth=1)

    # Highlight high-change rolling zones
    ax.fill_between(x, 0, d_roll.values, where=high_mask.values, color="#ff5a5f", alpha=0.20, label=f"Top {int(top_pct*100)}% rolling-change zone")
    ax.plot(x, d_roll.values, color="#ffd84d", linewidth=2.2, label=f"{rolling_window}d rolling D_F")
    ax.axhline(thr, color="#ff7a7a", linestyle="--", linewidth=1.2, alpha=0.9, label=f"{int((1-top_pct)*100)}th percentile")

    # Scaled stress ratio line (for temporal alignment)
    y_max = float(np.nanmax(d_roll.values))
    ax.plot(x, stress_roll.values * y_max, color="#46c2ff", linewidth=1.2, alpha=0.75, label=f"{rolling_window}d stress-day ratio (scaled)")

    # Spaced top peaks
    px = x.iloc[peak_idx]
    py = d_roll.iloc[peak_idx]
    ax.scatter(px, py, color="#ff4d4d", s=56, edgecolors="white", linewidths=0.8, zorder=5, label=f"Top {len(px)} peaks")
    for i, (dx, dy) in enumerate(zip(px, py), start=1):
        ax.annotate(
            f"#{i}",
            xy=(dx, dy),
            xytext=(0, 12 if i % 2 else -16),
            textcoords="offset points",
            ha="center",
            va="bottom" if i % 2 else "top",
            fontsize=9,
            color="#f1f3ff",
            bbox=dict(boxstyle="round,pad=0.2", fc="#4a1618", ec="#ff5a5f", lw=1.0, alpha=0.95),
        )

    ax.set_title("Graph Reconfiguration Intensity Concentrates Around Stress Regimes", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling graph-change intensity (D_F)")
    ax.grid(alpha=0.16, color="#8aa0ff")
    ax.legend(loc="upper left")

    info = (
        f"Period: {x.min().date()} - {x.max().date()}\n"
        f"Rolling window: {rolling_window}d, top zone: {int(top_pct*100)}%\n"
        f"Stress days: {int(is_stress.sum())}/{len(is_stress)}"
    )
    ax.text(
        0.995,
        0.98,
        info,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#dce4ff",
        bbox=dict(boxstyle="round,pad=0.3", fc="#182447", ec="#465d9e", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(out_dir / "graph_change_shock_focus.png", dpi=170)
    fig.savefig(out_dir / "graph_change_shock_focus.pdf")
    plt.close(fig)


def _overlap_table(df: pd.DataFrame, out_dir: Path, rolling_window: int, top_pct: float):
    x = pd.to_datetime(df["date"])
    d_roll = df["D_F"].astype(float).rolling(rolling_window, min_periods=1).mean()
    is_stress = (
        df["is_vix_stress"].astype(bool)
        | df["is_us2y_shock"].astype(bool)
        | df["is_oil_shock"].astype(bool)
        | df["is_composite_stress"].astype(bool)
    )
    thr = float(np.nanquantile(d_roll.values, 1.0 - top_pct))
    is_high = d_roll >= thr
    out = pd.DataFrame(
        [
            {
                "rolling_window": int(rolling_window),
                "top_pct": float(top_pct),
                "threshold": thr,
                "high_change_days": int(is_high.sum()),
                "stress_days": int(is_stress.sum()),
                "high_change_share": float(is_high.mean()),
                "stress_share": float(is_stress.mean()),
                "stress_share_among_high_change": float(is_stress[is_high].mean()) if is_high.any() else np.nan,
                "high_change_share_among_stress": float(is_high[is_stress].mean()) if is_stress.any() else np.nan,
                "high_change_share_among_normal": float(is_high[~is_stress].mean()) if (~is_stress).any() else np.nan,
            }
        ]
    )
    out.to_csv(out_dir / "graph_change_shock_focus_stats.csv", index=False)

    top_days = pd.DataFrame(
        {
            "date": x,
            "rolling_D_F": d_roll.values,
            "is_stress": is_stress.values,
            "is_high_change": is_high.values,
        }
    ).sort_values("rolling_D_F", ascending=False)
    top_days.head(50).to_csv(out_dir / "graph_change_top50_days.csv", index=False)


def _macro_overlay_and_scatter(df: pd.DataFrame, out_dir: Path, rolling_window: int):
    x = pd.to_datetime(df["date"])
    d_roll = df["D_F"].astype(float).rolling(rolling_window, min_periods=1).mean()

    # Macro volatility proxies
    shock = df["ShockScore"].astype(float)
    vix = df["VIX"].astype(float)
    du2 = df["Delta_US2Y_abs"].astype(float)
    oil = df["Oil_return_abs"].astype(float)

    shock_r = shock.rolling(rolling_window, min_periods=1).mean()
    vix_r = vix.rolling(rolling_window, min_periods=1).mean()
    du2_r = du2.rolling(rolling_window, min_periods=1).mean()
    oil_r = oil.rolling(rolling_window, min_periods=1).mean()

    # Normalize for same-axis visual comparison
    def _z(s: pd.Series) -> pd.Series:
        m = float(np.nanmean(s.values))
        sd = float(np.nanstd(s.values)) + 1e-12
        return (s - m) / sd

    d_z = _z(d_roll)
    shock_z = _z(shock_r)
    vix_z = _z(vix_r)
    du2_z = _z(du2_r)
    oil_z = _z(oil_r)

    # Overlay figure
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, d_z.values, color="#1f77b4", linewidth=2.0, label=f"Rolling D_F ({rolling_window}d, z-score)")
    ax.plot(x, shock_z.values, color="#d62728", linewidth=1.8, alpha=0.9, label="Rolling ShockScore (z-score)")
    ax.plot(x, vix_z.values, color="#9467bd", linewidth=1.4, alpha=0.85, label="Rolling VIX (z-score)")
    ax.plot(x, du2_z.values, color="#2ca02c", linewidth=1.4, alpha=0.85, label="Rolling |ΔUS2Y| (z-score)")
    ax.plot(x, oil_z.values, color="#ff7f0e", linewidth=1.4, alpha=0.85, label="Rolling |Oil return| (z-score)")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_title("Macro Volatility vs Dynamic Graph Reconfiguration")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized level (z-score)")
    ax.legend(loc="upper right", ncol=2, frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "macro_volatility_vs_graph_change_overlay.png", dpi=170)
    fig.savefig(out_dir / "macro_volatility_vs_graph_change_overlay.pdf")
    plt.close(fig)

    # Scatter panels with simple linear fit
    pairs = [
        ("ShockScore", shock.values, "#d62728"),
        ("VIX", vix.values, "#9467bd"),
        ("Delta_US2Y_abs", du2.values, "#2ca02c"),
        ("Oil_return_abs", oil.values, "#ff7f0e"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    axs = axs.ravel()
    rows = []
    y = d_roll.values.astype(float)
    for i, (name, xv_raw, color) in enumerate(pairs):
        xv = np.asarray(xv_raw, dtype=float)
        m = np.isfinite(xv) & np.isfinite(y)
        xv = xv[m]
        yy = y[m]
        if len(xv) >= 3:
            corr = float(np.corrcoef(xv, yy)[0, 1])
            a, b = np.polyfit(xv, yy, 1)
            xs = np.linspace(float(np.nanmin(xv)), float(np.nanmax(xv)), 200)
            ys = a * xs + b
        else:
            corr, a, b = np.nan, np.nan, np.nan
            xs, ys = np.array([]), np.array([])
        rows.append({"factor": name, "corr_with_rolling_D_F": corr, "slope": a, "intercept": b, "n_obs": int(len(xv))})

        ax = axs[i]
        ax.scatter(xv, yy, s=10, alpha=0.35, color=color, edgecolors="none")
        if len(xs):
            ax.plot(xs, ys, color="black", linewidth=1.2)
        ax.set_title(f"{name} vs rolling D_F (corr={corr:.3f})" if np.isfinite(corr) else f"{name} vs rolling D_F")
        ax.set_xlabel(name)
        ax.set_ylabel(f"rolling D_F ({rolling_window}d)")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "macro_volatility_vs_graph_change_scatter.png", dpi=170)
    fig.savefig(out_dir / "macro_volatility_vs_graph_change_scatter.pdf")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(out_dir / "macro_volatility_graph_change_correlation.csv", index=False)


def main():
    args = _args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ts_path = Path(args.timeseries_csv)
    if not ts_path.exists():
        raise FileNotFoundError(f"timeseries csv not found: {ts_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ts_path)
    required = {
        "date",
        "D_F",
        "is_vix_stress",
        "is_us2y_shock",
        "is_oil_shock",
        "is_composite_stress",
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {ts_path}: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    _main_figure(
        df=df,
        out_dir=out_dir,
        rolling_window=args.rolling_window,
        top_pct=args.top_pct,
        top_n=args.top_n,
        min_gap_days=args.min_gap_days,
    )
    _overlap_table(df=df, out_dir=out_dir, rolling_window=args.rolling_window, top_pct=args.top_pct)
    _macro_overlay_and_scatter(df=df, out_dir=out_dir, rolling_window=args.rolling_window)
    LOGGER.info("saved: %s", out_dir / "graph_change_shock_focus.png")
    LOGGER.info("saved: %s", out_dir / "graph_change_shock_focus_stats.csv")


if __name__ == "__main__":
    main()
