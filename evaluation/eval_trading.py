"""
Signal-Strength-Filtered Trading Backtest (Realistic)

Outputs:
  - trading_with_costs.png        (55 pairs, all currencies)
  - trading_45pairs.png           (45 pairs, USD excluded)
  - trading_confidence_no_usd.png (same as trading_45pairs)
  - trading_usd_excluded.png      (45 pairs, quintile-based filter)
  - eval_trading_results.json     (all numerical results)

Strategy (per day):
  1. Ridge predicts 11 currency strengths from last-day features
  2. 55 (or 45) pairwise signals: pred_ij = s_i - s_j
  3. Signal strength = |pred_ij|, filter top X% by strength
  4. Equal-weight allocation across selected pairs (each gets 1/n_select of capital)
  5. Per-pair simple return: sign(pred) * (exp(log_return_ij) - 1)
  6. Portfolio compounds daily; transaction cost deducted on notional

Returns are in USD terms assuming:
  - Initial capital $1 (normalized)
  - Each pair traded at 1/n_select notional
  - No leverage (total gross exposure = 1x)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from itertools import combinations
from datetime import datetime

from config import Config
from dataset import load_data, build_features, normalize_data

MACRO_FEATURES = ["Global_VIX", "Global_Gold", "Global_Oil",
                  "Global_Copper", "Global_US2Y", "Global_IronOre"]


def prepare_data():
    """Load data and train Ridge model, return predictions and actuals."""
    config = Config(seed=42, lookback=20)
    df = load_data(config)
    X_local_base, X_macro_full, Y = build_features(df, config)

    macro_idx = [config.global_features.index(f) for f in MACRO_FEATURES]
    X_macro = X_macro_full[:, macro_idx]

    n_total = len(X_local_base)
    split_idx = int(n_total * 0.8)
    X_local_scaled, X_macro_scaled, Y_raw, _ = normalize_data(
        X_local_base, X_macro, Y, train_idx=split_idx)

    L = config.lookback
    n = n_total - L
    split = int(n * 0.8)

    X_local_list, X_macro_list, Y_list = [], [], []
    for idx in range(n):
        X_local_list.append(X_local_scaled[idx + L - 1])
        X_macro_list.append(X_macro_scaled[idx + L - 1])
        Y_list.append(Y_raw[idx + L])

    X_local = np.stack(X_local_list)
    X_macro = np.stack(X_macro_list)
    Y = np.stack(Y_list)

    X = np.concatenate([X_local.reshape(n, -1), X_macro], axis=1)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, Y_train)
    pred = ridge.predict(X_test)

    info = {
        'n_train': int(split),
        'n_test': int(n - split),
        'n_currencies': int(config.n_ccy),
        'currencies': config.ccys,
    }

    return pred, Y_test, config, info


def compute_pairwise(strengths, n_ccy, exclude_usd=False, usd_idx=0):
    """Convert strength predictions to pairwise predictions."""
    pairs = []
    for i, j in combinations(range(n_ccy), 2):
        if exclude_usd and (i == usd_idx or j == usd_idx):
            continue
        pairs.append((i, j))

    pred_pairs = np.stack([strengths[:, i] - strengths[:, j] for i, j in pairs], axis=1)
    return pred_pairs, pairs


def trading_backtest(pred_pairs, actual_log_pairs, top_frac, cost_bps=0):
    """
    Realistic compounding backtest with hit rate tracking.

    Returns:
        nav: [T+1] normalized NAV starting at 1.0
        daily_simple_returns: [T] portfolio simple returns
        hit_rate: direction accuracy of selected pairs
    """
    T, P = pred_pairs.shape
    n_select = max(1, int(P * top_frac))
    cost = cost_bps / 10000

    nav = np.ones(T + 1)
    daily_returns = np.zeros(T)
    daily_hits = np.zeros(T)

    for t in range(T):
        signal_strength = np.abs(pred_pairs[t])
        top_idx = np.argsort(signal_strength)[-n_select:]

        simple_returns = np.expm1(actual_log_pairs[t, top_idx])
        signals = np.sign(pred_pairs[t, top_idx])
        pair_returns = signals * simple_returns

        # Hit rate: how many selected pairs had correct direction
        actual_signs = np.sign(actual_log_pairs[t, top_idx])
        daily_hits[t] = np.mean(signals == actual_signs)

        port_return = pair_returns.mean() - 2 * cost
        daily_returns[t] = port_return
        nav[t + 1] = nav[t] * (1 + port_return)

    hit_rate = np.mean(daily_hits)
    return nav, daily_returns, hit_rate


def compute_stats(daily_returns, nav, hit_rate):
    """Compute realistic trading statistics."""
    total_return = (nav[-1] / nav[0] - 1) * 100
    n_years = len(daily_returns) / 252
    cagr = ((nav[-1] / nav[0]) ** (1 / n_years) - 1) * 100
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
    max_dd = _max_drawdown(nav)
    win_rate = np.mean(daily_returns > 0)
    vol_annual = np.std(daily_returns) * np.sqrt(252) * 100

    return {
        'total_return_pct': round(total_return, 2),
        'cagr_pct': round(cagr, 2),
        'sharpe_ratio': round(sharpe, 2),
        'hit_rate': round(hit_rate, 4),
        'annual_vol_pct': round(vol_annual, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'win_rate': round(win_rate, 4),
        'n_days': len(daily_returns),
        'final_nav': round(float(nav[-1]), 4),
    }


def _max_drawdown(nav):
    """Compute maximum drawdown in % from NAV series."""
    peak = np.maximum.accumulate(nav)
    dd = (peak - nav) / peak * 100
    return np.max(dd) if len(dd) > 0 else 0.0


def run_backtest_suite(pred_pairs, actual_log_pairs, n_pairs, fracs, costs):
    """Run all backtest combinations."""
    results = {}
    plot_data = {}

    for cost in costs:
        cost_key = f"{cost}bps"
        results[cost_key] = {}
        plot_data[cost_key] = {}

        for frac in fracs:
            nav, daily, hit_rate = trading_backtest(
                pred_pairs, actual_log_pairs, frac, cost_bps=cost)
            stats = compute_stats(daily, nav, hit_rate)
            n_select = max(1, int(n_pairs * frac))

            frac_key = "all" if frac == 1.0 else f"top{int(frac*100)}pct"
            stats['n_pairs_traded'] = n_select
            stats['filter_pct'] = frac
            results[cost_key][frac_key] = stats
            plot_data[cost_key][frac_key] = {'nav': nav, 'daily': daily}

    return results, plot_data


def plot_trading_with_costs(plot_data, n_pairs, fracs, costs, title_prefix, output_name=None):
    """Plot NAV curves at different cost levels."""
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(fracs)))

    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

    for ax_idx, cost in enumerate(costs):
        ax = axes[ax_idx]
        cost_key = f"{cost}bps"

        for i, frac in enumerate(fracs):
            frac_key = "all" if frac == 1.0 else f"top{int(frac*100)}pct"
            nav = plot_data[cost_key][frac_key]['nav']
            cum_pct = (nav / nav[0] - 1) * 100

            label = f"All ({n_pairs})" if frac == 1.0 else f"Top {int(frac*100)}%"
            ax.plot(cum_pct, label=label, color=colors[i], linewidth=1.2)

        all_nav = plot_data[cost_key]["all"]['nav']
        all_daily = plot_data[cost_key]["all"]['daily']
        total_ret = (all_nav[-1] / all_nav[0] - 1) * 100
        n_years = len(all_daily) / 252
        cagr = ((all_nav[-1] / all_nav[0]) ** (1 / n_years) - 1) * 100
        sharpe = np.mean(all_daily) / (np.std(all_daily) + 1e-10) * np.sqrt(252)

        cost_label = "0 bps (Ideal)" if cost == 0 else f"{cost} bps" + (" (Realistic)" if cost == 10 else "")
        ax.set_title(f"{title_prefix} | Cost: {cost_label}  |  "
                     f"All: {total_ret:.1f}% total, CAGR {cagr:.1f}%, Sharpe {sharpe:.1f}", fontsize=11)
        ax.set_ylabel("Cumulative Return (%)")
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Days")
    plt.tight_layout()

    if output_name:
        out_path = os.path.join(os.path.dirname(__file__), '..', 'results', output_name)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out_path}")
    plt.close()


def plot_trading_quintile(pred_pairs, actual_log_pairs, n_pairs, costs, output_name=None):
    """Plot USD-excluded trading with quintile-based filtering."""
    n_quintile = n_pairs // 5
    quintiles = [
        (f"All ({n_pairs}/55)", 1.0),
        (f"Top 4 ({4*n_quintile}/45)", 4*n_quintile/n_pairs),
        (f"Top 3 ({3*n_quintile}/45)", 3*n_quintile/n_pairs),
        (f"Top 2 ({2*n_quintile}/45)", 2*n_quintile/n_pairs),
        (f"Top 1 ({n_quintile}/45)", n_quintile/n_pairs),
    ]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(quintiles)))

    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

    results_quintile = {}
    for ax_idx, cost in enumerate(costs):
        ax = axes[ax_idx]
        cost_key = f"{cost}bps"
        results_quintile[cost_key] = {}

        for i, (label, frac) in enumerate(quintiles):
            nav, daily, hit_rate = trading_backtest(
                pred_pairs, actual_log_pairs, frac, cost_bps=cost)
            stats = compute_stats(daily, nav, hit_rate)
            q_key = f"quintile_{5-i}" if i > 0 else "all"
            results_quintile[cost_key][q_key] = stats

            cum_pct = (nav / nav[0] - 1) * 100

            if i == 0:
                title_stats = (f"All: {stats['total_return_pct']:.1f}% total, "
                               f"CAGR {stats['cagr_pct']:.1f}%, Sharpe {stats['sharpe_ratio']:.1f}")

            ax.plot(cum_pct, label=label, color=colors[i], linewidth=1.2)

        cost_label = "0 bps (Ideal)" if cost == 0 else f"{cost} bps" + (" (Realistic)" if cost == 10 else "")
        ax.set_title(f"Transaction Cost: {cost_label}  |  {title_stats}", fontsize=11)
        ax.set_ylabel("Cumulative Return (%)")
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Days")
    plt.tight_layout()

    if output_name:
        out_path = os.path.join(os.path.dirname(__file__), '..', 'results', output_name)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out_path}")
    plt.close()

    return results_quintile


def compute_pair_hit_rates(pred, actual, config):
    """Compute per-pair hit rate and signal strength for all 55 pairs."""
    pairs = list(combinations(range(config.n_ccy), 2))
    pair_stats = {}
    for i, j in pairs:
        pred_diff = pred[:, i] - pred[:, j]
        actual_diff = actual[:, i] - actual[:, j]
        hit = np.mean(np.sign(pred_diff) == np.sign(actual_diff))
        strength = np.mean(np.abs(pred_diff))
        pair_name = f"{config.ccys[i]}/{config.ccys[j]}"
        pair_stats[pair_name] = {
            'hit_rate': round(float(hit), 4),
            'avg_signal_strength': round(float(strength), 6),
            'reliability': round(float(hit * strength), 6),
        }
    return pair_stats


def print_strategy_table(results, n_pairs, fracs, cost_key="0bps"):
    """Print strategy comparison table."""
    print(f"\n  Signal-strength-filtered strategy comparison ({cost_key}):")
    print(f"  {'Strategy':<12} {'Trades':>6} {'Hit Rate':>9} {'Total':>9} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>7} {'Vol':>6}")
    print(f"  {'-'*68}")

    for frac in fracs:
        frac_key = "all" if frac == 1.0 else f"top{int(frac*100)}pct"
        s = results[cost_key][frac_key]
        n_traded = s['n_pairs_traded']

        if frac == 1.0:
            name = f"All ({n_pairs})"
        else:
            name = f"Top {int(frac*100)}%"

        print(f"  {name:<12} {n_traded:>5.1f} {s['hit_rate']:>8.1%} "
              f"{s['total_return_pct']:>8.1f}% {s['cagr_pct']:>7.1f}% "
              f"{s['sharpe_ratio']:>6.2f} {s['max_drawdown_pct']:>6.1f}% "
              f"{s['annual_vol_pct']:>5.1f}%")


def main():
    print("=" * 60)
    print("Trading Backtest Evaluation (Realistic Compounding)")
    print("=" * 60)

    pred, Y_test, config, data_info = prepare_data()
    n_years = data_info['n_test'] / 252
    print(f"Test period: {data_info['n_test']} days ({n_years:.1f} years), {config.n_ccy} currencies")

    fracs = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
    costs = [0, 5, 10]
    all_results = {'timestamp': datetime.now().isoformat(), 'data': data_info}

    # --- 55 pairs (all currencies) ---
    print("\n[1/4] 55 pairs, all currencies...")
    pred_55, pairs_55 = compute_pairwise(pred, config.n_ccy, exclude_usd=False, usd_idx=config.usd_idx)
    actual_55, _ = compute_pairwise(Y_test, config.n_ccy, exclude_usd=False, usd_idx=config.usd_idx)
    results_55, plot_55 = run_backtest_suite(pred_55, actual_55, len(pairs_55), fracs, costs)
    plot_trading_with_costs(plot_55, len(pairs_55), fracs, costs,
                           f"{len(pairs_55)} Pairs", output_name="trading_with_costs.png")
    all_results['55_pairs'] = results_55

    # --- 45 pairs (USD excluded) ---
    print("[2/4] 45 pairs, USD excluded...")
    pred_45, pairs_45 = compute_pairwise(pred, config.n_ccy, exclude_usd=True, usd_idx=config.usd_idx)
    actual_45, _ = compute_pairwise(Y_test, config.n_ccy, exclude_usd=True, usd_idx=config.usd_idx)
    results_45, plot_45 = run_backtest_suite(pred_45, actual_45, len(pairs_45), fracs, costs)
    plot_trading_with_costs(plot_45, len(pairs_45), fracs, costs,
                           f"{len(pairs_45)} Pairs (USD excluded)", output_name="trading_45pairs.png")
    plot_trading_with_costs(plot_45, len(pairs_45), fracs, costs,
                           f"{len(pairs_45)} Pairs (USD excluded)", output_name="trading_confidence_no_usd.png")
    all_results['45_pairs_no_usd'] = results_45

    # --- 45 pairs quintile ---
    print("[3/4] 45 pairs, quintile filter...")
    results_q = plot_trading_quintile(pred_45, actual_45, len(pairs_45), costs,
                                     output_name="trading_usd_excluded.png")
    all_results['45_pairs_quintile'] = results_q

    # --- Per-pair hit rates ---
    print("[4/4] Per-pair statistics...")
    pair_stats = compute_pair_hit_rates(pred, Y_test, config)
    all_results['pair_stats'] = pair_stats

    sorted_pairs = sorted(pair_stats.items(), key=lambda x: x[1]['reliability'], reverse=True)
    print(f"\n  Top 5 reliable pairs:")
    for name, s in sorted_pairs[:5]:
        print(f"    {name}: hit={s['hit_rate']:.3f}, strength={s['avg_signal_strength']:.5f}")
    print(f"  Bottom 5:")
    for name, s in sorted_pairs[-5:]:
        print(f"    {name}: hit={s['hit_rate']:.3f}, strength={s['avg_signal_strength']:.5f}")

    # Strategy comparison tables
    print_strategy_table(results_55, len(pairs_55), fracs, "0bps")
    print_strategy_table(results_55, len(pairs_55), fracs, "5bps")
    print_strategy_table(results_55, len(pairs_55), fracs, "10bps")

    # Save JSON
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'eval_trading_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
