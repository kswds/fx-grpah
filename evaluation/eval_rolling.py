"""
Rolling Window Backtest for Ridge

Train on 4-year window, test on next 1 year, step by 1 year.
Checks whether Ridge performance is stable across time periods.

Outputs:
  - results/eval_rolling_results.json
  - results/rolling_backtest.png
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
from dataset import load_data, build_features
from eval_trading import trading_backtest, compute_pairwise, compute_stats

MACRO_FEATURES = ["Global_VIX", "Global_Gold", "Global_Oil",
                  "Global_Copper", "Global_US2Y", "Global_IronOre"]

TRAIN_DAYS = 1000  # ~4 years
TEST_DAYS = 250    # ~1 year


def run_rolling_backtest():
    config = Config(seed=42, lookback=20)
    df = load_data(config)
    X_local_base, X_macro_full, Y = build_features(df, config)

    macro_idx = [config.global_features.index(f) for f in MACRO_FEATURES]
    X_macro = X_macro_full[:, macro_idx]

    L = config.lookback
    n_total = len(X_local_base)

    # Build sample-level arrays (features = last day of lookback window, target = next day)
    n_samples = n_total - L
    X_local_all = np.stack([X_local_base[idx + L - 1] for idx in range(n_samples)])  # [n, N, 3]
    X_macro_all = np.stack([X_macro[idx + L - 1] for idx in range(n_samples)])        # [n, M]
    Y_all = np.stack([Y[idx + L] for idx in range(n_samples)])                        # [n, N]

    # Flatten local features for Ridge
    X_local_flat = X_local_all.reshape(n_samples, -1)  # [n, N*3]

    # Get dates for reporting
    dates = df["Date"].values
    # After build_features drops first row (diff), and lookback offset:
    # sample idx corresponds to date at index (idx + L + 1) in original df
    # (1 row lost to diff in build_features, L rows for lookback)
    sample_dates = dates[L + 1: L + 1 + n_samples]

    print("=" * 70)
    print("Rolling Window Backtest: Ridge")
    print("=" * 70)
    print(f"Total samples: {n_samples}, Train window: {TRAIN_DAYS}, Test window: {TEST_DAYS}")
    print(f"Date range: {str(sample_dates[0])[:10]} ~ {str(sample_dates[-1])[:10]}")

    # Generate folds
    folds = []
    start = 0
    while start + TRAIN_DAYS + TEST_DAYS <= n_samples:
        train_end = start + TRAIN_DAYS
        test_end = min(train_end + TEST_DAYS, n_samples)
        folds.append((start, train_end, test_end))
        start += TEST_DAYS  # step by 1 year

    print(f"Number of folds: {len(folds)}\n")

    # Run each fold
    all_fold_results = []
    all_navs = []

    for fold_idx, (tr_start, tr_end, te_end) in enumerate(folds):
        # Normalize using this fold's train data
        X_local_train = X_local_flat[tr_start:tr_end]
        X_macro_train = X_macro_all[tr_start:tr_end]

        local_mean = X_local_train.mean(axis=0, keepdims=True)
        local_std = X_local_train.std(axis=0, keepdims=True) + 1e-6
        macro_mean = X_macro_train.mean(axis=0, keepdims=True)
        macro_std = X_macro_train.std(axis=0, keepdims=True) + 1e-6

        # Normalize both train and test with train stats
        X_local_norm = (X_local_flat - local_mean) / local_std
        X_macro_norm = (X_macro_all - macro_mean) / macro_std

        X_train = np.concatenate([X_local_norm[tr_start:tr_end], X_macro_norm[tr_start:tr_end]], axis=1)
        X_test = np.concatenate([X_local_norm[tr_end:te_end], X_macro_norm[tr_end:te_end]], axis=1)
        Y_train = Y_all[tr_start:tr_end]
        Y_test = Y_all[tr_end:te_end]

        # Train Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, Y_train)
        pred = ridge.predict(X_test)

        n_test = len(Y_test)
        test_dates = sample_dates[tr_end:te_end]

        # Hit rates
        edges = list(combinations(range(config.n_ccy), 2))
        pred_pairs = np.stack([pred[:, i] - pred[:, j] for i, j in edges], axis=1)
        actual_pairs = np.stack([Y_test[:, i] - Y_test[:, j] for i, j in edges], axis=1)

        mask = np.ones(config.n_ccy, dtype=bool)
        mask[config.usd_idx] = False
        hit_ccy = np.mean(np.sign(pred[:, mask]) == np.sign(Y_test[:, mask]))
        hit_55 = np.mean(np.sign(pred_pairs) == np.sign(actual_pairs))

        # Trading backtests
        fold_trading = {}
        for cost in [0, 5, 10]:
            cost_key = f"{cost}bps"
            fold_trading[cost_key] = {}
            for frac, label in [(1.0, "all"), (0.3, "top30")]:
                nav, daily, hit_rate = trading_backtest(pred_pairs, actual_pairs, frac, cost_bps=cost)
                stats = compute_stats(daily, nav, hit_rate)
                fold_trading[cost_key][label] = stats
                if cost == 0 and frac == 1.0:
                    all_navs.append((fold_idx, nav, str(test_dates[0])[:10], str(test_dates[-1])[:10]))

        fold_result = {
            'fold': fold_idx,
            'train_period': f"{str(sample_dates[tr_start])[:10]} ~ {str(sample_dates[tr_end-1])[:10]}",
            'test_period': f"{str(test_dates[0])[:10]} ~ {str(test_dates[-1])[:10]}",
            'n_test': n_test,
            'hit_ccy': round(float(hit_ccy), 4),
            'hit_55': round(float(hit_55), 4),
            'trading': fold_trading,
        }
        all_fold_results.append(fold_result)

        # Print fold summary
        t0 = fold_trading['0bps']['all']
        t5 = fold_trading['5bps']['all']
        t0_top = fold_trading['0bps']['top30']
        print(f"Fold {fold_idx}: {fold_result['test_period']} ({n_test}d)")
        print(f"  Hit(ccy)={hit_ccy:.3f}  Hit(55)={hit_55:.3f}")
        print(f"  0bps All: CAGR={t0['cagr_pct']:.1f}% Sharpe={t0['sharpe_ratio']:.2f}")
        print(f"  0bps Top30: CAGR={t0_top['cagr_pct']:.1f}% Sharpe={t0_top['sharpe_ratio']:.2f}")
        print(f"  5bps All: CAGR={t5['cagr_pct']:.1f}% Sharpe={t5['sharpe_ratio']:.2f}")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY (mean ± std across folds)")
    print("=" * 70)

    hit_ccys = [f['hit_ccy'] for f in all_fold_results]
    hit_55s = [f['hit_55'] for f in all_fold_results]
    print(f"Hit(ccy):  {np.mean(hit_ccys):.3f} ± {np.std(hit_ccys):.3f}")
    print(f"Hit(55):   {np.mean(hit_55s):.3f} ± {np.std(hit_55s):.3f}")

    for cost_key in ['0bps', '5bps', '10bps']:
        for strat in ['all', 'top30']:
            cagrs = [f['trading'][cost_key][strat]['cagr_pct'] for f in all_fold_results]
            sharpes = [f['trading'][cost_key][strat]['sharpe_ratio'] for f in all_fold_results]
            totals = [f['trading'][cost_key][strat]['total_return_pct'] for f in all_fold_results]
            label = f"{cost_key} {strat}"
            print(f"  {label:<16}: CAGR={np.mean(cagrs):>7.1f}% ± {np.std(cagrs):>5.1f}%  "
                  f"Sharpe={np.mean(sharpes):>5.2f} ± {np.std(sharpes):.2f}  "
                  f"Total={np.mean(totals):>7.1f}%")

    # Plot fold NAVs
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    for fold_idx, nav, d0, d1 in all_navs:
        ax.plot(nav, label=f"Fold {fold_idx}: {d0}~{d1}")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("NAV (start=1.0)")
    ax.set_title("Rolling Window Backtest — Per-Fold NAV (0bps, All 55 pairs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)

    out_png = os.path.join(os.path.dirname(__file__), '..', 'results', 'rolling_backtest.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")

    # Save JSON
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'train_days': TRAIN_DAYS,
            'test_days': TEST_DAYS,
            'n_folds': len(folds),
        },
        'summary': {
            'hit_ccy': {'mean': round(np.mean(hit_ccys), 4), 'std': round(np.std(hit_ccys), 4)},
            'hit_55': {'mean': round(np.mean(hit_55s), 4), 'std': round(np.std(hit_55s), 4)},
        },
        'folds': all_fold_results,
    }

    out_json = os.path.join(os.path.dirname(__file__), '..', 'results', 'eval_rolling_results.json')
    with open(out_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    run_rolling_backtest()
