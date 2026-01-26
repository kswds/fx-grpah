"""
Experiment 13: Error Pattern Analysis
=====================================
예측이 틀린 것들의 패턴 분석

1. Per-currency 방향 정확도: 특정 통화가 체계적으로 틀리나?
2. 시간적 패턴: 틀린 예측이 특정 시기에 몰려있나?
3. Lag 분석: 방향은 맞는데 타이밍이 1일 밀린건 아닌가?
4. Cycle 분석: 오류에 주기성이 있나?
5. 연속 오류 분석: 같은 방향으로 계속 틀리나?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

from config import Config
from models import FXStrengthGNN
from dataset import load_data, build_features, FXDataset, fully_connected_edge_index, create_dataloaders
from train import Trainer


def get_predictions_and_targets(model, test_loader, edge_index, device, config):
    """Get all predictions and targets from test set"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xl, xm, yb in test_loader:
            xl, xm = xl.to(device), xm.to(device)
            rhat, ds, z_ccy, m_msg = model(xl, xm, edge_index)
            all_preds.append(rhat.cpu().numpy())
            all_targets.append(yb.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return preds, targets


def analyze_per_currency_accuracy(preds, targets, config):
    """Per-currency directional accuracy analysis"""
    results = {}

    for i, ccy in enumerate(config.ccys):
        if ccy == "USD":
            continue

        pred_sign = np.sign(preds[:, i])
        target_sign = np.sign(targets[:, i])

        correct = (pred_sign == target_sign)
        accuracy = correct.mean()

        # Bias analysis: does model predict one direction more often?
        pred_positive_rate = (pred_sign > 0).mean()
        target_positive_rate = (target_sign > 0).mean()

        # When wrong, which direction is it usually?
        wrong_mask = ~correct
        if wrong_mask.sum() > 0:
            # Model predicted positive but was negative
            false_positive = ((pred_sign > 0) & (target_sign < 0)).sum()
            # Model predicted negative but was positive
            false_negative = ((pred_sign < 0) & (target_sign > 0)).sum()
        else:
            false_positive = 0
            false_negative = 0

        results[ccy] = {
            "accuracy": float(accuracy),
            "pred_positive_rate": float(pred_positive_rate),
            "target_positive_rate": float(target_positive_rate),
            "bias": float(pred_positive_rate - target_positive_rate),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "n_wrong": int(wrong_mask.sum()),
        }

    return results


def analyze_lag_effect(preds, targets, config, max_lag=5):
    """
    Lag analysis: 방향은 맞는데 타이밍이 밀린건 아닌가?
    - 예측이 실제보다 1일 뒤처지거나 앞서는지 확인
    """
    results = {}

    for i, ccy in enumerate(config.ccys):
        if ccy == "USD":
            continue

        pred_sign = np.sign(preds[:, i])
        target_sign = np.sign(targets[:, i])

        lag_accuracies = {}

        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                # Normal accuracy
                acc = (pred_sign == target_sign).mean()
            elif lag > 0:
                # Prediction is ahead: compare pred[t] with target[t+lag]
                acc = (pred_sign[:-lag] == target_sign[lag:]).mean()
            else:
                # Prediction is behind: compare pred[t] with target[t+lag]
                acc = (pred_sign[-lag:] == target_sign[:lag]).mean()

            lag_accuracies[lag] = float(acc)

        # Find best lag
        best_lag = max(lag_accuracies, key=lag_accuracies.get)
        best_acc = lag_accuracies[best_lag]

        results[ccy] = {
            "lag_accuracies": lag_accuracies,
            "best_lag": best_lag,
            "best_lag_accuracy": best_acc,
            "current_accuracy": lag_accuracies[0],
            "improvement_with_lag": best_acc - lag_accuracies[0],
        }

    return results


def analyze_temporal_error_pattern(preds, targets, config):
    """
    시간에 따른 오류 패턴 분석
    - 오류가 특정 시기에 몰려있나?
    - 연속 오류 streak
    """
    mask = np.ones(config.n_ccy, dtype=bool)
    mask[config.usd_idx] = False

    # Overall correctness per timestep
    correct = (np.sign(preds[:, mask]) == np.sign(targets[:, mask]))
    accuracy_per_time = correct.mean(axis=1)  # Average accuracy across currencies

    # Rolling accuracy (20-day window)
    rolling_window = 20
    rolling_acc = pd.Series(accuracy_per_time).rolling(rolling_window).mean()

    # Find periods of poor performance
    poor_threshold = 0.5
    poor_periods = accuracy_per_time < poor_threshold
    poor_streaks = []
    streak_start = None

    for t in range(len(poor_periods)):
        if poor_periods[t] and streak_start is None:
            streak_start = t
        elif not poor_periods[t] and streak_start is not None:
            poor_streaks.append((streak_start, t - 1, t - streak_start))
            streak_start = None

    # Error autocorrelation (do errors cluster?)
    errors = 1 - accuracy_per_time
    autocorr = np.correlate(errors - errors.mean(), errors - errors.mean(), mode='full')
    autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]

    return {
        "accuracy_per_time": accuracy_per_time.tolist(),
        "rolling_accuracy": rolling_acc.dropna().tolist(),
        "poor_streaks": poor_streaks[:10],  # Top 10 longest
        "mean_accuracy": float(accuracy_per_time.mean()),
        "std_accuracy": float(accuracy_per_time.std()),
        "autocorr_lag1": float(autocorr[1]) if len(autocorr) > 1 else 0,
        "autocorr_lag5": float(autocorr[5]) if len(autocorr) > 5 else 0,
    }


def analyze_cycle_in_errors(preds, targets, config):
    """
    FFT로 오류에 주기성이 있는지 분석
    """
    mask = np.ones(config.n_ccy, dtype=bool)
    mask[config.usd_idx] = False

    correct = (np.sign(preds[:, mask]) == np.sign(targets[:, mask]))
    errors = 1 - correct.mean(axis=1)

    # FFT
    n = len(errors)
    yf = fft(errors - errors.mean())
    xf = fftfreq(n, 1)[:n//2]
    power = 2.0/n * np.abs(yf[0:n//2])

    # Find dominant frequencies
    peaks, _ = find_peaks(power, height=power.mean() + power.std())
    dominant_periods = []
    for peak in peaks[:5]:  # Top 5 peaks
        if xf[peak] > 0:
            period = 1 / xf[peak]
            dominant_periods.append({
                "period_days": float(period),
                "power": float(power[peak]),
            })

    return {
        "dominant_periods": dominant_periods,
        "has_weekly_cycle": any(5 < p["period_days"] < 7 for p in dominant_periods),
        "has_monthly_cycle": any(20 < p["period_days"] < 25 for p in dominant_periods),
    }


def analyze_consecutive_errors(preds, targets, config):
    """
    연속 오류 분석: 같은 통화가 연속으로 같은 방향으로 틀리나?
    """
    results = {}

    for i, ccy in enumerate(config.ccys):
        if ccy == "USD":
            continue

        pred_sign = np.sign(preds[:, i])
        target_sign = np.sign(targets[:, i])
        correct = (pred_sign == target_sign)

        # Find consecutive error streaks
        error_streaks = []
        streak_len = 0
        streak_direction = None

        for t in range(len(correct)):
            if not correct[t]:
                if streak_len == 0:
                    streak_direction = int(pred_sign[t])
                streak_len += 1
            else:
                if streak_len > 0:
                    error_streaks.append((streak_len, streak_direction))
                streak_len = 0
                streak_direction = None

        # Statistics
        if error_streaks:
            max_streak = max(s[0] for s in error_streaks)
            avg_streak = np.mean([s[0] for s in error_streaks])
            # Same direction bias in streaks
            pos_streaks = sum(1 for s in error_streaks if s[1] > 0)
            neg_streaks = sum(1 for s in error_streaks if s[1] < 0)
        else:
            max_streak = 0
            avg_streak = 0
            pos_streaks = 0
            neg_streaks = 0

        results[ccy] = {
            "max_error_streak": int(max_streak),
            "avg_error_streak": float(avg_streak),
            "positive_direction_streaks": pos_streaks,
            "negative_direction_streaks": neg_streaks,
            "direction_bias": "positive" if pos_streaks > neg_streaks * 1.2 else
                             ("negative" if neg_streaks > pos_streaks * 1.2 else "balanced"),
        }

    return results


def analyze_cross_currency_error_correlation(preds, targets, config):
    """
    통화 간 오류 상관관계: 특정 통화들이 같이 틀리나?
    """
    mask = np.ones(config.n_ccy, dtype=bool)
    mask[config.usd_idx] = False
    ccys = [c for c in config.ccys if c != "USD"]

    # Error matrix (1 = wrong, 0 = correct)
    errors = (np.sign(preds[:, mask]) != np.sign(targets[:, mask])).astype(float)

    # Correlation matrix
    corr_matrix = np.corrcoef(errors.T)

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(ccys)):
        for j in range(i + 1, len(ccys)):
            if abs(corr_matrix[i, j]) > 0.3:
                high_corr_pairs.append({
                    "pair": (ccys[i], ccys[j]),
                    "correlation": float(corr_matrix[i, j]),
                })

    return {
        "correlation_matrix": corr_matrix.tolist(),
        "currencies": ccys,
        "high_corr_pairs": high_corr_pairs,
    }


def create_error_analysis_plots(results, config, save_path):
    """Create comprehensive error analysis visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ccys = [c for c in config.ccys if c != "USD"]

    # 1. Per-currency accuracy
    ax1 = axes[0, 0]
    accuracies = [results["per_currency"][c]["accuracy"] for c in ccys]
    colors = ['green' if a > 0.6 else 'orange' if a > 0.5 else 'red' for a in accuracies]
    bars = ax1.bar(ccys, accuracies, color=colors, alpha=0.7)
    ax1.axhline(y=0.5, color='red', linestyle='--', label='Random')
    ax1.axhline(y=0.65, color='green', linestyle='--', label='Target 65%')
    ax1.set_ylabel("Directional Accuracy")
    ax1.set_title("Per-Currency Accuracy", fontweight='bold')
    ax1.set_xticklabels(ccys, rotation=45, ha='right')
    ax1.legend()

    # 2. Lag effect
    ax2 = axes[0, 1]
    current_acc = [results["lag_effect"][c]["current_accuracy"] for c in ccys]
    best_acc = [results["lag_effect"][c]["best_lag_accuracy"] for c in ccys]
    best_lags = [results["lag_effect"][c]["best_lag"] for c in ccys]

    x = np.arange(len(ccys))
    width = 0.35
    ax2.bar(x - width/2, current_acc, width, label='Current (lag=0)', alpha=0.7)
    ax2.bar(x + width/2, best_acc, width, label='Best lag', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ccys, rotation=45, ha='right')
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Lag Effect Analysis", fontweight='bold')
    ax2.legend()

    # Add best lag labels
    for i, (lag, acc) in enumerate(zip(best_lags, best_acc)):
        if lag != 0:
            ax2.text(i + width/2, acc + 0.01, f'lag={lag}', ha='center', fontsize=8)

    # 3. Temporal error pattern
    ax3 = axes[0, 2]
    rolling_acc = results["temporal"]["rolling_accuracy"]
    ax3.plot(rolling_acc, alpha=0.7)
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(y=0.65, color='green', linestyle='--', alpha=0.5)
    ax3.fill_between(range(len(rolling_acc)), 0.5, rolling_acc,
                     where=np.array(rolling_acc) < 0.5, color='red', alpha=0.3)
    ax3.set_xlabel("Time (test period)")
    ax3.set_ylabel("Rolling Accuracy (20-day)")
    ax3.set_title("Temporal Error Pattern", fontweight='bold')

    # 4. Prediction bias
    ax4 = axes[1, 0]
    pred_pos = [results["per_currency"][c]["pred_positive_rate"] for c in ccys]
    target_pos = [results["per_currency"][c]["target_positive_rate"] for c in ccys]
    bias = [results["per_currency"][c]["bias"] for c in ccys]

    x = np.arange(len(ccys))
    ax4.bar(x - width/2, pred_pos, width, label='Pred positive rate', alpha=0.7)
    ax4.bar(x + width/2, target_pos, width, label='Actual positive rate', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(ccys, rotation=45, ha='right')
    ax4.set_ylabel("Positive Rate")
    ax4.set_title("Directional Bias Analysis", fontweight='bold')
    ax4.legend()
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # 5. Error correlation heatmap
    ax5 = axes[1, 1]
    corr_matrix = np.array(results["cross_currency"]["correlation_matrix"])
    im = ax5.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax5.set_xticks(range(len(ccys)))
    ax5.set_xticklabels(ccys, rotation=45, ha='right')
    ax5.set_yticks(range(len(ccys)))
    ax5.set_yticklabels(ccys)
    ax5.set_title("Error Correlation Across Currencies", fontweight='bold')
    plt.colorbar(im, ax=ax5)

    # 6. Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')

    # Find problematic patterns
    worst_ccy = min(ccys, key=lambda c: results["per_currency"][c]["accuracy"])
    worst_acc = results["per_currency"][worst_ccy]["accuracy"]

    lag_issues = [(c, results["lag_effect"][c]["best_lag"])
                  for c in ccys if results["lag_effect"][c]["best_lag"] != 0]

    biased_ccys = [(c, results["per_currency"][c]["bias"])
                   for c in ccys if abs(results["per_currency"][c]["bias"]) > 0.1]

    summary = f"""
    Error Pattern Analysis Summary
    ==============================

    Worst Currency: {worst_ccy} ({worst_acc:.1%} accuracy)

    Lag Issues (best lag ≠ 0):
    {chr(10).join([f"  • {c}: best at lag={lag}" for c, lag in lag_issues]) if lag_issues else "  None detected"}

    Directional Bias (|bias| > 10%):
    {chr(10).join([f"  • {c}: {'+' if b > 0 else ''}{b:.1%}" for c, b in biased_ccys]) if biased_ccys else "  None detected"}

    Temporal Pattern:
    • Error autocorr(lag=1): {results["temporal"]["autocorr_lag1"]:.3f}
    • Error autocorr(lag=5): {results["temporal"]["autocorr_lag5"]:.3f}

    Cycle Detection:
    • Weekly cycle: {"Yes" if results["cycle"]["has_weekly_cycle"] else "No"}
    • Monthly cycle: {"Yes" if results["cycle"]["has_monthly_cycle"] else "No"}

    High Error Correlation Pairs:
    {chr(10).join([f"  • {p['pair'][0]}-{p['pair'][1]}: {p['correlation']:.2f}" for p in results["cross_currency"]["high_corr_pairs"][:5]]) if results["cross_currency"]["high_corr_pairs"] else "  None (errors are independent)"}
    """

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("Experiment 13: Error Pattern Analysis")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config()

    # Train model
    print("\n[1/6] Training model...")
    train_loader, test_loader = create_dataloaders(config)
    edge_index = fully_connected_edge_index(config.n_ccy).to(device)

    model = FXStrengthGNN(config)
    trainer = Trainer(model, config, device)
    trainer.train(train_loader, test_loader, edge_index, label="Full Model")

    # Get predictions
    print("\n[2/6] Getting predictions...")
    preds, targets = get_predictions_and_targets(model, test_loader, edge_index, device, config)

    # Analyses
    print("\n[3/6] Per-currency accuracy analysis...")
    per_currency = analyze_per_currency_accuracy(preds, targets, config)

    print("\n[4/6] Lag effect analysis...")
    lag_effect = analyze_lag_effect(preds, targets, config)

    print("\n[5/6] Temporal & cycle analysis...")
    temporal = analyze_temporal_error_pattern(preds, targets, config)
    cycle = analyze_cycle_in_errors(preds, targets, config)
    consecutive = analyze_consecutive_errors(preds, targets, config)
    cross_currency = analyze_cross_currency_error_correlation(preds, targets, config)

    # Compile results
    results = {
        "per_currency": per_currency,
        "lag_effect": lag_effect,
        "temporal": {k: v for k, v in temporal.items() if k not in ['accuracy_per_time', 'rolling_accuracy']},
        "cycle": cycle,
        "consecutive": consecutive,
        "cross_currency": {k: v for k, v in cross_currency.items() if k != 'correlation_matrix'},
    }

    # Full results for plotting
    full_results = {
        "per_currency": per_currency,
        "lag_effect": lag_effect,
        "temporal": temporal,
        "cycle": cycle,
        "consecutive": consecutive,
        "cross_currency": cross_currency,
    }

    # Create visualizations
    print("\n[6/6] Creating visualizations...")
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(exp_dir, exist_ok=True)

    create_error_analysis_plots(full_results, config, os.path.join(exp_dir, "error_analysis.png"))

    # Save results
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 13 SUMMARY")
    print("=" * 60)

    print("\n📊 Per-Currency Accuracy:")
    for ccy in [c for c in config.ccys if c != "USD"]:
        acc = per_currency[ccy]["accuracy"]
        bias = per_currency[ccy]["bias"]
        status = "✓" if acc > 0.6 else "⚠" if acc > 0.5 else "✗"
        print(f"  {status} {ccy:4s}: {acc:.1%} (bias: {bias:+.1%})")

    print("\n🕐 Lag Effect (best_lag ≠ 0 means timing issue):")
    for ccy in [c for c in config.ccys if c != "USD"]:
        best_lag = lag_effect[ccy]["best_lag"]
        improvement = lag_effect[ccy]["improvement_with_lag"]
        if best_lag != 0:
            print(f"  ⚠ {ccy:4s}: best_lag={best_lag:+d} (+{improvement:.1%} accuracy)")

    print("\n🔄 Cycle Detection:")
    print(f"  Weekly cycle: {'⚠ YES' if cycle['has_weekly_cycle'] else '✓ No'}")
    print(f"  Monthly cycle: {'⚠ YES' if cycle['has_monthly_cycle'] else '✓ No'}")

    print("\n🔗 Cross-Currency Error Correlation:")
    if cross_currency["high_corr_pairs"]:
        for pair in cross_currency["high_corr_pairs"][:3]:
            print(f"  ⚠ {pair['pair'][0]}-{pair['pair'][1]}: {pair['correlation']:.2f}")
    else:
        print("  ✓ No strong correlations (errors are independent)")

    print("\n✅ Outputs saved:")
    print("  - error_analysis.png")
    print("  - results.json")


if __name__ == "__main__":
    main()
