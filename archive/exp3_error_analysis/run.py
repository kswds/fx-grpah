"""
Exp3: Error Analysis
- Which currencies are hard to predict?
- When does the model fail?
- Correlation with market conditions
"""
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

import random
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from dataset import create_dataloaders, fully_connected_edge_index
from models import FXStrengthGNN
from train import Trainer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_predictions(model, test_loader, edge_index, device, config):
    """Get all predictions and targets"""
    model.eval()

    all_preds = []
    all_targets = []
    all_macro = []

    with torch.no_grad():
        for xl, xm, yb in test_loader:
            xl, xm, yb = xl.to(device), xm.to(device), yb.to(device)
            rhat, ds, _, _ = model(xl, xm, edge_index)

            all_preds.append(rhat.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
            all_macro.append(xm[:, -1, :].cpu().numpy())  # Last timestep macro

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    macro = np.concatenate(all_macro, axis=0)

    return preds, targets, macro


def analyze_by_currency(preds, targets, config):
    """Analyze errors per currency"""
    ccys = config.ccys
    n_ccy = config.n_ccy
    usd_idx = config.usd_idx

    results = {}

    for i, ccy in enumerate(ccys):
        if i == usd_idx:
            continue

        pred_i = preds[:, i]
        target_i = targets[:, i]

        rmse = np.sqrt(np.mean((pred_i - target_i) ** 2))
        mae = np.mean(np.abs(pred_i - target_i))

        # Directional accuracy
        hit = np.mean(np.sign(pred_i) == np.sign(target_i))

        # Correlation
        corr = np.corrcoef(pred_i, target_i)[0, 1]

        results[ccy] = {
            'rmse': rmse,
            'mae': mae,
            'hit': hit,
            'corr': corr,
            'pred_std': np.std(pred_i),
            'target_std': np.std(target_i),
        }

    return results


def analyze_by_macro_regime(preds, targets, macro, config):
    """Analyze errors by macro conditions"""
    macro_features = config.global_features

    # Focus on VIX (index 1) for regime analysis
    vix_idx = macro_features.index("Global_VIX")
    vix = macro[:, vix_idx]

    # Split into high/low VIX
    vix_median = np.median(vix)
    high_vix_mask = vix > vix_median
    low_vix_mask = ~high_vix_mask

    # Exclude USD
    mask = np.ones(config.n_ccy, dtype=bool)
    mask[config.usd_idx] = False

    results = {}

    # High VIX regime
    preds_high = preds[high_vix_mask][:, mask]
    targets_high = targets[high_vix_mask][:, mask]
    rmse_high = np.sqrt(np.mean((preds_high - targets_high) ** 2))
    hit_high = np.mean(np.sign(preds_high) == np.sign(targets_high))

    # Low VIX regime
    preds_low = preds[low_vix_mask][:, mask]
    targets_low = targets[low_vix_mask][:, mask]
    rmse_low = np.sqrt(np.mean((preds_low - targets_low) ** 2))
    hit_low = np.mean(np.sign(preds_low) == np.sign(targets_low))

    results['high_vix'] = {'rmse': rmse_high, 'hit': hit_high, 'n_samples': high_vix_mask.sum()}
    results['low_vix'] = {'rmse': rmse_low, 'hit': hit_low, 'n_samples': low_vix_mask.sum()}

    return results


def analyze_error_vs_magnitude(preds, targets, config):
    """Analyze if errors correlate with prediction magnitude"""
    mask = np.ones(config.n_ccy, dtype=bool)
    mask[config.usd_idx] = False

    preds_flat = preds[:, mask].flatten()
    targets_flat = targets[:, mask].flatten()
    errors = np.abs(preds_flat - targets_flat)

    # Bin by prediction magnitude
    pred_abs = np.abs(preds_flat)

    bins = np.percentile(pred_abs, [0, 25, 50, 75, 100])
    results = {}

    for i in range(4):
        mask_bin = (pred_abs >= bins[i]) & (pred_abs < bins[i+1])
        if mask_bin.sum() > 0:
            results[f'q{i+1}'] = {
                'mean_error': errors[mask_bin].mean(),
                'mean_pred_magnitude': pred_abs[mask_bin].mean(),
                'hit_rate': (np.sign(preds_flat[mask_bin]) == np.sign(targets_flat[mask_bin])).mean(),
                'n_samples': mask_bin.sum(),
            }

    return results


def analyze_a_matrix(model, config):
    """Analyze learned A matrix"""
    A = model.A.detach().cpu().numpy()

    ccys = config.ccys
    macro_features = [f.replace("Global_", "") for f in config.global_features]

    results = {
        'matrix': A.tolist(),
        'currencies': ccys,
        'macro_factors': macro_features,
    }

    # Find strongest sensitivities
    strongest = []
    for i, ccy in enumerate(ccys):
        for j, macro in enumerate(macro_features):
            strongest.append({
                'currency': ccy,
                'macro': macro,
                'sensitivity': A[i, j],
                'abs_sensitivity': abs(A[i, j]),
            })

    strongest = sorted(strongest, key=lambda x: x['abs_sensitivity'], reverse=True)
    results['top_sensitivities'] = strongest[:20]

    return results


def create_visualizations(preds, targets, macro, model, config, output_dir):
    """Create visualization plots"""

    # 1. Currency-wise performance
    ccy_results = analyze_by_currency(preds, targets, config)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ccys = [c for c in config.ccys if c != "USD"]
    rmses = [ccy_results[c]['rmse'] for c in ccys]
    hits = [ccy_results[c]['hit'] for c in ccys]

    axes[0].bar(ccys, rmses)
    axes[0].set_title('RMSE by Currency')
    axes[0].set_ylabel('RMSE')
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(ccys, hits)
    axes[1].axhline(y=0.5, color='r', linestyle='--', label='Random')
    axes[1].set_title('Hit Rate by Currency')
    axes[1].set_ylabel('Hit Rate')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/currency_performance.png', dpi=150)
    plt.close()

    # 2. A matrix heatmap
    A = model.A.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))

    macro_labels = [f.replace("Global_", "") for f in config.global_features]

    sns.heatmap(A,
                xticklabels=macro_labels,
                yticklabels=config.ccys,
                cmap='RdBu_r',
                center=0,
                annot=True,
                fmt='.2f',
                ax=ax)

    ax.set_title('Learned A Matrix\n(Currency Sensitivity to Macro Factors)')
    ax.set_xlabel('Macro Factor')
    ax.set_ylabel('Currency')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/a_matrix_heatmap.png', dpi=150)
    plt.close()

    print(f"Visualizations saved to {output_dir}/")


def main():
    seed = 42
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = Config(seed=seed, epochs=30, batch_size=128, lr=3e-4)
    config.use_skip_connection = True
    config.use_layer_norm = True

    edge_index = fully_connected_edge_index(config.n_ccy).to(device)

    # Train model
    print("\n>>> Training model...")
    set_seed(seed)
    train_loader, test_loader = create_dataloaders(config, macro_mode="real")
    model = FXStrengthGNN(config)
    trainer = Trainer(model, config, device)
    trainer.train(train_loader, test_loader, edge_index, label="FULL")

    # Get predictions
    print("\n>>> Getting predictions...")
    preds, targets, macro = get_predictions(model, test_loader, edge_index, device, config)

    print(f"Predictions shape: {preds.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Macro shape: {macro.shape}")

    # Analysis
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    # 1. By currency
    print("\n--- Performance by Currency ---")
    ccy_results = analyze_by_currency(preds, targets, config)

    print(f"{'Currency':<8} {'RMSE':>8} {'MAE':>8} {'Hit':>8} {'Corr':>8}")
    print("-" * 40)
    for ccy in sorted(ccy_results.keys(), key=lambda x: ccy_results[x]['rmse']):
        r = ccy_results[ccy]
        print(f"{ccy:<8} {r['rmse']:>8.4f} {r['mae']:>8.4f} {r['hit']:>8.4f} {r['corr']:>8.4f}")

    # 2. By VIX regime
    print("\n--- Performance by VIX Regime ---")
    regime_results = analyze_by_macro_regime(preds, targets, macro, config)

    for regime, r in regime_results.items():
        print(f"{regime}: RMSE={r['rmse']:.4f}, Hit={r['hit']:.4f}, n={r['n_samples']}")

    # 3. By prediction magnitude
    print("\n--- Error vs Prediction Magnitude ---")
    mag_results = analyze_error_vs_magnitude(preds, targets, config)

    for q, r in mag_results.items():
        print(f"{q}: mean_error={r['mean_error']:.4f}, hit={r['hit_rate']:.4f}, "
              f"pred_mag={r['mean_pred_magnitude']:.4f}")

    # 4. A matrix analysis
    print("\n--- A Matrix Analysis (Top Sensitivities) ---")
    a_results = analyze_a_matrix(model, config)

    print(f"{'Currency':<6} {'Macro':<8} {'Sensitivity':>12}")
    print("-" * 30)
    for item in a_results['top_sensitivities'][:15]:
        print(f"{item['currency']:<6} {item['macro']:<8} {item['sensitivity']:>12.4f}")

    # Create visualizations
    output_dir = "exp3_error_analysis"
    create_visualizations(preds, targets, macro, model, config, output_dir)

    # Save results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'by_currency': ccy_results,
        'by_regime': regime_results,
        'by_magnitude': mag_results,
        'a_matrix': a_results,
    }

    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
