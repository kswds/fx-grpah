"""
Exp14: Statistical Significance of A Matrix Coefficients

Method: Bootstrap with multiple seeds
- Train model N times with different seeds
- Collect A matrix from each run
- Compute mean, std, t-stat, p-value for each coefficient
- Identify statistically significant macro-currency relationships
"""

import sys
sys.path.append('/home/gdro/experiment/fx-graph-other')

import random
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from scipy import stats

from config import Config
from dataset import create_dataloaders, fully_connected_edge_index
from models import FXStrengthGNN
from train import Trainer

cfg = Config()
CURRENCIES = cfg.ccys
MACRO_FEATURES = [f.replace('Global_', '') for f in cfg.global_features]

N_BOOTSTRAP = 10  # Number of bootstrap runs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_get_A(seed, device, edge_index):
    """Train model with given seed and return A matrix"""
    set_seed(seed)
    train_loader, test_loader = create_dataloaders(cfg, macro_mode="real")
    model = FXStrengthGNN(cfg)
    trainer = Trainer(model, cfg, device)

    # Train silently
    for epoch in range(cfg.epochs):
        trainer.train_epoch(train_loader, edge_index)

    # Get A matrix
    A = model.A.detach().cpu().numpy()

    # Also get hit rate for reference
    metrics = trainer.evaluate(test_loader, edge_index)

    return A, metrics['hit']


def main():
    print("=" * 70)
    print("EXP14: STATISTICAL SIGNIFICANCE OF A MATRIX")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Bootstrap runs: {N_BOOTSTRAP}")

    output_dir = 'exp14_significance'
    edge_index = fully_connected_edge_index(cfg.n_ccy).to(device)

    # Collect A matrices from multiple runs
    all_A = []
    all_hits = []

    for i in range(N_BOOTSTRAP):
        seed = 42 + i * 7  # Different seeds
        print(f"\n>>> Run {i+1}/{N_BOOTSTRAP} (seed={seed})...")
        A, hit = train_and_get_A(seed, device, edge_index)
        all_A.append(A)
        all_hits.append(hit)
        print(f"    Hit Rate: {hit:.1%}")

    all_A = np.array(all_A)  # [N_BOOTSTRAP, N_ccy, N_macro]
    all_hits = np.array(all_hits)

    print(f"\n>>> Mean Hit Rate: {all_hits.mean():.1%} ± {all_hits.std():.1%}")

    # ======================================
    # Statistical Analysis
    # ======================================
    print("\n>>> Computing statistical significance...")

    # Mean and std of each coefficient
    A_mean = all_A.mean(axis=0)  # [N_ccy, N_macro]
    A_std = all_A.std(axis=0, ddof=1)  # [N_ccy, N_macro]

    # t-statistic: mean / (std / sqrt(n))
    # H0: A[i,j] = 0
    t_stats = A_mean / (A_std / np.sqrt(N_BOOTSTRAP) + 1e-10)

    # p-values (two-tailed)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=N_BOOTSTRAP-1))

    # Significant at different levels
    sig_001 = p_values < 0.01
    sig_005 = p_values < 0.05
    sig_010 = p_values < 0.10

    # ======================================
    # Results Summary
    # ======================================
    print("\n" + "=" * 70)
    print("SIGNIFICANT MACRO-CURRENCY RELATIONSHIPS")
    print("=" * 70)

    print("\n=== Significant at p < 0.01 ===")
    results_001 = []
    for i in range(len(CURRENCIES)):
        for j in range(len(MACRO_FEATURES)):
            if sig_001[i, j]:
                ccy = CURRENCIES[i]
                macro = MACRO_FEATURES[j]
                coef = A_mean[i, j]
                std = A_std[i, j]
                p = p_values[i, j]
                direction = "+" if coef > 0 else ""
                print(f"  {ccy} ← {macro}: {direction}{coef:.3f} ± {std:.3f} (p={p:.4f})")
                results_001.append({
                    'currency': ccy, 'macro': macro,
                    'coef': float(coef), 'std': float(std),
                    'p_value': float(p), 'significant': '***'
                })

    print("\n=== Significant at p < 0.05 (excluding p < 0.01) ===")
    results_005 = []
    for i in range(len(CURRENCIES)):
        for j in range(len(MACRO_FEATURES)):
            if sig_005[i, j] and not sig_001[i, j]:
                ccy = CURRENCIES[i]
                macro = MACRO_FEATURES[j]
                coef = A_mean[i, j]
                std = A_std[i, j]
                p = p_values[i, j]
                direction = "+" if coef > 0 else ""
                print(f"  {ccy} ← {macro}: {direction}{coef:.3f} ± {std:.3f} (p={p:.4f})")
                results_005.append({
                    'currency': ccy, 'macro': macro,
                    'coef': float(coef), 'std': float(std),
                    'p_value': float(p), 'significant': '**'
                })

    print("\n=== Significant at p < 0.10 (excluding p < 0.05) ===")
    results_010 = []
    for i in range(len(CURRENCIES)):
        for j in range(len(MACRO_FEATURES)):
            if sig_010[i, j] and not sig_005[i, j]:
                ccy = CURRENCIES[i]
                macro = MACRO_FEATURES[j]
                coef = A_mean[i, j]
                std = A_std[i, j]
                p = p_values[i, j]
                direction = "+" if coef > 0 else ""
                print(f"  {ccy} ← {macro}: {direction}{coef:.3f} ± {std:.3f} (p={p:.4f})")
                results_010.append({
                    'currency': ccy, 'macro': macro,
                    'coef': float(coef), 'std': float(std),
                    'p_value': float(p), 'significant': '*'
                })

    # Count
    n_sig_001 = sig_001.sum()
    n_sig_005 = sig_005.sum()
    n_sig_010 = sig_010.sum()
    n_total = len(CURRENCIES) * len(MACRO_FEATURES)

    print(f"\n=== Summary ===")
    print(f"Total relationships: {n_total}")
    print(f"Significant at p < 0.01: {n_sig_001} ({n_sig_001/n_total*100:.1f}%)")
    print(f"Significant at p < 0.05: {n_sig_005} ({n_sig_005/n_total*100:.1f}%)")
    print(f"Significant at p < 0.10: {n_sig_010} ({n_sig_010/n_total*100:.1f}%)")

    # ======================================
    # Visualization
    # ======================================

    # 1. A matrix with significance markers
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Mean A matrix
    vmax = np.abs(A_mean).max()
    im1 = axes[0].imshow(A_mean, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[0].set_title('Mean A Matrix (Bootstrap)', fontsize=14)
    axes[0].set_xlabel('Macro Factors')
    axes[0].set_ylabel('Currencies')
    axes[0].set_xticks(range(len(MACRO_FEATURES)))
    axes[0].set_xticklabels(MACRO_FEATURES, rotation=45, ha='right')
    axes[0].set_yticks(range(len(CURRENCIES)))
    axes[0].set_yticklabels(CURRENCIES)
    plt.colorbar(im1, ax=axes[0])

    # Add significance markers
    for i in range(len(CURRENCIES)):
        for j in range(len(MACRO_FEATURES)):
            if sig_001[i, j]:
                axes[0].text(j, i, '***', ha='center', va='center', fontsize=8, fontweight='bold')
            elif sig_005[i, j]:
                axes[0].text(j, i, '**', ha='center', va='center', fontsize=8)
            elif sig_010[i, j]:
                axes[0].text(j, i, '*', ha='center', va='center', fontsize=8)

    # p-value heatmap
    im2 = axes[1].imshow(-np.log10(p_values + 1e-10), cmap='Reds', aspect='auto', vmin=0, vmax=3)
    axes[1].set_title('-log10(p-value)', fontsize=14)
    axes[1].set_xlabel('Macro Factors')
    axes[1].set_ylabel('Currencies')
    axes[1].set_xticks(range(len(MACRO_FEATURES)))
    axes[1].set_xticklabels(MACRO_FEATURES, rotation=45, ha='right')
    axes[1].set_yticks(range(len(CURRENCIES)))
    axes[1].set_yticklabels(CURRENCIES)
    cbar = plt.colorbar(im2, ax=axes[1])
    cbar.set_label('-log10(p)')

    # Add threshold lines in colorbar
    axes[1].axhline(y=-0.5, color='white', linewidth=0)  # dummy

    plt.tight_layout()
    plt.savefig(f'{output_dir}/significance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Coefficient stability plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Flatten and sort by absolute mean
    flat_data = []
    for i in range(len(CURRENCIES)):
        for j in range(len(MACRO_FEATURES)):
            flat_data.append({
                'label': f"{CURRENCIES[i]}←{MACRO_FEATURES[j]}",
                'mean': A_mean[i, j],
                'std': A_std[i, j],
                'p': p_values[i, j],
                'sig': sig_005[i, j]
            })

    # Sort by significance and then by absolute mean
    flat_data.sort(key=lambda x: (not x['sig'], -abs(x['mean'])))

    # Plot top 20
    top_n = 20
    labels = [d['label'] for d in flat_data[:top_n]]
    means = [d['mean'] for d in flat_data[:top_n]]
    stds = [d['std'] for d in flat_data[:top_n]]
    sigs = [d['sig'] for d in flat_data[:top_n]]

    colors = ['green' if s else 'gray' for s in sigs]
    x = range(top_n)

    ax.bar(x, means, yerr=stds, color=colors, alpha=0.7, capsize=3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Coefficient (mean ± std)')
    ax.set_title(f'Top {top_n} A Matrix Coefficients (Green = p < 0.05)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/coefficient_stability.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ======================================
    # Save Results
    # ======================================
    all_results = results_001 + results_005 + results_010

    # Sort by p-value
    all_results.sort(key=lambda x: x['p_value'])

    results = {
        'n_bootstrap': N_BOOTSTRAP,
        'mean_hit_rate': float(all_hits.mean()),
        'std_hit_rate': float(all_hits.std()),
        'n_significant_001': int(n_sig_001),
        'n_significant_005': int(n_sig_005),
        'n_significant_010': int(n_sig_010),
        'n_total': n_total,
        'significant_relationships': all_results,
        'A_mean': A_mean.tolist(),
        'A_std': A_std.tolist(),
        'p_values': p_values.tolist()
    }

    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"  - significance_heatmap.png")
    print(f"  - coefficient_stability.png")
    print(f"  - results.json")

    return results


if __name__ == '__main__':
    main()
