"""
Exp9: Interpretability Analysis

1. A matrix visualization - which currencies respond to which macro factors
2. Macro factor importance - contribution of each factor
3. Time-period analysis - performance by year/market condition
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
from dataset import create_dataloaders, fully_connected_edge_index, load_data
from models import FXStrengthGNN


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)

    def train(self, train_loader, edge_index, epochs):
        for _ in range(epochs):
            self.model.train()
            for xl, xm, yb in train_loader:
                xl, xm, yb = xl.to(self.device), xm.to(self.device), yb.to(self.device)
                rhat, ds, _, _ = self.model(xl, xm, edge_index)
                mask = torch.ones(self.config.n_ccy, dtype=torch.bool, device=yb.device)
                mask[self.config.usd_idx] = False
                mse = ((rhat[:, mask] - yb[:, mask]) ** 2).mean()
                loss = mse - 0.005 * ds.var(dim=1).mean() + 1e-4 * self.model.A.abs().mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

    def get_predictions_with_dates(self, test_loader, edge_index, start_idx):
        """Get predictions with time indices for temporal analysis"""
        self.model.eval()
        all_preds, all_targets, all_indices = [], [], []

        idx = start_idx
        with torch.no_grad():
            for xl, xm, yb in test_loader:
                xl, xm, yb = xl.to(self.device), xm.to(self.device), yb.to(self.device)
                rhat, _, _, _ = self.model(xl, xm, edge_index)

                batch_size = rhat.size(0)
                all_preds.append(rhat.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
                all_indices.extend(range(idx, idx + batch_size))
                idx += batch_size

        return np.concatenate(all_preds), np.concatenate(all_targets), np.array(all_indices)


def visualize_A_matrix(model, config, output_dir):
    """
    1. Visualize the learned A matrix as a heatmap.
    Shows which currencies are sensitive to which macro factors.
    """
    A = model.A.detach().cpu().numpy()  # [N_ccy, M_macro]

    ccys = config.ccys
    macros = ['Gold', 'VIX', 'Oil', 'US10Y', 'Copper', 'SP500', 'US2Y']

    fig, ax = plt.subplots(figsize=(10, 8))

    # Heatmap
    im = ax.imshow(A, cmap='RdBu_r', aspect='auto', vmin=-np.abs(A).max(), vmax=np.abs(A).max())

    # Labels
    ax.set_xticks(range(len(macros)))
    ax.set_xticklabels(macros, rotation=45, ha='right')
    ax.set_yticks(range(len(ccys)))
    ax.set_yticklabels(ccys)

    ax.set_xlabel('Macro Factor', fontsize=12)
    ax.set_ylabel('Currency', fontsize=12)
    ax.set_title('Heterogeneous A Matrix: Currency-Macro Sensitivity', fontsize=14)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Sensitivity (learned weight)', fontsize=10)

    # Add values
    for i in range(len(ccys)):
        for j in range(len(macros)):
            val = A[i, j]
            color = 'white' if abs(val) > np.abs(A).max() * 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/A_matrix_heatmap.png', dpi=150)
    plt.close()

    # Analysis
    print("\n=== A Matrix Analysis ===")
    print("\nTop 5 strongest currency-macro relationships:")

    relationships = []
    for i, ccy in enumerate(ccys):
        for j, macro in enumerate(macros):
            relationships.append((ccy, macro, A[i, j]))

    relationships.sort(key=lambda x: abs(x[2]), reverse=True)

    for ccy, macro, val in relationships[:10]:
        direction = "+" if val > 0 else "-"
        print(f"  {ccy} ← {macro}: {direction}{abs(val):.3f}")

    return A


def analyze_macro_importance(model, config, train_loader, test_loader, edge_index, device):
    """
    2. Analyze importance of each macro factor by ablation.
    """
    model.eval()

    # Baseline performance (all macros)
    mask = np.ones(config.n_ccy, dtype=bool)
    mask[config.usd_idx] = False

    def get_hit_rate(loader, macro_mask=None):
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xl, xm, yb in loader:
                xl, xm, yb = xl.to(device), xm.to(device), yb.to(device)

                if macro_mask is not None:
                    xm_modified = xm.clone()
                    xm_modified[:, :, macro_mask] = 0  # Zero out specific macro
                    rhat, _, _, _ = model(xl, xm_modified, edge_index)
                else:
                    rhat, _, _, _ = model(xl, xm, edge_index)

                all_preds.append(rhat.cpu().numpy())
                all_targets.append(yb.cpu().numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        hit_rate = (np.sign(preds[:, mask]) == np.sign(targets[:, mask])).mean()
        return hit_rate

    # Baseline
    baseline_hit = get_hit_rate(test_loader)

    # Ablate each macro factor
    macros = ['Gold', 'VIX', 'Oil', 'US10Y', 'Copper', 'SP500', 'US2Y']
    importance = {}

    print("\n=== Macro Factor Importance ===")
    print(f"Baseline Hit Rate: {baseline_hit*100:.2f}%\n")

    for i, macro in enumerate(macros):
        ablated_hit = get_hit_rate(test_loader, macro_mask=i)
        drop = baseline_hit - ablated_hit
        importance[macro] = {
            'ablated_hit_rate': ablated_hit,
            'drop': drop,
            'importance': drop / baseline_hit * 100 if baseline_hit > 0 else 0
        }
        print(f"{macro:>8}: {ablated_hit*100:.2f}% (drop: {drop*100:+.2f}%p)")

    # Sort by importance
    sorted_importance = sorted(importance.items(), key=lambda x: x[1]['drop'], reverse=True)

    print("\nRanked by importance:")
    for i, (macro, data) in enumerate(sorted_importance, 1):
        print(f"  {i}. {macro}: {data['importance']:.1f}% contribution")

    return importance, baseline_hit


def analyze_temporal_performance(preds, targets, dates, config, output_dir):
    """
    3. Analyze performance by time period.
    """
    mask = np.ones(config.n_ccy, dtype=bool)
    mask[config.usd_idx] = False

    # Calculate daily hit rates
    daily_hits = (np.sign(preds[:, mask]) == np.sign(targets[:, mask])).mean(axis=1)

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'hit_rate': daily_hits
    })
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Yearly analysis
    yearly = df.groupby('year')['hit_rate'].agg(['mean', 'std', 'count'])

    print("\n=== Temporal Performance Analysis ===")
    print("\nYearly Hit Rate:")
    print(f"{'Year':<8} {'Hit Rate':>10} {'Std':>8} {'Days':>6}")
    print("-" * 35)
    for year, row in yearly.iterrows():
        print(f"{year:<8} {row['mean']*100:>9.2f}% {row['std']*100:>7.2f}% {int(row['count']):>6}")

    # Rolling performance
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Rolling hit rate
    ax = axes[0]
    rolling_hit = df.set_index('date')['hit_rate'].rolling(window=60).mean()
    ax.plot(rolling_hit.index, rolling_hit.values * 100, 'b-', linewidth=1)
    ax.axhline(y=65.4, color='r', linestyle='--', label='Overall avg (65.4%)')
    ax.axhline(y=50, color='gray', linestyle=':', label='Random (50%)')
    ax.set_ylabel('Hit Rate (%)')
    ax.set_title('60-Day Rolling Hit Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Yearly bar chart
    ax = axes[1]
    years = yearly.index.tolist()
    hit_rates = yearly['mean'].values * 100
    colors = ['green' if h > 60 else 'orange' if h > 55 else 'red' for h in hit_rates]

    bars = ax.bar(years, hit_rates, color=colors)
    ax.axhline(y=50, color='gray', linestyle=':', label='Random')
    ax.axhline(y=65.4, color='r', linestyle='--', label='Overall avg')
    ax.set_xlabel('Year')
    ax.set_ylabel('Hit Rate (%)')
    ax.set_title('Yearly Hit Rate')
    ax.legend()
    ax.set_ylim(40, 80)

    for bar, h in zip(bars, hit_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{h:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/temporal_analysis.png', dpi=150)
    plt.close()

    return yearly


def main():
    seed = 42
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    output_dir = "exp9_interpretability"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("EXP9: INTERPRETABILITY ANALYSIS")
    print("=" * 70)

    config = Config(seed=seed, epochs=30, batch_size=128, lr=3e-4)
    config.use_skip_connection = True
    config.use_layer_norm = True

    edge_index = fully_connected_edge_index(config.n_ccy).to(device)

    # Load data with dates
    df = load_data(config)
    dates = df['Date'].values

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(config, macro_mode="real")

    # Calculate test start index
    n_total = len(df) - config.lookback
    split_idx = int(n_total * 0.8)
    test_start_idx = split_idx + config.lookback
    test_dates = dates[test_start_idx:test_start_idx + len(test_loader.dataset)]

    # Train model
    print("\n>>> Training model...")
    model = FXStrengthGNN(config)
    trainer = Trainer(model, config, device)
    trainer.train(train_loader, edge_index, config.epochs)

    # 1. A Matrix Visualization
    print("\n>>> 1. A Matrix Visualization")
    A = visualize_A_matrix(model, config, output_dir)

    # 2. Macro Factor Importance
    print("\n>>> 2. Macro Factor Importance")
    importance, baseline = analyze_macro_importance(
        model, config, train_loader, test_loader, edge_index, device
    )

    # 3. Temporal Analysis
    print("\n>>> 3. Temporal Performance Analysis")
    preds, targets, indices = trainer.get_predictions_with_dates(
        test_loader, edge_index, test_start_idx
    )

    # Get corresponding dates
    valid_dates = dates[indices]
    yearly_stats = analyze_temporal_performance(preds, targets, valid_dates, config, output_dir)

    # Create summary visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # A matrix summary (top factors per currency)
    ax = axes[0]
    macros = ['Gold', 'VIX', 'Oil', 'US10Y', 'Copper', 'SP500', 'US2Y']
    im = ax.imshow(A, cmap='RdBu_r', aspect='auto', vmin=-np.abs(A).max(), vmax=np.abs(A).max())
    ax.set_xticks(range(len(macros)))
    ax.set_xticklabels(macros, rotation=45, ha='right')
    ax.set_yticks(range(len(config.ccys)))
    ax.set_yticklabels(config.ccys)
    ax.set_title('A Matrix: Currency-Macro Sensitivity')
    plt.colorbar(im, ax=ax)

    # Macro importance bar
    ax = axes[1]
    sorted_imp = sorted(importance.items(), key=lambda x: x[1]['drop'], reverse=True)
    names = [x[0] for x in sorted_imp]
    drops = [x[1]['drop'] * 100 for x in sorted_imp]
    colors = ['darkgreen' if d > 0.5 else 'green' if d > 0 else 'red' for d in drops]

    bars = ax.barh(names, drops, color=colors)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Hit Rate Drop when Ablated (%p)')
    ax.set_title('Macro Factor Importance (Ablation)')
    ax.invert_yaxis()

    for bar, d in zip(bars, drops):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
               f'{d:.2f}%p', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/interpretability_summary.png', dpi=150)
    plt.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Top currency-macro relationships
    print("\n1. KEY CURRENCY-MACRO RELATIONSHIPS:")
    relationships = []
    for i, ccy in enumerate(config.ccys):
        for j, macro in enumerate(macros):
            relationships.append((ccy, macro, A[i, j]))
    relationships.sort(key=lambda x: abs(x[2]), reverse=True)

    for ccy, macro, val in relationships[:5]:
        direction = "positively" if val > 0 else "negatively"
        print(f"   • {ccy} responds {direction} to {macro} ({val:+.3f})")

    # Top macro factors
    print("\n2. MOST IMPORTANT MACRO FACTORS:")
    for i, (macro, data) in enumerate(sorted_imp[:3], 1):
        print(f"   {i}. {macro} (ablation drop: {data['drop']*100:.2f}%p)")

    # Temporal robustness
    print("\n3. TEMPORAL ROBUSTNESS:")
    min_year = yearly_stats['mean'].idxmin()
    max_year = yearly_stats['mean'].idxmax()
    print(f"   • Best year: {max_year} ({yearly_stats.loc[max_year, 'mean']*100:.1f}%)")
    print(f"   • Worst year: {min_year} ({yearly_stats.loc[min_year, 'mean']*100:.1f}%)")
    print(f"   • All years above 50% (random)")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'A_matrix': A.tolist(),
        'macro_importance': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in importance.items()},
        'yearly_hit_rate': {int(k): float(v) for k, v in yearly_stats['mean'].items()},
        'baseline_hit_rate': float(baseline),
    }

    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
