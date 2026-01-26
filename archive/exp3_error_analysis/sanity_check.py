"""
Sanity Check: Is the model learning something meaningful?
- Are predictions varied or constant?
- Sign distribution
- Triangle consistency of errors
"""
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

import random
import numpy as np
import torch
import matplotlib.pyplot as plt

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


def get_predictions(model, test_loader, edge_index, device):
    model.eval()
    all_preds, all_targets, all_ds = [], [], []

    with torch.no_grad():
        for xl, xm, yb in test_loader:
            xl, xm, yb = xl.to(device), xm.to(device), yb.to(device)
            rhat, ds, _, _ = model(xl, xm, edge_index)

            all_preds.append(rhat.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
            all_ds.append(ds.cpu().numpy())

    return (np.concatenate(all_preds, axis=0),
            np.concatenate(all_targets, axis=0),
            np.concatenate(all_ds, axis=0))


def main():
    seed = 42
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(seed=seed, epochs=30, batch_size=128, lr=3e-4)
    config.use_skip_connection = True
    config.use_layer_norm = True

    edge_index = fully_connected_edge_index(config.n_ccy).to(device)

    # Train
    print(">>> Training...")
    set_seed(seed)
    train_loader, test_loader = create_dataloaders(config, macro_mode="real")
    model = FXStrengthGNN(config)
    trainer = Trainer(model, config, device)
    trainer.train(train_loader, test_loader, edge_index, label="FULL")

    # Get predictions
    preds, targets, ds = get_predictions(model, test_loader, edge_index, device)
    ccys = config.ccys

    print("\n" + "=" * 70)
    print("SANITY CHECK")
    print("=" * 70)

    # 1. Prediction Distribution
    print("\n--- 1. Prediction Distribution ---")
    mask = np.ones(config.n_ccy, dtype=bool)
    mask[config.usd_idx] = False

    preds_flat = preds[:, mask].flatten()
    targets_flat = targets[:, mask].flatten()

    print(f"Predictions: mean={preds_flat.mean():.4f}, std={preds_flat.std():.4f}, "
          f"min={preds_flat.min():.4f}, max={preds_flat.max():.4f}")
    print(f"Targets:     mean={targets_flat.mean():.4f}, std={targets_flat.std():.4f}, "
          f"min={targets_flat.min():.4f}, max={targets_flat.max():.4f}")

    # Check if predictions are near-constant
    if preds_flat.std() < 0.01:
        print("⚠️ WARNING: Predictions are nearly constant!")
    else:
        print("✓ Predictions have reasonable variance")

    # 2. Sign Distribution
    print("\n--- 2. Sign Distribution ---")
    pred_positive = (preds_flat > 0).mean()
    pred_negative = (preds_flat < 0).mean()
    target_positive = (targets_flat > 0).mean()
    target_negative = (targets_flat < 0).mean()

    print(f"Predictions: {pred_positive*100:.1f}% positive, {pred_negative*100:.1f}% negative")
    print(f"Targets:     {target_positive*100:.1f}% positive, {target_negative*100:.1f}% negative")

    if abs(pred_positive - 0.5) > 0.3:
        print("⚠️ WARNING: Predictions are heavily biased to one sign!")
    else:
        print("✓ Sign distribution is reasonable")

    # 3. Per-currency sign distribution
    print("\n--- 3. Per-Currency Sign Distribution ---")
    print(f"{'Currency':<6} {'Pred+':>8} {'Pred-':>8} {'Target+':>8} {'Target-':>8}")
    print("-" * 40)

    for i, ccy in enumerate(ccys):
        if i == config.usd_idx:
            continue
        pred_i = preds[:, i]
        target_i = targets[:, i]

        print(f"{ccy:<6} {(pred_i > 0).mean()*100:>7.1f}% {(pred_i < 0).mean()*100:>7.1f}% "
              f"{(target_i > 0).mean()*100:>7.1f}% {(target_i < 0).mean()*100:>7.1f}%")

    # 4. Confusion Matrix for Direction
    print("\n--- 4. Direction Confusion Matrix ---")
    pred_sign = np.sign(preds_flat)
    target_sign = np.sign(targets_flat)

    tp = ((pred_sign > 0) & (target_sign > 0)).sum()
    tn = ((pred_sign < 0) & (target_sign < 0)).sum()
    fp = ((pred_sign > 0) & (target_sign < 0)).sum()
    fn = ((pred_sign < 0) & (target_sign > 0)).sum()

    total = tp + tn + fp + fn
    print(f"True Positive (both +):  {tp:>6} ({tp/total*100:.1f}%)")
    print(f"True Negative (both -):  {tn:>6} ({tn/total*100:.1f}%)")
    print(f"False Positive (pred+, target-): {fp:>6} ({fp/total*100:.1f}%)")
    print(f"False Negative (pred-, target+): {fn:>6} ({fn/total*100:.1f}%)")
    print(f"Accuracy: {(tp+tn)/total*100:.1f}%")

    # 5. Triangle Consistency Analysis
    print("\n--- 5. Triangle Consistency ---")
    # For currency strength ds, triangle should sum to 0: ds_i - ds_j + ds_j - ds_k + ds_k - ds_i = 0
    # But let's check if errors are correlated across related pairs

    # Check if when we're wrong about EUR/USD, are we also wrong about GBP/USD?
    print("\nCorrelation of errors across currencies:")

    errors = preds - targets  # [T, N]
    error_corr = np.corrcoef(errors[:, mask].T)

    ccy_names = [c for i, c in enumerate(ccys) if i != config.usd_idx]
    print(f"\n{'':>6}", end="")
    for c in ccy_names[:5]:  # Show first 5
        print(f"{c:>8}", end="")
    print()

    for i, c1 in enumerate(ccy_names[:5]):
        print(f"{c1:<6}", end="")
        for j in range(5):
            print(f"{error_corr[i,j]:>8.2f}", end="")
        print()

    # 6. Persistent Errors
    print("\n--- 6. Persistent Error Analysis ---")
    # Are there samples that are ALWAYS predicted wrong?
    wrong_mask = (np.sign(preds) != np.sign(targets))  # [T, N]

    # Per-sample: how many currencies wrong?
    wrong_per_sample = wrong_mask[:, mask].sum(axis=1)

    print(f"Samples with 0 wrong currencies: {(wrong_per_sample == 0).sum()}")
    print(f"Samples with 1-3 wrong currencies: {((wrong_per_sample >= 1) & (wrong_per_sample <= 3)).sum()}")
    print(f"Samples with 4-6 wrong currencies: {((wrong_per_sample >= 4) & (wrong_per_sample <= 6)).sum()}")
    print(f"Samples with 7+ wrong currencies: {(wrong_per_sample >= 7).sum()}")

    # Per-currency: how often wrong?
    print("\nPer-currency wrong rate:")
    for i, ccy in enumerate(ccys):
        if i == config.usd_idx:
            continue
        wrong_rate = wrong_mask[:, i].mean()
        print(f"  {ccy}: {wrong_rate*100:.1f}% wrong")

    # 7. Histogram of predictions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(preds_flat, bins=50, alpha=0.7, label='Predictions', density=True)
    axes[0].hist(targets_flat, bins=50, alpha=0.7, label='Targets', density=True)
    axes[0].set_title('Distribution of Predictions vs Targets')
    axes[0].legend()
    axes[0].set_xlabel('Value')

    # Scatter plot
    sample_idx = np.random.choice(len(preds_flat), min(1000, len(preds_flat)), replace=False)
    axes[1].scatter(targets_flat[sample_idx], preds_flat[sample_idx], alpha=0.3, s=5)
    axes[1].plot([-3, 3], [-3, 3], 'r--', label='Perfect prediction')
    axes[1].set_xlabel('Target')
    axes[1].set_ylabel('Prediction')
    axes[1].set_title('Prediction vs Target (sampled)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('exp3_error_analysis/sanity_check.png', dpi=150)
    print("\nPlot saved to exp3_error_analysis/sanity_check.png")


if __name__ == "__main__":
    main()
