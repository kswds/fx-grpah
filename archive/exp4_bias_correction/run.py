"""
Exp4: Bias Correction
- Fix positive bias (60% vs 50%)
- Fix conservative predictions (std 0.34 vs 0.84)
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
import torch
import torch.nn as nn

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Custom loss functions

def loss_fn_direction(rhat, y, ds, A_param, config, lambda_dir=0.1):
    """Original loss + direction penalty"""
    mask = torch.ones(config.n_ccy, dtype=torch.bool, device=y.device)
    mask[config.usd_idx] = False

    # MSE loss
    mse = ((rhat[:, mask] - y[:, mask]) ** 2).mean()

    # Variance term
    var_term = -ds.var(dim=1).mean()

    # L1 on A
    l1_A = A_param.abs().mean()

    # Direction penalty: penalize when sign(pred) != sign(target)
    pred_sign = torch.sign(rhat[:, mask])
    target_sign = torch.sign(y[:, mask])
    wrong_direction = (pred_sign != target_sign).float()
    direction_penalty = wrong_direction.mean()

    return mse + config.lambda_var * var_term + config.lambda_a_l1 * l1_A + lambda_dir * direction_penalty


def loss_fn_scaled(rhat, y, ds, A_param, config):
    """MSE on scaled predictions to match target std"""
    mask = torch.ones(config.n_ccy, dtype=torch.bool, device=y.device)
    mask[config.usd_idx] = False

    # Scale predictions to match target std
    pred_std = rhat[:, mask].std() + 1e-8
    target_std = y[:, mask].std() + 1e-8
    scale = target_std / pred_std

    rhat_scaled = rhat * scale.detach()  # Detach to avoid gradient through scale

    # MSE loss on scaled
    mse = ((rhat_scaled[:, mask] - y[:, mask]) ** 2).mean()

    # Other terms
    var_term = -ds.var(dim=1).mean()
    l1_A = A_param.abs().mean()

    return mse + config.lambda_var * var_term + config.lambda_a_l1 * l1_A


def loss_fn_asymmetric(rhat, y, ds, A_param, config, neg_weight=1.5):
    """Asymmetric loss: weight negative predictions more to fix positive bias"""
    mask = torch.ones(config.n_ccy, dtype=torch.bool, device=y.device)
    mask[config.usd_idx] = False

    errors = (rhat[:, mask] - y[:, mask]) ** 2

    # Weight: higher for samples where target is negative (to encourage more negative predictions)
    weights = torch.ones_like(errors)
    weights[y[:, mask] < 0] = neg_weight

    mse = (errors * weights).mean()

    var_term = -ds.var(dim=1).mean()
    l1_A = A_param.abs().mean()

    return mse + config.lambda_var * var_term + config.lambda_a_l1 * l1_A


def loss_fn_huber_direction(rhat, y, ds, A_param, config, delta=1.0, lambda_dir=0.2):
    """Huber loss + strong direction penalty"""
    mask = torch.ones(config.n_ccy, dtype=torch.bool, device=y.device)
    mask[config.usd_idx] = False

    # Huber loss (less sensitive to outliers)
    diff = rhat[:, mask] - y[:, mask]
    abs_diff = torch.abs(diff)
    huber = torch.where(abs_diff <= delta,
                        0.5 * diff ** 2,
                        delta * (abs_diff - 0.5 * delta))
    huber_loss = huber.mean()

    # Strong direction penalty
    pred_sign = torch.sign(rhat[:, mask])
    target_sign = torch.sign(y[:, mask])
    wrong_direction = (pred_sign != target_sign).float()
    direction_penalty = wrong_direction.mean()

    var_term = -ds.var(dim=1).mean()
    l1_A = A_param.abs().mean()

    return huber_loss + config.lambda_var * var_term + config.lambda_a_l1 * l1_A + lambda_dir * direction_penalty


class TrainerCustomLoss(Trainer):
    """Trainer with custom loss function"""
    def __init__(self, model, config, device, loss_fn):
        super().__init__(model, config, device)
        self.custom_loss_fn = loss_fn

    def train_epoch(self, train_loader, edge_index):
        self.model.train()
        losses = []

        for xl, xm, yb in train_loader:
            xl, xm, yb = xl.to(self.device), xm.to(self.device), yb.to(self.device)
            rhat, ds, z_ccy, m_msg = self.model(xl, xm, edge_index)
            loss = self.custom_loss_fn(rhat, yb, ds, self.model.A, self.config)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            losses.append(loss.item())

        return np.mean(losses)


def evaluate_detailed(model, test_loader, edge_index, device, config):
    """Detailed evaluation"""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for xl, xm, yb in test_loader:
            xl, xm, yb = xl.to(device), xm.to(device), yb.to(device)
            rhat, _, _, _ = model(xl, xm, edge_index)
            all_preds.append(rhat.cpu())
            all_targets.append(yb.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    mask = np.ones(config.n_ccy, dtype=bool)
    mask[config.usd_idx] = False

    preds_flat = preds[:, mask].flatten()
    targets_flat = targets[:, mask].flatten()

    rmse = np.sqrt(np.mean((preds_flat - targets_flat) ** 2))
    mae = np.mean(np.abs(preds_flat - targets_flat))
    hit = np.mean(np.sign(preds_flat) == np.sign(targets_flat))

    pred_positive_ratio = (preds_flat > 0).mean()
    pred_std = preds_flat.std()
    target_std = targets_flat.std()

    return {
        'rmse': rmse,
        'mae': mae,
        'hit': hit,
        'pred_positive_ratio': pred_positive_ratio,
        'pred_std': pred_std,
        'target_std': target_std,
    }


def run_experiment(name, loss_fn, config, device, edge_index, seed=42):
    """Run single experiment with custom loss"""
    set_seed(seed)
    train_loader, test_loader = create_dataloaders(config, macro_mode="real")
    model = FXStrengthGNN(config)
    trainer = TrainerCustomLoss(model, config, device, loss_fn)

    # Training
    for epoch in range(1, config.epochs + 1):
        loss = trainer.train_epoch(train_loader, edge_index)
        if epoch in (1, config.epochs) or epoch % 10 == 0:
            print(f"[{name}] epoch {epoch:02d}/{config.epochs} | loss = {loss:.4f}")

    # Evaluation
    metrics = evaluate_detailed(model, test_loader, edge_index, device, config)
    return metrics


def main():
    seed = 42
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = Config(seed=seed, epochs=30, batch_size=128, lr=3e-4)
    config.use_skip_connection = True
    config.use_layer_norm = True

    edge_index = fully_connected_edge_index(config.n_ccy).to(device)

    # Original loss for comparison
    from train import loss_fn as original_loss_fn

    experiments = [
        ("baseline", lambda r, y, ds, A, c: original_loss_fn(r, y, ds, A, c)),
        ("direction_0.1", lambda r, y, ds, A, c: loss_fn_direction(r, y, ds, A, c, lambda_dir=0.1)),
        ("direction_0.3", lambda r, y, ds, A, c: loss_fn_direction(r, y, ds, A, c, lambda_dir=0.3)),
        ("asymmetric_1.5", lambda r, y, ds, A, c: loss_fn_asymmetric(r, y, ds, A, c, neg_weight=1.5)),
        ("asymmetric_2.0", lambda r, y, ds, A, c: loss_fn_asymmetric(r, y, ds, A, c, neg_weight=2.0)),
        ("huber_dir", lambda r, y, ds, A, c: loss_fn_huber_direction(r, y, ds, A, c)),
    ]

    results = {}

    print("\n" + "=" * 70)
    print("EXP4: BIAS CORRECTION")
    print("=" * 70)

    for name, loss_fn in experiments:
        print(f"\n>>> Running {name}...")
        metrics = run_experiment(name, loss_fn, config, device, edge_index, seed)
        results[name] = metrics
        print(f"  RMSE={metrics['rmse']:.4f}, Hit={metrics['hit']:.4f}, "
              f"Pred+={metrics['pred_positive_ratio']*100:.1f}%, "
              f"PredStd={metrics['pred_std']:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("EXP4 RESULTS")
    print("=" * 70)
    print(f"{'Method':<18} {'RMSE':>8} {'MAE':>8} {'Hit':>8} {'Pred+%':>8} {'PredStd':>8} {'(target)':>8}")
    print("-" * 70)

    for name, m in results.items():
        print(f"{name:<18} {m['rmse']:>8.4f} {m['mae']:>8.4f} {m['hit']:>8.4f} "
              f"{m['pred_positive_ratio']*100:>7.1f}% {m['pred_std']:>8.3f} ({m['target_std']:.3f})")

    # Best method
    best_hit = max(results.items(), key=lambda x: x[1]['hit'])
    best_rmse = min(results.items(), key=lambda x: x[1]['rmse'])

    print(f"\nBest Hit Rate: {best_hit[0]} ({best_hit[1]['hit']:.4f})")
    print(f"Best RMSE: {best_rmse[0]} ({best_rmse[1]['rmse']:.4f})")

    # Check bias improvement
    baseline_bias = abs(results['baseline']['pred_positive_ratio'] - 0.5)
    for name, m in results.items():
        if name == 'baseline':
            continue
        new_bias = abs(m['pred_positive_ratio'] - 0.5)
        if new_bias < baseline_bias:
            print(f"✓ {name} reduced positive bias: {baseline_bias*100:.1f}% → {new_bias*100:.1f}%")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
    }
    with open('exp4_bias_correction/results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to exp4_bias_correction/")


if __name__ == "__main__":
    main()
