"""
Experiment: Confidence-Weighted Sparse Graph Model

Tests different approaches to improve magnitude prediction:
1. Baseline FXStrengthGNN
2. FXStrengthConfidenceGraph - learned confidence for edge weighting
3. FXStrengthMagnitudeAware - |prediction| as confidence
4. FXStrengthIterativeRefinement - multi-pass refinement
"""
import torch
import torch.nn.functional as F
import numpy as np
from config import Config
from models import (
    FXStrengthGNN,
    FXStrengthConfidenceGraph,
    FXStrengthMagnitudeAware,
    FXStrengthIterativeRefinement
)
from dataset import create_dataloaders, fully_connected_edge_index
from train import loss_fn, confidence_loss_fn


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def train_baseline(config, train_loader, test_loader, edge_index, device, epochs=30):
    """Train baseline FXStrengthGNN"""
    model = FXStrengthGNN(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xl, xm, yb in train_loader:
            xl, xm, yb = xl.to(device), xm.to(device), yb.to(device)
            rhat, ds, z_ccy, m_msg = model(xl, xm, edge_index)
            loss = loss_fn(rhat, yb, ds, model.A, config)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [Baseline] Epoch {epoch:02d}: loss = {np.mean(losses):.4f}")

    return model


def train_magnitude_aware(config, train_loader, test_loader, edge_index, device,
                           epochs=30, lambda_aux=0.3, lambda_mag=0.2):
    """Train Magnitude-Aware model"""
    model = FXStrengthMagnitudeAware(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for xl, xm, yb in train_loader:
            xl, xm, yb = xl.to(device), xm.to(device), yb.to(device)
            rhat, ds, z_ccy, m_msg, pre_pred, pre_conf = model(xl, xm, edge_index)

            mask = torch.ones(config.n_ccy, dtype=torch.bool, device=yb.device)
            mask[config.usd_idx] = False

            # Main MSE loss
            mse = ((rhat[:, mask] - yb[:, mask]) ** 2).mean()

            # Auxiliary loss: pre-GNN prediction should also be good
            aux_loss = ((pre_pred[:, mask] - yb[:, mask]) ** 2).mean()

            # Magnitude matching loss
            pred_abs = torch.abs(rhat[:, mask])
            target_abs = torch.abs(yb[:, mask])
            mag_loss = F.l1_loss(pred_abs, target_abs)

            # Variance term
            var_term = -ds.var(dim=1).mean()

            # L1 sparsity on A
            l1_A = model.A.abs().mean()

            loss = (mse +
                    lambda_aux * aux_loss +
                    lambda_mag * mag_loss +
                    config.lambda_var * var_term +
                    config.lambda_a_l1 * l1_A)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [MagAware] Epoch {epoch:02d}: loss = {np.mean(losses):.4f}")

    return model


def train_iterative(config, train_loader, test_loader, edge_index, device,
                    epochs=30, n_iterations=2, lambda_intermediate=0.2):
    """Train Iterative Refinement model"""
    model = FXStrengthIterativeRefinement(config, n_iterations=n_iterations).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for xl, xm, yb in train_loader:
            xl, xm, yb = xl.to(device), xm.to(device), yb.to(device)
            rhat, ds, z_ccy, m_msg, predictions = model(xl, xm, edge_index)

            mask = torch.ones(config.n_ccy, dtype=torch.bool, device=yb.device)
            mask[config.usd_idx] = False

            # Main MSE loss on final prediction
            mse = ((rhat[:, mask] - yb[:, mask]) ** 2).mean()

            # Intermediate losses (encourage early iterations to also be good)
            inter_loss = 0
            for pred in predictions[:-1]:  # All but the last (which is similar to rhat)
                pred_rel = pred - pred[:, config.usd_idx:config.usd_idx + 1]
                inter_loss = inter_loss + ((pred_rel[:, mask] - yb[:, mask]) ** 2).mean()
            inter_loss = inter_loss / max(len(predictions) - 1, 1)

            # Variance term
            var_term = -ds.var(dim=1).mean()

            # L1 sparsity on A
            l1_A = model.A.abs().mean()

            loss = (mse +
                    lambda_intermediate * inter_loss +
                    config.lambda_var * var_term +
                    config.lambda_a_l1 * l1_A)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [Iterative] Epoch {epoch:02d}: loss = {np.mean(losses):.4f}")

    return model


def train_confidence_graph(config, train_loader, test_loader, edge_index, device,
                           epochs=30, conf_threshold=0.3,
                           lambda_dir=0.5, lambda_mag=0.3, lambda_conf=0.1):
    """Train Confidence-Weighted Sparse Graph model"""
    model = FXStrengthConfidenceGraph(config, conf_threshold=conf_threshold).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for xl, xm, yb in train_loader:
            xl, xm, yb = xl.to(device), xm.to(device), yb.to(device)
            rhat, ds, z_ccy, m_msg, confidence, direction, magnitude = model(xl, xm, edge_index)

            loss, _ = confidence_loss_fn(
                rhat, yb, ds, model.A, confidence, direction, magnitude, config,
                lambda_dir=lambda_dir, lambda_mag=lambda_mag, lambda_conf=lambda_conf
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [ConfGraph] Epoch {epoch:02d}: loss = {np.mean(losses):.4f}")

    return model


def evaluate_model(model, test_loader, edge_index, config, device, model_type='baseline'):
    """Evaluate model and return comprehensive metrics"""
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for xl, xm, yb in test_loader:
            xl, xm = xl.to(device), xm.to(device)

            if model_type == 'confidence':
                rhat, ds, z_ccy, m_msg, confidence, direction, magnitude = model(xl, xm, edge_index)
            elif model_type == 'magnitude_aware':
                rhat, ds, z_ccy, m_msg, pre_pred, pre_conf = model(xl, xm, edge_index)
            elif model_type == 'iterative':
                rhat, ds, z_ccy, m_msg, predictions = model(xl, xm, edge_index)
            else:
                rhat, ds, z_ccy, m_msg = model(xl, xm, edge_index)

            all_preds.append(rhat.cpu().numpy())
            all_targets.append(yb.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Mask USD
    mask = np.array([i != config.usd_idx for i in range(config.n_ccy)])
    preds_masked = preds[:, mask]
    targets_masked = targets[:, mask]

    # Metrics
    results = {}

    # 1. Direction accuracy
    correct = (np.sign(preds_masked) == np.sign(targets_masked))
    results['hit_rate'] = correct.mean()

    # 2. Magnitude-weighted hit rate
    weights = np.abs(targets_masked)
    results['weighted_hit_rate'] = (correct * weights).sum() / weights.sum()

    # 3. MSE / MAE
    results['mse'] = ((preds_masked - targets_masked) ** 2).mean()
    results['mae'] = np.abs(preds_masked - targets_masked).mean()

    # 4. Magnitude ratio (|pred| / |target|)
    pred_abs = np.abs(preds_masked)
    target_abs = np.abs(targets_masked)
    results['magnitude_ratio'] = pred_abs.mean() / target_abs.mean()

    # 5. Prediction std vs target std
    results['pred_std'] = preds_masked.std()
    results['target_std'] = targets_masked.std()

    # 6. Per-quantile analysis
    results['quantile_analysis'] = {}
    flat_preds = preds_masked.flatten()
    flat_targets = targets_masked.flatten()
    abs_targets = np.abs(flat_targets)

    for q, (low, high) in enumerate([(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]):
        low_pct = np.percentile(abs_targets, low)
        high_pct = np.percentile(abs_targets, high)
        mask_q = (abs_targets >= low_pct) & (abs_targets < high_pct if high < 100 else True)

        if mask_q.sum() > 0:
            q_preds = flat_preds[mask_q]
            q_targets = flat_targets[mask_q]
            q_correct = (np.sign(q_preds) == np.sign(q_targets))

            results['quantile_analysis'][f'Q{q+1}'] = {
                'hit_rate': q_correct.mean(),
                'magnitude_ratio': np.abs(q_preds).mean() / np.abs(q_targets).mean(),
                'pred_std': np.abs(q_preds).std(),
                'target_std': np.abs(q_targets).std()
            }

    return results


def print_results(name, results):
    """Print results in a formatted way"""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"  Hit Rate:          {results['hit_rate']:.1%}")
    print(f"  Weighted Hit Rate: {results['weighted_hit_rate']:.1%}")
    print(f"  MSE:               {results['mse']:.6f}")
    print(f"  MAE:               {results['mae']:.6f}")
    print(f"  Magnitude Ratio:   {results['magnitude_ratio']:.1%}")
    print(f"  Pred Std / Target Std: {results['pred_std']:.4f} / {results['target_std']:.4f} = {results['pred_std']/results['target_std']:.1%}")

    print(f"\n  Per-Quantile Analysis:")
    for q, data in results['quantile_analysis'].items():
        print(f"    {q}: Hit={data['hit_rate']:.1%}, MagRatio={data['magnitude_ratio']:.1%}")


def main():
    print("="*60)
    print("Experiment: Confidence-Weighted & Magnitude-Aware Models")
    print("="*60)

    # Setup
    seed = 42
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    config = Config()
    config.seed = seed

    # Load data
    train_loader, test_loader = create_dataloaders(config)
    edge_index = fully_connected_edge_index(config.n_ccy).to(device)

    print(f"\nCurrencies ({config.n_ccy}): {config.ccys}")
    print(f"Macros ({config.macro_dim}): {config.global_features}")

    all_results = {}

    # 1. Train baseline
    print("\n" + "-"*60)
    print("1. Training Baseline Model")
    print("-"*60)
    set_seed(seed)
    baseline_model = train_baseline(config, train_loader, test_loader, edge_index, device)
    all_results['Baseline'] = evaluate_model(baseline_model, test_loader, edge_index, config, device, 'baseline')

    # 2. Train Magnitude-Aware model
    print("\n" + "-"*60)
    print("2. Training Magnitude-Aware Model")
    print("-"*60)
    set_seed(seed)
    mag_model = train_magnitude_aware(config, train_loader, test_loader, edge_index, device,
                                       lambda_aux=0.3, lambda_mag=0.3)
    all_results['MagnitudeAware'] = evaluate_model(mag_model, test_loader, edge_index, config, device, 'magnitude_aware')

    # 3. Train Iterative Refinement model
    print("\n" + "-"*60)
    print("3. Training Iterative Refinement Model (2 iterations)")
    print("-"*60)
    set_seed(seed)
    iter_model = train_iterative(config, train_loader, test_loader, edge_index, device,
                                  n_iterations=2, lambda_intermediate=0.2)
    all_results['Iterative (K=2)'] = evaluate_model(iter_model, test_loader, edge_index, config, device, 'iterative')

    # 4. Train Iterative Refinement model with more iterations
    print("\n" + "-"*60)
    print("4. Training Iterative Refinement Model (3 iterations)")
    print("-"*60)
    set_seed(seed)
    iter_model3 = train_iterative(config, train_loader, test_loader, edge_index, device,
                                   n_iterations=3, lambda_intermediate=0.2)
    all_results['Iterative (K=3)'] = evaluate_model(iter_model3, test_loader, edge_index, config, device, 'iterative')

    # 5. Train Confidence Graph
    print("\n" + "-"*60)
    print("5. Training Confidence Graph Model")
    print("-"*60)
    set_seed(seed)
    conf_model = train_confidence_graph(config, train_loader, test_loader, edge_index, device,
                                         conf_threshold=0.3, lambda_dir=0.5, lambda_mag=0.5, lambda_conf=0.1)
    all_results['ConfidenceGraph'] = evaluate_model(conf_model, test_loader, edge_index, config, device, 'confidence')

    # Print all results
    for name, results in all_results.items():
        print_results(name, results)

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Model':<25} | {'Hit%':>6} | {'WHit%':>6} | {'MagR':>6} | {'Std Ratio':>9}")
    print("-"*70)
    for name, r in all_results.items():
        std_ratio = r['pred_std'] / r['target_std']
        print(f"{name:<25} | {r['hit_rate']:>5.1%} | {r['weighted_hit_rate']:>5.1%} | {r['magnitude_ratio']:>5.1%} | {std_ratio:>8.1%}")


if __name__ == '__main__':
    main()
