"""
Model saving and loading with config tracking
- Saves: model weights, config (currencies, macros), seed, results
- Ensures reproducibility
"""
import os
import json
import torch
import numpy as np
from datetime import datetime

from config import Config
from models import FXStrengthGNN
from dataset import create_dataloaders, fully_connected_edge_index
from train import Trainer


def train_and_save(save_dir: str, seed: int = 42, name: str = None):
    """Train model and save with full config tracking"""

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    config = Config()
    config.seed = seed

    # Create save directory
    if name is None:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, name)
    os.makedirs(save_path, exist_ok=True)

    # Train
    print("=" * 60)
    print(f"Training model: {name}")
    print("=" * 60)
    print(f"Currencies ({len(config.ccys)}): {config.ccys}")
    print(f"Macros ({config.macro_dim}): {config.global_features}")
    print(f"Seed: {seed}")
    print("-" * 60)

    train_loader, test_loader = create_dataloaders(config)
    edge_index = fully_connected_edge_index(config.n_ccy).to(device)

    model = FXStrengthGNN(config)
    trainer = Trainer(model, config, device)
    trainer.train(train_loader, test_loader, edge_index, label=name)

    # Evaluate
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xl, xm, yb in test_loader:
            xl, xm = xl.to(device), xm.to(device)
            rhat, ds, z, m = model(xl, xm, edge_index)
            all_preds.append(rhat.cpu().numpy())
            all_targets.append(yb.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Calculate metrics
    results = {}
    mask = np.array([i != config.usd_idx for i in range(config.n_ccy)])

    # Overall
    correct = (np.sign(preds[:, mask]) == np.sign(targets[:, mask]))
    results['overall'] = {
        'hit_rate': float(correct.mean()),
        'weighted_hit_rate': float((correct * np.abs(targets[:, mask])).sum() / np.abs(targets[:, mask]).sum()),
        'n_samples': int(preds.shape[0])
    }

    # Per currency
    results['per_currency'] = {}
    for i, ccy in enumerate(config.ccys):
        if ccy == 'USD':
            continue
        c = (np.sign(preds[:, i]) == np.sign(targets[:, i]))
        w = np.abs(targets[:, i])
        results['per_currency'][ccy] = {
            'hit_rate': float(c.mean()),
            'weighted_hit_rate': float((c * w).sum() / w.sum()),
            'mur': float((np.sign(preds[:, i]) * targets[:, i]).mean())
        }

    # Save config
    config_dict = {
        'seed': seed,
        'ccys': config.ccys,
        'global_features': config.global_features,
        'lookback': config.lookback,
        'hidden': config.hidden,
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'lr': config.lr,
        'gnn_type': config.gnn_type,
        'heads': config.heads,
        'lambda_var': config.lambda_var,
        'lambda_a_l1': config.lambda_a_l1,
    }

    # Save A matrix
    A_matrix = model.A.detach().cpu().numpy().tolist()

    # Save everything
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))

    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    with open(os.path.join(save_path, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(save_path, 'A_matrix.json'), 'w') as f:
        json.dump({'ccys': config.ccys, 'macros': config.global_features, 'A': A_matrix}, f, indent=2)

    # Save predictions for verification
    np.savez(os.path.join(save_path, 'predictions.npz'), preds=preds, targets=targets)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Overall Hit Rate: {results['overall']['hit_rate']:.1%}")
    print(f"Weighted Hit Rate: {results['overall']['weighted_hit_rate']:.1%}")
    print("\nPer Currency:")
    for ccy, r in sorted(results['per_currency'].items(), key=lambda x: x[1]['hit_rate'], reverse=True):
        print(f"  {ccy}: {r['hit_rate']:.1%} (weighted: {r['weighted_hit_rate']:.1%})")

    print(f"\nSaved to: {save_path}")
    return save_path, results


def load_and_verify(save_path: str):
    """Load saved model and verify results match"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load config
    with open(os.path.join(save_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)

    with open(os.path.join(save_path, 'results.json'), 'r') as f:
        saved_results = json.load(f)

    # Set seed
    seed = config_dict['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Rebuild config
    config = Config()
    config.seed = seed
    # Note: ccys and global_features should match current config

    print("=" * 60)
    print(f"Loading model from: {save_path}")
    print("=" * 60)
    print(f"Saved config:")
    print(f"  Currencies: {config_dict['ccys']}")
    print(f"  Macros: {config_dict['global_features']}")
    print(f"  Seed: {seed}")

    # Check config match
    if config.ccys != config_dict['ccys']:
        print(f"\n⚠️  WARNING: Currency mismatch!")
        print(f"  Saved: {config_dict['ccys']}")
        print(f"  Current: {config.ccys}")
        return None

    if config.global_features != config_dict['global_features']:
        print(f"\n⚠️  WARNING: Macro mismatch!")
        print(f"  Saved: {config_dict['global_features']}")
        print(f"  Current: {config.global_features}")
        return None

    # Load model
    model = FXStrengthGNN(config)
    model.load_state_dict(torch.load(os.path.join(save_path, 'model.pt'), map_location=device))
    model.to(device)
    model.eval()

    # Load saved predictions
    saved_preds = np.load(os.path.join(save_path, 'predictions.npz'))

    # Re-evaluate
    train_loader, test_loader = create_dataloaders(config)
    edge_index = fully_connected_edge_index(config.n_ccy).to(device)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for xl, xm, yb in test_loader:
            xl, xm = xl.to(device), xm.to(device)
            rhat, ds, z, m = model(xl, xm, edge_index)
            all_preds.append(rhat.cpu().numpy())
            all_targets.append(yb.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Verify
    preds_match = np.allclose(preds, saved_preds['preds'], atol=1e-5)
    targets_match = np.allclose(targets, saved_preds['targets'], atol=1e-5)

    mask = np.array([i != config.usd_idx for i in range(config.n_ccy)])
    hit_rate = (np.sign(preds[:, mask]) == np.sign(targets[:, mask])).mean()

    print("\n" + "-" * 60)
    print("Verification:")
    print(f"  Predictions match: {'✓' if preds_match else '✗'}")
    print(f"  Targets match: {'✓' if targets_match else '✗'}")
    print(f"  Hit Rate: {hit_rate:.1%} (saved: {saved_results['overall']['hit_rate']:.1%})")

    if preds_match and targets_match:
        print("\n✓ Model successfully verified!")
    else:
        print("\n✗ Verification failed - results don't match")

    return model, config, saved_results


def compare_models(save_dir: str):
    """Compare all saved models"""

    models = []
    for name in os.listdir(save_dir):
        path = os.path.join(save_dir, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, 'results.json')):
            with open(os.path.join(path, 'config.json'), 'r') as f:
                config = json.load(f)
            with open(os.path.join(path, 'results.json'), 'r') as f:
                results = json.load(f)
            models.append({
                'name': name,
                'ccys': config['ccys'],
                'macros': config['global_features'],
                'hit_rate': results['overall']['hit_rate'],
                'weighted_hit_rate': results['overall']['weighted_hit_rate'],
                'per_currency': results['per_currency']
            })

    if not models:
        print("No saved models found")
        return

    print("=" * 80)
    print("Model Comparison")
    print("=" * 80)
    print(f"{'Name':<25} | {'CCYs':>5} | {'Macros':>6} | {'Hit':>7} | {'Weighted':>8}")
    print("-" * 80)

    for m in sorted(models, key=lambda x: x['hit_rate'], reverse=True):
        print(f"{m['name']:<25} | {len(m['ccys']):>5} | {len(m['macros']):>6} | {m['hit_rate']:>6.1%} | {m['weighted_hit_rate']:>7.1%}")

    return models


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'verify', 'compare'], default='train')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='saved_models')
    args = parser.parse_args()

    if args.mode == 'train':
        train_and_save(args.save_dir, seed=args.seed, name=args.name)
    elif args.mode == 'verify':
        if args.name is None:
            print("Please specify --name for verification")
        else:
            load_and_verify(os.path.join(args.save_dir, args.name))
    elif args.mode == 'compare':
        compare_models(args.save_dir)
