"""
Exp1: Run Dynamic Cross-Attention experiment
Compares: Static A vs Dynamic A (with/without GNN)
"""
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)  # Change to parent for data loading

import random
import json
from datetime import datetime
import numpy as np
import torch

from config import Config
from dataset import create_dataloaders, fully_connected_edge_index
from train import Trainer
from models import FXStrengthGNN, FXStrengthNoGNN

# Import local dynamic models
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from dynamic_models import FXStrengthDynamicA, FXStrengthDynamicANoGNN


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_model(model_name, model_cls, config, device, edge_index, seed=42):
    set_seed(seed)
    train_loader, test_loader = create_dataloaders(config, macro_mode="real")
    model = model_cls(config)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    trainer = Trainer(model, config, device)
    metrics = trainer.train(train_loader, test_loader, edge_index, label=model_name.upper())
    metrics['n_params'] = n_params
    return metrics


def main():
    seed = 42
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = Config(seed=seed, epochs=30, batch_size=128, lr=3e-4)
    config.use_skip_connection = True
    config.use_layer_norm = True
    config.heads = 4  # For attention

    edge_index = fully_connected_edge_index(config.n_ccy).to(device)

    results = {}
    models_to_run = [
        ("static_a", FXStrengthGNN),           # Baseline: Static A + GNN
        ("static_a_no_gnn", FXStrengthNoGNN),  # Static A without GNN
        ("dynamic_a", FXStrengthDynamicA),     # NEW: Dynamic A + GNN
        ("dynamic_a_no_gnn", FXStrengthDynamicANoGNN),  # NEW: Dynamic A without GNN
    ]

    print("\n" + "=" * 70)
    print("EXP1: DYNAMIC CROSS-ATTENTION A")
    print("=" * 70)

    for name, model_cls in models_to_run:
        print(f"\n>>> Running {name}...")
        metrics = run_model(name, model_cls, config, device, edge_index, seed)
        results[name] = {
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "hit": metrics["hit"],
            "mur": metrics["mur"],
            "hs_mean": metrics["hs_mean"],
            "n_params": metrics["n_params"],
        }

    # Print summary
    print("\n" + "=" * 70)
    print("EXP1 RESULTS: STATIC A vs DYNAMIC A")
    print("=" * 70)
    print(f"{'Model':<20} {'RMSE':>10} {'MAE':>10} {'Hit Rate':>10} {'Params':>12}")
    print("-" * 62)

    for name, _ in models_to_run:
        r = results[name]
        print(f"{name:<20} {r['rmse']:>10.4f} {r['mae']:>10.4f} {r['hit']:>10.4f} {r['n_params']:>12,}")

    # Comparison
    print("\n" + "-" * 62)
    print("Improvement Analysis:")
    print("-" * 62)

    # Static vs Dynamic (with GNN)
    static = results["static_a"]
    dynamic = results["dynamic_a"]
    print(f"\nDynamic A vs Static A (with GNN):")
    print(f"  ΔRMSE: {static['rmse'] - dynamic['rmse']:+.4f} (negative = dynamic better)")
    print(f"  ΔMAE:  {static['mae'] - dynamic['mae']:+.4f}")
    print(f"  ΔHit:  {dynamic['hit'] - static['hit']:+.4f} (positive = dynamic better)")

    # Static vs Dynamic (without GNN)
    static_ng = results["static_a_no_gnn"]
    dynamic_ng = results["dynamic_a_no_gnn"]
    print(f"\nDynamic A vs Static A (without GNN):")
    print(f"  ΔRMSE: {static_ng['rmse'] - dynamic_ng['rmse']:+.4f}")
    print(f"  ΔMAE:  {static_ng['mae'] - dynamic_ng['mae']:+.4f}")
    print(f"  ΔHit:  {dynamic_ng['hit'] - static_ng['hit']:+.4f}")

    # Best model
    best_model = min(results.keys(), key=lambda k: results[k]['rmse'])
    print(f"\nBest model by RMSE: {best_model} (RMSE={results[best_model]['rmse']:.4f})")

    best_hit = max(results.keys(), key=lambda k: results[k]['hit'])
    print(f"Best model by Hit:  {best_hit} (Hit={results[best_hit]['hit']:.4f})")

    # Save results
    output = {
        "experiment": "exp1_dynamic_attention",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "hidden": config.hidden,
            "heads": config.heads,
        },
        "results": results,
        "conclusion": {
            "best_rmse_model": best_model,
            "best_hit_model": best_hit,
            "dynamic_vs_static_gnn": {
                "rmse_improvement": static['rmse'] - dynamic['rmse'],
                "hit_improvement": dynamic['hit'] - static['hit'],
            },
            "dynamic_vs_static_no_gnn": {
                "rmse_improvement": static_ng['rmse'] - dynamic_ng['rmse'],
                "hit_improvement": dynamic_ng['hit'] - static_ng['hit'],
            }
        }
    }

    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to results.json")

    # Also save as text
    with open("results.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EXP1: DYNAMIC CROSS-ATTENTION A\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"{'Model':<20} {'RMSE':>10} {'MAE':>10} {'Hit Rate':>10} {'Params':>12}\n")
        f.write("-" * 62 + "\n")

        for name, _ in models_to_run:
            r = results[name]
            f.write(f"{name:<20} {r['rmse']:>10.4f} {r['mae']:>10.4f} {r['hit']:>10.4f} {r['n_params']:>12,}\n")

        f.write("\n" + "-" * 62 + "\n")
        f.write("Conclusion:\n")
        f.write(f"  Best RMSE: {best_model}\n")
        f.write(f"  Best Hit:  {best_hit}\n")

    print(f"Text results saved to results.txt")


if __name__ == "__main__":
    main()
