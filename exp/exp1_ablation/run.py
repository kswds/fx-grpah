"""
Exp1: Ablation Study - Component Contribution Analysis

Tests the value of each component:
- No GNN: Value of currency-currency spillover
- Homo A: Value of heterogeneous macro transmission
- No Macro: Value of macro-economic information
- Full: Complete model

Results: exp1_ablation/results.json
"""
import sys
import os
# Add project root to path (two levels up: exp/exp1_ablation -> exp -> project_root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import json
from datetime import datetime

from config import Config
from dataset import create_dataloaders, fully_connected_edge_index
from models import ABLATION_MODELS
from train import Trainer
from utils import set_seed, get_device, save_results

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_model(model_name, model_cls, config, device, edge_index, seed=42):
    """Train and evaluate a single model variant"""
    set_seed(seed)
    train_loader, test_loader = create_dataloaders(config, macro_mode="real")
    model = model_cls(config)
    trainer = Trainer(model, config, device)
    metrics = trainer.train(train_loader, test_loader, edge_index, label=model_name.upper())
    return metrics


def main():
    print("=" * 70)
    print("EXP1: ABLATION STUDY")
    print("=" * 70)

    seed = 42
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    config = Config(seed=seed, epochs=30, batch_size=128, lr=3e-4)
    edge_index = fully_connected_edge_index(config.n_ccy).to(device)

    results = {}
    model_order = ["full", "no_gnn", "homo_a", "no_macro"]

    for name in model_order:
        model_cls = ABLATION_MODELS[name]
        print(f"\n>>> Running {name}...")
        metrics = run_model(name, model_cls, config, device, edge_index, seed)
        results[name] = {
            "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
            "hit": float(metrics["hit"]),
            "mur": float(metrics["mur"]),
            "hs_mean": float(metrics["hs_mean"]),
        }

    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    print(f"{'Model':<15} {'RMSE':>10} {'MAE':>10} {'Hit Rate':>10}")
    print("-" * 45)

    for name in model_order:
        r = results[name]
        print(f"{name:<15} {r['rmse']:>10.4f} {r['mae']:>10.4f} {r['hit']:>10.1%}")

    # Component impact analysis
    full = results["full"]
    print("\n--- Component Impact (vs Full) ---")

    impact = {}
    ablation_mapping = {
        "no_gnn": "GNN (Currency Spillover)",
        "homo_a": "Heterogeneous A Matrix",
        "no_macro": "Macro Information",
    }

    for name, component in ablation_mapping.items():
        r = results[name]
        hit_drop = full["hit"] - r["hit"]
        impact[name] = {
            "component": component,
            "hit_drop": float(hit_drop),
            "rmse_increase": float(r["rmse"] - full["rmse"]),
        }
        print(f"Without {component}: ΔHit = {hit_drop:+.1%}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "seed": seed,
        },
        "results": results,
        "impact": impact,
    }

    save_results(output, OUTPUT_DIR, "results.json")
    print(f"\nResults saved to {OUTPUT_DIR}/results.json")


if __name__ == "__main__":
    main()
