"""
Exp2: Graph Structure Analysis
Tests different graph structures to understand why GNN doesn't help
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

from config import Config
from dataset import create_dataloaders
from models import FXStrengthGNN, FXStrengthNoGNN
from train import Trainer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Currency indices
# USD=0, EUR=1, JPY=2, GBP=3, CAD=4, AUD=5, CHF=6, NZD=7, SEK=8, NOK=9
CCY_IDX = {
    "USD": 0, "EUR": 1, "JPY": 2, "GBP": 3, "CAD": 4,
    "AUD": 5, "CHF": 6, "NZD": 7, "SEK": 8, "NOK": 9
}


def create_edge_index(edges, device):
    """Create edge_index tensor from list of (src, dst) tuples"""
    if len(edges) == 0:
        # Return minimal valid edge index (self-loop on node 0)
        return torch.tensor([[0], [0]], dtype=torch.long, device=device)

    # Make bidirectional
    all_edges = []
    for src, dst in edges:
        all_edges.append((src, dst))
        all_edges.append((dst, src))

    src_nodes = [e[0] for e in all_edges]
    dst_nodes = [e[1] for e in all_edges]

    return torch.tensor([src_nodes, dst_nodes], dtype=torch.long, device=device)


def fully_connected_edge_index(n_nodes, device):
    """Create fully connected graph"""
    edges = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                edges.append((i, j))
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    return torch.tensor([src, dst], dtype=torch.long, device=device)


def get_graph_structures(n_ccy, device):
    """Define different graph structures"""
    graphs = {}

    # 1. No graph (will use NoGNN model)
    graphs["no_graph"] = None

    # 2. Fully connected
    graphs["full"] = fully_connected_edge_index(n_ccy, device)

    # 3. Regional clusters
    # Americas: USD(0) - CAD(4)
    # Europe: EUR(1) - GBP(3) - CHF(6) - SEK(8) - NOK(9)
    # Asia-Pacific: JPY(2) - AUD(5) - NZD(7)
    regional_edges = [
        # Americas
        (0, 4),
        # Europe (chain)
        (1, 3), (3, 6), (6, 8), (8, 9), (1, 6), (1, 8), (1, 9), (3, 8), (3, 9),
        # Asia-Pacific
        (2, 5), (5, 7), (2, 7),
    ]
    graphs["regional"] = create_edge_index(regional_edges, device)

    # 4. Commodity currencies
    # AUD(5), CAD(4), NOK(9), NZD(7) - all connected
    commodity_edges = [
        (5, 4), (5, 9), (5, 7),
        (4, 9), (4, 7),
        (9, 7),
    ]
    graphs["commodity"] = create_edge_index(commodity_edges, device)

    # 5. Safe haven currencies
    # USD(0), JPY(2), CHF(6)
    safe_haven_edges = [
        (0, 2), (0, 6), (2, 6),
    ]
    graphs["safe_haven"] = create_edge_index(safe_haven_edges, device)

    # 6. USD as hub (star graph)
    # USD connected to all others
    usd_hub_edges = [(0, i) for i in range(1, n_ccy)]
    graphs["usd_hub"] = create_edge_index(usd_hub_edges, device)

    # 7. Major pairs only (G7 currencies tightly connected)
    # USD(0), EUR(1), JPY(2), GBP(3), CAD(4) - all connected
    major_edges = []
    g7 = [0, 1, 2, 3, 4]
    for i in g7:
        for j in g7:
            if i < j:
                major_edges.append((i, j))
    graphs["major_pairs"] = create_edge_index(major_edges, device)

    # 8. Hybrid: Regional + Commodity links
    hybrid_edges = regional_edges + [
        # Cross-regional commodity links
        (4, 5), (4, 9),  # CAD to AUD, NOK
        (5, 9),  # AUD to NOK
    ]
    graphs["hybrid"] = create_edge_index(hybrid_edges, device)

    return graphs


def run_experiment(model_cls, graph_name, edge_index, config, device, seed=42):
    """Run a single experiment"""
    set_seed(seed)
    train_loader, test_loader = create_dataloaders(config, macro_mode="real")
    model = model_cls(config)
    trainer = Trainer(model, config, device)
    metrics = trainer.train(train_loader, test_loader, edge_index, label=f"{graph_name.upper()}")
    return metrics


def main():
    seed = 42
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = Config(seed=seed, epochs=30, batch_size=128, lr=3e-4)
    config.use_skip_connection = True
    config.use_layer_norm = True

    # Get all graph structures
    graphs = get_graph_structures(config.n_ccy, device)

    results = {}

    print("\n" + "=" * 70)
    print("EXP2: GRAPH STRUCTURE ANALYSIS")
    print("=" * 70)

    for graph_name, edge_index in graphs.items():
        print(f"\n>>> Testing graph: {graph_name}")

        if edge_index is not None:
            print(f"    Edges: {edge_index.size(1)}")

        if graph_name == "no_graph":
            # Use NoGNN model
            metrics = run_experiment(
                FXStrengthNoGNN,
                graph_name,
                fully_connected_edge_index(config.n_ccy, device),  # Dummy, not used
                config, device, seed
            )
        else:
            # Use GNN model with specific graph
            metrics = run_experiment(
                FXStrengthGNN,
                graph_name,
                edge_index,
                config, device, seed
            )

        results[graph_name] = {
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "hit": metrics["hit"],
            "n_edges": edge_index.size(1) if edge_index is not None else 0,
        }

    # Summary
    print("\n" + "=" * 70)
    print("EXP2 RESULTS: GRAPH STRUCTURE COMPARISON")
    print("=" * 70)
    print(f"{'Graph':<15} {'Edges':>8} {'RMSE':>10} {'MAE':>10} {'Hit Rate':>10}")
    print("-" * 53)

    # Sort by RMSE
    sorted_results = sorted(results.items(), key=lambda x: x[1]['rmse'])

    for name, r in sorted_results:
        print(f"{name:<15} {r['n_edges']:>8} {r['rmse']:>10.4f} {r['mae']:>10.4f} {r['hit']:>10.4f}")

    # Analysis
    print("\n" + "-" * 53)
    print("Analysis:")
    print("-" * 53)

    best_name, best_r = sorted_results[0]
    worst_name, worst_r = sorted_results[-1]
    full_r = results["full"]
    no_graph_r = results["no_graph"]

    print(f"Best graph:  {best_name} (RMSE={best_r['rmse']:.4f})")
    print(f"Worst graph: {worst_name} (RMSE={worst_r['rmse']:.4f})")
    print(f"\nFull vs No Graph:")
    print(f"  Full graph RMSE:     {full_r['rmse']:.4f}")
    print(f"  No graph RMSE:       {no_graph_r['rmse']:.4f}")
    print(f"  Difference:          {full_r['rmse'] - no_graph_r['rmse']:+.4f}")

    if no_graph_r['rmse'] < full_r['rmse']:
        print("\n  → No graph is BETTER than full graph!")
        print("  → GNN with full connectivity hurts performance")

    # Check if any sparse graph beats no_graph
    sparse_better = [n for n, r in sorted_results
                     if n not in ["no_graph", "full"] and r['rmse'] < no_graph_r['rmse']]
    if sparse_better:
        print(f"\n  → Sparse graphs that beat no_graph: {sparse_better}")
    else:
        print("\n  → No sparse graph beats no_graph")
        print("  → GNN may not be useful for this task")

    # Save results
    output = {
        "experiment": "exp2_graph_structure",
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "ranking": [name for name, _ in sorted_results],
        "conclusion": {
            "best_graph": best_name,
            "best_rmse": best_r['rmse'],
            "full_vs_no_graph": full_r['rmse'] - no_graph_r['rmse'],
            "sparse_better_than_no_graph": sparse_better,
        }
    }

    with open("exp2_graph_structure/results.json", "w") as f:
        json.dump(output, f, indent=2)

    with open("exp2_graph_structure/results.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EXP2: GRAPH STRUCTURE ANALYSIS\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"{'Graph':<15} {'Edges':>8} {'RMSE':>10} {'MAE':>10} {'Hit Rate':>10}\n")
        f.write("-" * 53 + "\n")

        for name, r in sorted_results:
            f.write(f"{name:<15} {r['n_edges']:>8} {r['rmse']:>10.4f} {r['mae']:>10.4f} {r['hit']:>10.4f}\n")

    print(f"\nResults saved to exp2_graph_structure/")


if __name__ == "__main__":
    main()
