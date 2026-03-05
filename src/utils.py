"""
Shared utility functions for FX Strength GNN experiments
"""
import random
import numpy as np
import torch
import json
import os
from datetime import datetime


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get available device (CUDA or CPU)"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_results(results: dict, output_dir: str, filename: str = "results.json"):
    """Save results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")


def load_results(output_dir: str, filename: str = "results.json") -> dict:
    """Load results from JSON file"""
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'r') as f:
        return json.load(f)


def print_metrics(metrics: dict, label: str = ""):
    """Print evaluation metrics in consistent format"""
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | "
          f"Hit: {metrics['hit']:.1%} | MUR: {metrics['mur']:.4f}")


def format_experiment_header(title: str, width: int = 70):
    """Print formatted experiment header"""
    print("=" * width)
    print(title.center(width))
    print("=" * width)
