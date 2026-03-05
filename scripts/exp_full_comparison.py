"""
Full Experiment: Ridge vs MLP vs GNN × Lookback modes × Seeds

Parallelized with ProcessPoolExecutor for speed.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from itertools import combinations, product
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import Config
from dataset import load_data, build_features, normalize_data

# Macro features (excluding duplicates)
MACRO_FEATURES = ["Global_VIX", "Global_Gold", "Global_Oil", "Global_Copper", "Global_US2Y", "Global_IronOre"]


def compute_metrics(pred, target, usd_idx, n_ccy):
    """Compute Hit(ccy) and Hit(55)."""
    mask = np.ones(n_ccy, dtype=bool)
    mask[usd_idx] = False
    hit_ccy = np.mean(np.sign(pred[:, mask]) == np.sign(target[:, mask]))

    edges = list(combinations(range(n_ccy), 2))
    pred_pairs = np.stack([pred[:, i] - pred[:, j] for i, j in edges], axis=1)
    target_pairs = np.stack([target[:, i] - target[:, j] for i, j in edges], axis=1)
    hit_55 = np.mean(np.sign(pred_pairs) == np.sign(target_pairs))

    return hit_ccy, hit_55


def prepare_data_global():
    """Prepare data once (called in main process)."""
    config = Config(seed=42, lookback=20)
    df = load_data(config)
    X_local_base, X_macro_full, Y = build_features(df, config)

    macro_idx = [config.global_features.index(f) for f in MACRO_FEATURES]
    X_macro = X_macro_full[:, macro_idx]

    n_total = len(X_local_base)
    split_idx = int(n_total * 0.8)
    X_local_scaled, X_macro_scaled, Y_raw, _ = normalize_data(X_local_base, X_macro, Y, train_idx=split_idx)

    L = config.lookback
    n = n_total - L
    split = int(n * 0.8)

    return {
        'X_local_scaled': X_local_scaled,
        'X_macro_scaled': X_macro_scaled,
        'Y_raw': Y_raw,
        'L': L,
        'n': n,
        'split': split,
        'n_ccy': config.n_ccy,
        'usd_idx': config.usd_idx,
    }


# ============ Models ============
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimpleGNN(nn.Module):
    def __init__(self, local_dim, macro_dim, hidden_dim, n_ccy):
        super().__init__()
        self.node_enc = nn.Linear(local_dim + macro_dim, hidden_dim)
        self.msg1 = nn.Linear(hidden_dim, hidden_dim)
        self.msg2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        adj = torch.ones(n_ccy, n_ccy) - torch.eye(n_ccy)
        self.register_buffer('adj', adj / (n_ccy - 1))

    def forward(self, x_local, x_macro):
        B, N, D = x_local.shape
        x_macro_exp = x_macro.unsqueeze(1).expand(B, N, -1)
        x = torch.cat([x_local, x_macro_exp], dim=-1)
        h = torch.relu(self.node_enc(x))
        h = torch.relu(h + torch.matmul(self.adj, self.msg1(h)))
        h = torch.relu(h + torch.matmul(self.adj, self.msg2(h)))
        return self.out(h).squeeze(-1)


def run_single_experiment(args):
    """Run a single experiment (model × lookback × seed)."""
    model_name, lookback_mode, seed, data = args

    X_local_scaled = data['X_local_scaled']
    X_macro_scaled = data['X_macro_scaled']
    Y_raw = data['Y_raw']
    L = data['L']
    n = data['n']
    split = data['split']
    n_ccy = data['n_ccy']
    usd_idx = data['usd_idx']

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build features based on lookback mode
    X_local_list, X_macro_list, Y_list = [], [], []
    for idx in range(n):
        local_window = X_local_scaled[idx:idx+L]
        macro_window = X_macro_scaled[idx:idx+L]

        if lookback_mode == 'last':
            k = 1
        elif lookback_mode == 'full':
            k = L
        else:
            k = lookback_mode

        X_local_list.append(local_window[-k:])
        X_macro_list.append(macro_window[-k:])
        Y_list.append(Y_raw[idx + L])

    X_local = np.stack(X_local_list)  # [n, k, N, 3]
    X_macro = np.stack(X_macro_list)  # [n, k, M]
    Y = np.stack(Y_list)  # [n, N]

    # Split
    X_local_train, X_local_test = X_local[:split], X_local[split:]
    X_macro_train, X_macro_test = X_macro[:split], X_macro[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    # Normalize Y for NN training
    Y_train_mean = Y_train.mean(axis=0, keepdims=True)
    Y_train_std = Y_train.std(axis=0, keepdims=True) + 1e-8

    if model_name == 'Ridge':
        # Flatten features
        X_train = np.concatenate([X_local_train.reshape(len(X_local_train), -1),
                                   X_macro_train.reshape(len(X_macro_train), -1)], axis=1)
        X_test = np.concatenate([X_local_test.reshape(len(X_local_test), -1),
                                  X_macro_test.reshape(len(X_macro_test), -1)], axis=1)

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, Y_train)
        pred = ridge.predict(X_test)

    elif model_name == 'MLP':
        X_train = np.concatenate([X_local_train.reshape(len(X_local_train), -1),
                                   X_macro_train.reshape(len(X_macro_train), -1)], axis=1)
        X_test = np.concatenate([X_local_test.reshape(len(X_local_test), -1),
                                  X_macro_test.reshape(len(X_macro_test), -1)], axis=1)

        Y_train_norm = (Y_train - Y_train_mean) / Y_train_std

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        Y_train_t = torch.tensor(Y_train_norm, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)

        model = MLP(X_train.shape[1], 128, n_ccy)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)

        model.train()
        for _ in range(50):
            perm = np.random.permutation(len(X_train))
            for i in range(0, len(perm), 256):
                idx = perm[i:i+256]
                optimizer.zero_grad()
                loss = ((model(X_train_t[idx]) - Y_train_t[idx]) ** 2).mean()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_test_t).numpy() * Y_train_std + Y_train_mean

    elif model_name == 'GNN':
        # Keep structure for GNN
        Y_train_norm = (Y_train - Y_train_mean) / Y_train_std

        # Last day only for GNN (consistent with paper)
        X_local_train_last = X_local_train[:, -1]  # [n, N, 3]
        X_local_test_last = X_local_test[:, -1]
        X_macro_train_last = X_macro_train[:, -1]  # [n, M]
        X_macro_test_last = X_macro_test[:, -1]

        X_train_l_t = torch.tensor(X_local_train_last, dtype=torch.float32)
        X_train_m_t = torch.tensor(X_macro_train_last, dtype=torch.float32)
        Y_train_t = torch.tensor(Y_train_norm, dtype=torch.float32)
        X_test_l_t = torch.tensor(X_local_test_last, dtype=torch.float32)
        X_test_m_t = torch.tensor(X_macro_test_last, dtype=torch.float32)

        model = SimpleGNN(3, len(MACRO_FEATURES), 128, n_ccy)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)

        model.train()
        for _ in range(50):
            perm = np.random.permutation(len(X_train_l_t))
            for i in range(0, len(perm), 256):
                idx = perm[i:i+256]
                optimizer.zero_grad()
                loss = ((model(X_train_l_t[idx], X_train_m_t[idx]) - Y_train_t[idx]) ** 2).mean()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_test_l_t, X_test_m_t).numpy() * Y_train_std + Y_train_mean

    hit_ccy, hit_55 = compute_metrics(pred, Y_test, usd_idx, n_ccy)

    return {
        'model': model_name,
        'lookback': str(lookback_mode),
        'seed': seed,
        'hit_ccy': hit_ccy,
        'hit_55': hit_55,
    }


def main():
    print("=" * 90)
    print("FULL EXPERIMENT: Ridge vs MLP vs GNN × Lookback × Seeds")
    print("=" * 90)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Prepare data
    print("\nPreparing data...")
    data = prepare_data_global()
    print(f"Data: {data['n']} samples, Train={data['split']}, Test={data['n']-data['split']}")

    # Experiment config
    models = ['Ridge', 'MLP', 'GNN']
    lookbacks = ['last', 2, 5, 'full']  # Reduced for speed
    seeds = [42, 123, 456]  # 3 seeds

    # Generate all combinations
    experiments = []
    for model, lookback, seed in product(models, lookbacks, seeds):
        # GNN only uses last day (per original design)
        if model == 'GNN' and lookback != 'last':
            continue
        experiments.append((model, lookback, seed, data))

    print(f"Total experiments: {len(experiments)}")
    print(f"Running with 4 workers...")

    # Run experiments
    all_results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_single_experiment, exp): exp for exp in experiments}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            all_results.append(result)
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{len(experiments)}")

    print(f"\nAll experiments completed!")

    # Aggregate results
    summary = {}
    for model in models:
        for lookback in lookbacks:
            if model == 'GNN' and lookback != 'last':
                continue
            key = f"{model}_{lookback}"
            results = [r for r in all_results if r['model'] == model and r['lookback'] == str(lookback)]
            if results:
                summary[key] = {
                    'model': model,
                    'lookback': str(lookback),
                    'hit_ccy_mean': np.mean([r['hit_ccy'] for r in results]),
                    'hit_ccy_std': np.std([r['hit_ccy'] for r in results]),
                    'hit_55_mean': np.mean([r['hit_55'] for r in results]),
                    'hit_55_std': np.std([r['hit_55'] for r in results]),
                    'n_seeds': len(results),
                }

    # Print table
    print("\n" + "=" * 90)
    print("RESULTS TABLE")
    print("=" * 90)

    # Header
    print(f"\n{'Lookback':<12}", end="")
    for model in models:
        print(f" | {model:^24}", end="")
    print()
    print("-" * 90)

    # Hit(ccy) row
    for lookback in lookbacks:
        row = f"{str(lookback):<12}"
        for model in models:
            key = f"{model}_{lookback}"
            if key in summary:
                s = summary[key]
                row += f" | {s['hit_ccy_mean']:.2%} ± {s['hit_ccy_std']:.2%}"
            else:
                row += f" | {'--':^24}"
        print(f"Hit(ccy) {row}")

    print("-" * 90)

    # Hit(55) row
    for lookback in lookbacks:
        row = f"{str(lookback):<12}"
        for model in models:
            key = f"{model}_{lookback}"
            if key in summary:
                s = summary[key]
                row += f" | {s['hit_55_mean']:.2%} ± {s['hit_55_std']:.2%}"
            else:
                row += f" | {'--':^24}"
        print(f"Hit(55)  {row}")

    # Compact table for paper
    print("\n" + "=" * 90)
    print("COMPACT TABLE (Hit(55) only, for paper)")
    print("=" * 90)

    print(f"\n{'Lookback':<10}", end="")
    for model in models:
        print(f" | {model:^20}", end="")
    print()
    print("-" * 75)

    for lookback in lookbacks:
        print(f"{str(lookback):<10}", end="")
        for model in models:
            key = f"{model}_{lookback}"
            if key in summary:
                s = summary[key]
                print(f" | {s['hit_55_mean']:.2%} ± {s['hit_55_std']:.2%}", end="")
            else:
                print(f" | {'--':^20}", end="")
        print()

    # Save results
    output = {
        'experiment': 'full_comparison',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'models': models,
            'lookbacks': [str(l) for l in lookbacks],
            'seeds': seeds,
            'n_experiments': len(experiments),
        },
        'summary': summary,
        'all_results': all_results,
    }

    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp_full_comparison.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
