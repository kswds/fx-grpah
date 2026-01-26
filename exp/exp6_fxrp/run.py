"""
Exp8: Comparison with Yoonsik's FXRP Model (arxiv 2508.14784)

Fair comparison on the same data:
1. Our model (FXStrengthGNN) - Node-level, Hetero A, 7 macro factors
2. Yoonsik's model (EdgeGNN) - Edge-level, IR only
"""
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'baselines'))
os.chdir(parent_dir)

import random
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import Config
from dataset import create_dataloaders, fully_connected_edge_index, load_data, build_features
from models import FXStrengthGNN
from yoonsik_model import YoonsikFXRP, YoonsikDataProcessor


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_yoonsik_data(config):
    """
    Prepare data for Yoonsik's model from our dataset.

    Yoonsik uses:
    - FX rates as edge features
    - Interest rates as node features
    """
    df = load_data(config)

    # Extract FX matrix [T, N, N]
    T = len(df)
    N = config.n_ccy
    ccys = config.ccys

    # FX rates
    fx_matrix = np.ones((T, N, N))
    for i, ccy_i in enumerate(ccys):
        for j, ccy_j in enumerate(ccys):
            if i != j:
                # We need to reconstruct FX rates from our data
                # Our data has {ccy}_FX which is vs USD
                if ccy_i == 'USD':
                    # USD/j = 1 / (j_FX)
                    fx_col = f"{ccy_j}_FX"
                    if fx_col in df.columns:
                        fx_matrix[:, i, j] = 1.0 / df[fx_col].values
                elif ccy_j == 'USD':
                    # i/USD = i_FX
                    fx_col = f"{ccy_i}_FX"
                    if fx_col in df.columns:
                        fx_matrix[:, i, j] = df[fx_col].values
                else:
                    # Cross rate: i/j = (i/USD) / (j/USD)
                    fx_i = df[f"{ccy_i}_FX"].values
                    fx_j = df[f"{ccy_j}_FX"].values
                    fx_matrix[:, i, j] = fx_i / fx_j

    # Handle missing/invalid values
    fx_matrix = np.nan_to_num(fx_matrix, nan=1.0, posinf=1.0, neginf=1.0)
    fx_matrix = np.clip(fx_matrix, 0.001, 1000)

    # Interest rates (10Y yields)
    ir_matrix = np.zeros((T, N))
    for i, ccy in enumerate(ccys):
        ir_col = f"{ccy}_Yield10Y"
        if ir_col in df.columns:
            ir_matrix[:, i] = df[ir_col].values / 100  # Convert to decimal

    # Handle missing values
    ir_matrix = np.nan_to_num(ir_matrix, nan=0.02)  # Default 2%

    return fx_matrix, ir_matrix


class OurModelTrainer:
    """Trainer for our FXStrengthGNN"""
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)

    def loss_fn(self, rhat, y, ds, A_param):
        mask = torch.ones(self.config.n_ccy, dtype=torch.bool, device=y.device)
        mask[self.config.usd_idx] = False
        mse = ((rhat[:, mask] - y[:, mask]) ** 2).mean()
        var_term = -ds.var(dim=1).mean()
        l1_A = A_param.abs().mean()
        return mse + self.config.lambda_var * var_term + self.config.lambda_a_l1 * l1_A

    def train_epoch(self, train_loader, edge_index):
        self.model.train()
        total_loss = 0
        n_batches = 0
        for xl, xm, yb in train_loader:
            xl, xm, yb = xl.to(self.device), xm.to(self.device), yb.to(self.device)
            rhat, ds, _, _ = self.model(xl, xm, edge_index)
            loss = self.loss_fn(rhat, yb, ds, self.model.A)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / n_batches

    def train(self, train_loader, edge_index, epochs):
        for _ in range(epochs):
            self.train_epoch(train_loader, edge_index)

    def evaluate(self, test_loader, edge_index):
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xl, xm, yb in test_loader:
                xl, xm, yb = xl.to(self.device), xm.to(self.device), yb.to(self.device)
                rhat, _, _, _ = self.model(xl, xm, edge_index)
                all_preds.append(rhat.cpu().numpy())
                all_targets.append(yb.cpu().numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        # Exclude USD
        mask = np.ones(self.config.n_ccy, dtype=bool)
        mask[self.config.usd_idx] = False

        preds_masked = preds[:, mask]
        targets_masked = targets[:, mask]

        mse = ((preds_masked - targets_masked) ** 2).mean()
        hit_rate = (np.sign(preds_masked) == np.sign(targets_masked)).mean()

        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'hit_rate': hit_rate,
            'preds': preds,
            'targets': targets,
        }


class YoonsikModelTrainer:
    """Trainer for Yoonsik's model adapted to our data"""
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 3e-4))

    def train(self, node_feat, edge_feat, targets, edge_index, epochs):
        """Train the model"""
        node_feat = torch.tensor(node_feat, dtype=torch.float32, device=self.device)
        edge_feat = torch.tensor(edge_feat, dtype=torch.float32, device=self.device)
        targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        edge_index = edge_index.to(self.device)

        T = node_feat.size(0)
        batch_size = self.config.get('batch_size', 64)

        for epoch in range(epochs):
            self.model.train()
            indices = np.random.permutation(T)
            total_loss = 0
            n_batches = 0

            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                batch_idx = indices[start:end]

                batch_loss = 0
                for t in batch_idx:
                    pred = self.model(node_feat[t], edge_feat[t], edge_index)
                    loss = F.mse_loss(pred, targets[t])
                    batch_loss += loss

                batch_loss = batch_loss / len(batch_idx)

                self.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += batch_loss.item()
                n_batches += 1

    def evaluate(self, node_feat, edge_feat, targets, edge_index):
        """Evaluate model"""
        self.model.eval()

        node_feat = torch.tensor(node_feat, dtype=torch.float32, device=self.device)
        edge_feat = torch.tensor(edge_feat, dtype=torch.float32, device=self.device)
        targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        edge_index = edge_index.to(self.device)

        T = node_feat.size(0)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for t in range(T):
                pred = self.model(node_feat[t], edge_feat[t], edge_index)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(targets[t].cpu().numpy())

        preds = np.array(all_preds)
        tgt = np.array(all_targets)

        mse = ((preds - tgt) ** 2).mean()
        hit_rate = (np.sign(preds) == np.sign(tgt)).mean()

        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'hit_rate': hit_rate,
            'preds': preds,
            'targets': tgt,
        }


def main():
    seed = 42
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\n" + "=" * 70)
    print("EXP8: COMPARISON WITH YOONSIK'S MODEL")
    print("=" * 70)

    # Our model config
    config = Config(seed=seed, epochs=30, batch_size=128, lr=3e-4)
    config.use_skip_connection = True
    config.use_layer_norm = True

    # Prepare data for both models
    print("\n>>> Preparing data...")

    # Our model data
    train_loader, test_loader = create_dataloaders(config, macro_mode="real")
    edge_index = fully_connected_edge_index(config.n_ccy).to(device)

    # Yoonsik model data
    fx_matrix, ir_matrix = prepare_yoonsik_data(config)

    # Process for Yoonsik
    processor = YoonsikDataProcessor(config.ccys, lookback_windows=[1, 3, 5, 10, 15, 20])
    node_feat, edge_feat, yoonsik_targets = processor.process_data(fx_matrix, ir_matrix)
    yoonsik_edge_index = processor.build_edge_index()

    # Train/test split (same as our model: 80/20)
    T = node_feat.shape[0]
    split_idx = int(T * 0.8)

    node_feat_train = node_feat[:split_idx]
    edge_feat_train = edge_feat[:split_idx]
    targets_train = yoonsik_targets[:split_idx]

    node_feat_test = node_feat[split_idx:]
    edge_feat_test = edge_feat[split_idx:]
    targets_test = yoonsik_targets[split_idx:]

    print(f"Data shape - Node: {node_feat.shape}, Edge: {edge_feat.shape}")
    print(f"Train: {split_idx}, Test: {T - split_idx}")

    # Train and evaluate both models
    results = {}

    # 1. Our model (FXStrengthGNN)
    print("\n>>> Training Our Model (FXStrengthGNN)...")
    our_model = FXStrengthGNN(config)
    our_trainer = OurModelTrainer(our_model, config, device)
    our_trainer.train(train_loader, edge_index, config.epochs)

    our_results = our_trainer.evaluate(test_loader, edge_index)
    results['Ours'] = {
        'rmse': our_results['rmse'],
        'hit_rate': our_results['hit_rate'],
    }
    print(f"Our Model - RMSE: {our_results['rmse']:.4f}, Hit Rate: {our_results['hit_rate']:.4f}")

    # 2. Yoonsik's model
    print("\n>>> Training Yoonsik's Model (EdgeGNN with IR)...")
    yoonsik_model = YoonsikFXRP(
        n_ccy=config.n_ccy,
        node_feat_dim=node_feat.shape[-1],
        edge_feat_dim=edge_feat.shape[-1],
        hidden_dim=64,
        n_layers=3
    )
    yoonsik_config = {'lr': 3e-4, 'batch_size': 64}
    yoonsik_trainer = YoonsikModelTrainer(yoonsik_model, yoonsik_config, device)
    yoonsik_trainer.train(node_feat_train, edge_feat_train, targets_train,
                          yoonsik_edge_index, epochs=30)

    yoonsik_results = yoonsik_trainer.evaluate(node_feat_test, edge_feat_test,
                                                targets_test, yoonsik_edge_index)
    results['Yoonsik'] = {
        'rmse': yoonsik_results['rmse'],
        'hit_rate': yoonsik_results['hit_rate'],
    }
    print(f"Yoonsik's Model - RMSE: {yoonsik_results['rmse']:.4f}, Hit Rate: {yoonsik_results['hit_rate']:.4f}")

    # 3. Our model without macro (ablation)
    print("\n>>> Training Our Model WITHOUT Macro...")
    train_loader_no_macro, test_loader_no_macro = create_dataloaders(config, macro_mode="zero")
    our_model_no_macro = FXStrengthGNN(config)
    our_trainer_no_macro = OurModelTrainer(our_model_no_macro, config, device)
    our_trainer_no_macro.train(train_loader_no_macro, edge_index, config.epochs)

    our_no_macro_results = our_trainer_no_macro.evaluate(test_loader_no_macro, edge_index)
    results['Ours (no macro)'] = {
        'rmse': our_no_macro_results['rmse'],
        'hit_rate': our_no_macro_results['hit_rate'],
    }
    print(f"Our Model (no macro) - RMSE: {our_no_macro_results['rmse']:.4f}, Hit Rate: {our_no_macro_results['hit_rate']:.4f}")

    # Multi-seed evaluation
    print("\n>>> Multi-seed evaluation...")
    seeds = [42, 123, 456, 789, 1000]

    multi_seed_results = {'Ours': [], 'Yoonsik': []}

    for s in seeds:
        set_seed(s)

        # Our model
        train_loader, test_loader = create_dataloaders(Config(seed=s, epochs=30, batch_size=128, lr=3e-4), macro_mode="real")
        cfg = Config(seed=s, epochs=30, batch_size=128, lr=3e-4)
        cfg.use_skip_connection = True
        cfg.use_layer_norm = True
        model = FXStrengthGNN(cfg)
        trainer = OurModelTrainer(model, cfg, device)
        trainer.train(train_loader, edge_index, 30)
        res = trainer.evaluate(test_loader, edge_index)
        multi_seed_results['Ours'].append(res['hit_rate'])

        # Yoonsik
        y_model = YoonsikFXRP(
            n_ccy=config.n_ccy,
            node_feat_dim=node_feat.shape[-1],
            edge_feat_dim=edge_feat.shape[-1],
            hidden_dim=64,
            n_layers=3
        )
        y_trainer = YoonsikModelTrainer(y_model, yoonsik_config, device)
        y_trainer.train(node_feat_train, edge_feat_train, targets_train,
                        yoonsik_edge_index, epochs=30)
        y_res = y_trainer.evaluate(node_feat_test, edge_feat_test,
                                   targets_test, yoonsik_edge_index)
        multi_seed_results['Yoonsik'].append(y_res['hit_rate'])

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print(f"\n{'Model':<25} {'RMSE':>10} {'Hit Rate':>12}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<25} {res['rmse']:>10.4f} {res['hit_rate']*100:>11.2f}%")

    print(f"\nMulti-seed Hit Rate (5 seeds):")
    print(f"  Ours:    {np.mean(multi_seed_results['Ours'])*100:.2f}% +/- {np.std(multi_seed_results['Ours'])*100:.2f}%")
    print(f"  Yoonsik: {np.mean(multi_seed_results['Yoonsik'])*100:.2f}% +/- {np.std(multi_seed_results['Yoonsik'])*100:.2f}%")

    # Key differences
    print("\n" + "=" * 70)
    print("KEY DIFFERENCES")
    print("=" * 70)
    print("""
| Aspect              | Ours (FXStrengthGNN)     | Yoonsik (EdgeGNN)        |
|---------------------|--------------------------|--------------------------|
| Prediction Level    | Node (currency strength) | Edge (FX rate)           |
| Macro Features      | 7 factors                | IR only                  |
| Key Component       | Heterogeneous A matrix   | Edge message passing     |
| Graph Structure     | Currency nodes           | Currency nodes, FX edges |
""")

    our_hit = np.mean(multi_seed_results['Ours'])
    yoonsik_hit = np.mean(multi_seed_results['Yoonsik'])
    improvement = (our_hit - yoonsik_hit) / yoonsik_hit * 100

    print(f"\n>>> Our model improvement over Yoonsik: {improvement:+.2f}%")

    # Save results
    output_dir = "exp8_yoonsik_comparison"
    os.makedirs(output_dir, exist_ok=True)

    output = {
        'timestamp': datetime.now().isoformat(),
        'single_seed_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
        'multi_seed_results': {
            'Ours': [float(x) for x in multi_seed_results['Ours']],
            'Yoonsik': [float(x) for x in multi_seed_results['Yoonsik']],
        },
        'our_mean_hit_rate': float(our_hit),
        'yoonsik_mean_hit_rate': float(yoonsik_hit),
        'improvement_pct': float(improvement),
    }

    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
