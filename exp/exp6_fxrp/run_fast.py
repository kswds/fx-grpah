"""
Exp8: Fast Comparison with Yoonsik-style Model

Simplified comparison:
1. Our model (FXStrengthGNN) - Node-level prediction
2. Edge-level GNN (Yoonsik-style) - Edge-level prediction with IR
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import Config
from dataset import create_dataloaders, fully_connected_edge_index, load_data
from models import FXStrengthGNN


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EdgeLevelGNN(nn.Module):
    """
    Yoonsik-style Edge-level GNN for FX rate prediction.

    Key differences from our model:
    1. Predicts FX rate changes (edge-level) instead of currency strength (node-level)
    2. Uses only IR as node features (no macro factors)
    """
    def __init__(self, n_ccy, node_dim, hidden_dim=64):
        super().__init__()
        self.n_ccy = n_ccy
        self.hidden_dim = hidden_dim

        # Node encoder (from IR features)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message passing layers
        self.mp1 = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # src, dst, edge
            nn.LeakyReLU(0.2),
        )
        self.mp2 = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Edge predictor
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_feat, edge_index):
        """
        Args:
            node_feat: [B, N, node_dim]
            edge_index: [2, E]
        Returns:
            edge_pred: [B, E] - predicted FX rate changes
        """
        B, N, _ = node_feat.shape
        src, dst = edge_index
        E = edge_index.size(1)

        # Encode nodes
        h = self.node_encoder(node_feat)  # [B, N, hidden]

        # Message passing (simplified)
        h_src = h[:, src]  # [B, E, hidden]
        h_dst = h[:, dst]  # [B, E, hidden]

        # Edge embeddings (initialized as difference)
        e = h_src - h_dst  # [B, E, hidden]

        # Layer 1
        msg = torch.cat([h_src, e, h_dst], dim=-1)  # [B, E, hidden*3]
        e = self.mp1(msg)  # [B, E, hidden]

        # Update node features by aggregation
        h_new = torch.zeros_like(h)
        for i in range(E):
            h_new[:, dst[i]] += e[:, i]
        h = h + h_new / (N - 1)  # residual + mean aggregation

        # Layer 2
        h_src = h[:, src]
        h_dst = h[:, dst]
        msg = torch.cat([h_src, e, h_dst], dim=-1)
        e = self.mp2(msg)

        # Edge prediction
        edge_feat = torch.cat([h_src, h_dst], dim=-1)
        edge_pred = self.edge_predictor(edge_feat).squeeze(-1)  # [B, E]

        return edge_pred


class EdgeDataset(Dataset):
    """Dataset for edge-level prediction (Yoonsik-style)"""
    def __init__(self, node_features, fx_returns, edge_index, lookback=20):
        """
        Args:
            node_features: [T, N, node_dim] - IR features
            fx_returns: [T, N, N] - FX rate returns
            edge_index: [2, E]
        """
        self.node_features = node_features
        self.fx_returns = fx_returns
        self.edge_index = edge_index
        self.lookback = lookback

    def __len__(self):
        return len(self.node_features) - self.lookback

    def __getitem__(self, idx):
        # Node features (use last lookback window)
        node_seq = self.node_features[idx:idx+self.lookback]
        node_feat = node_seq.mean(axis=0)  # Average over lookback

        # Edge targets (FX returns at next step)
        src, dst = self.edge_index
        edge_targets = self.fx_returns[idx+self.lookback, src, dst]

        return (
            torch.tensor(node_feat, dtype=torch.float32),
            torch.tensor(edge_targets, dtype=torch.float32)
        )


def prepare_edge_data(config):
    """Prepare data for edge-level model"""
    df = load_data(config)
    T = len(df)
    N = config.n_ccy
    ccys = config.ccys

    # Node features: IR changes (dY10)
    node_features = np.zeros((T, N, 1))
    for i, ccy in enumerate(ccys):
        ir_col = f"{ccy}_Yield10Y"
        if ir_col in df.columns:
            ir = df[ir_col].values
            ir_diff = np.diff(ir, prepend=ir[0])
            node_features[:, i, 0] = ir_diff

    # FX rate returns matrix
    fx_returns = np.zeros((T, N, N))

    # Get FX rates for each currency vs USD
    fx_vs_usd = {}
    for ccy in ccys:
        if ccy == 'USD':
            fx_vs_usd[ccy] = np.ones(T)
        else:
            fx_col = f"{ccy}_FX"
            if fx_col in df.columns:
                fx_vs_usd[ccy] = df[fx_col].values

    # Compute log returns for each pair
    for i, ccy_i in enumerate(ccys):
        for j, ccy_j in enumerate(ccys):
            if i != j:
                # FX rate i/j = (i/USD) / (j/USD)
                if ccy_i in fx_vs_usd and ccy_j in fx_vs_usd:
                    rate = fx_vs_usd[ccy_i] / (fx_vs_usd[ccy_j] + 1e-10)
                    log_rate = np.log(rate + 1e-10)
                    fx_returns[:, i, j] = np.diff(log_rate, prepend=log_rate[0])

    # Normalize
    node_features = (node_features - np.nanmean(node_features)) / (np.nanstd(node_features) + 1e-6)
    node_features = np.nan_to_num(node_features, 0)

    return node_features, fx_returns


def train_edge_model(model, train_loader, edge_index, device, epochs=30, lr=3e-4):
    """Train edge-level model"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    edge_index = edge_index.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for node_feat, edge_targets in train_loader:
            node_feat = node_feat.to(device)
            edge_targets = edge_targets.to(device)

            pred = model(node_feat, edge_index)
            loss = F.mse_loss(pred, edge_targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

    return model


def evaluate_edge_model(model, test_loader, edge_index, device):
    """Evaluate edge-level model"""
    model.eval()
    edge_index = edge_index.to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for node_feat, edge_targets in test_loader:
            node_feat = node_feat.to(device)
            pred = model(node_feat, edge_index)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(edge_targets.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    mse = ((preds - targets) ** 2).mean()
    hit_rate = (np.sign(preds) == np.sign(targets)).mean()

    return {'mse': mse, 'rmse': np.sqrt(mse), 'hit_rate': hit_rate}


class OurTrainer:
    """Our model trainer"""
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)

    def train(self, train_loader, edge_index, epochs):
        for _ in range(epochs):
            self.model.train()
            for xl, xm, yb in train_loader:
                xl, xm, yb = xl.to(self.device), xm.to(self.device), yb.to(self.device)
                rhat, ds, _, _ = self.model(xl, xm, edge_index)

                mask = torch.ones(self.config.n_ccy, dtype=torch.bool, device=yb.device)
                mask[self.config.usd_idx] = False
                mse = ((rhat[:, mask] - yb[:, mask]) ** 2).mean()
                loss = mse - 0.005 * ds.var(dim=1).mean() + 1e-4 * self.model.A.abs().mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

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

        mask = np.ones(self.config.n_ccy, dtype=bool)
        mask[self.config.usd_idx] = False

        mse = ((preds[:, mask] - targets[:, mask]) ** 2).mean()
        hit_rate = (np.sign(preds[:, mask]) == np.sign(targets[:, mask])).mean()

        return {'mse': mse, 'rmse': np.sqrt(mse), 'hit_rate': hit_rate, 'preds': preds, 'targets': targets}


def main():
    seed = 42
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\n" + "=" * 70)
    print("EXP8: COMPARISON - Our Model vs Yoonsik-style EdgeGNN")
    print("=" * 70)

    config = Config(seed=seed, epochs=30, batch_size=128, lr=3e-4)
    config.use_skip_connection = True
    config.use_layer_norm = True

    # Prepare data
    print("\n>>> Preparing data...")

    # Our model
    train_loader, test_loader = create_dataloaders(config, macro_mode="real")
    edge_index = fully_connected_edge_index(config.n_ccy).to(device)

    # Edge-level model data
    node_features, fx_returns = prepare_edge_data(config)

    # Create edge index for edge model
    edge_list = [(i, j) for i in range(config.n_ccy) for j in range(config.n_ccy) if i != j]
    edge_index_np = np.array(edge_list).T
    edge_index_edge = torch.tensor(edge_index_np, dtype=torch.long)

    # Train/test split
    T = len(node_features)
    split = int(T * 0.8)

    train_dataset = EdgeDataset(node_features[:split], fx_returns[:split], edge_index_np)
    test_dataset = EdgeDataset(node_features[split-20:], fx_returns[split-20:], edge_index_np)

    edge_train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    edge_test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    results = {}

    # 1. Our model (with macro)
    print("\n>>> Training Our Model (with 7 macro factors)...")
    our_model = FXStrengthGNN(config)
    our_trainer = OurTrainer(our_model, config, device)
    our_trainer.train(train_loader, edge_index, config.epochs)
    our_results = our_trainer.evaluate(test_loader, edge_index)
    results['Ours (7 macro)'] = {'rmse': our_results['rmse'], 'hit_rate': our_results['hit_rate']}
    print(f"  RMSE: {our_results['rmse']:.4f}, Hit Rate: {our_results['hit_rate']*100:.2f}%")

    # 2. Our model (no macro - fair comparison)
    print("\n>>> Training Our Model (no macro)...")
    train_loader_zero, test_loader_zero = create_dataloaders(config, macro_mode="zero")
    our_model_zero = FXStrengthGNN(config)
    our_trainer_zero = OurTrainer(our_model_zero, config, device)
    our_trainer_zero.train(train_loader_zero, edge_index, config.epochs)
    our_zero_results = our_trainer_zero.evaluate(test_loader_zero, edge_index)
    results['Ours (no macro)'] = {'rmse': our_zero_results['rmse'], 'hit_rate': our_zero_results['hit_rate']}
    print(f"  RMSE: {our_zero_results['rmse']:.4f}, Hit Rate: {our_zero_results['hit_rate']*100:.2f}%")

    # 3. Edge-level GNN (Yoonsik-style)
    print("\n>>> Training Yoonsik-style EdgeGNN (IR only)...")
    edge_model = EdgeLevelGNN(config.n_ccy, node_dim=1, hidden_dim=64)
    edge_model = train_edge_model(edge_model, edge_train_loader, edge_index_edge, device, epochs=30)
    edge_results = evaluate_edge_model(edge_model, edge_test_loader, edge_index_edge, device)
    results['Yoonsik-style'] = {'rmse': edge_results['rmse'], 'hit_rate': edge_results['hit_rate']}
    print(f"  RMSE: {edge_results['rmse']:.4f}, Hit Rate: {edge_results['hit_rate']*100:.2f}%")

    # Multi-seed
    print("\n>>> Multi-seed evaluation (3 seeds)...")
    seeds = [42, 123, 456]
    multi_results = {'Ours (7 macro)': [], 'Yoonsik-style': []}

    for s in seeds:
        set_seed(s)

        # Our model
        cfg = Config(seed=s, epochs=30, batch_size=128, lr=3e-4)
        cfg.use_skip_connection = True
        cfg.use_layer_norm = True
        train_ld, test_ld = create_dataloaders(cfg, macro_mode="real")
        model = FXStrengthGNN(cfg)
        trainer = OurTrainer(model, cfg, device)
        trainer.train(train_ld, edge_index, 30)
        res = trainer.evaluate(test_ld, edge_index)
        multi_results['Ours (7 macro)'].append(res['hit_rate'])

        # Edge model
        e_model = EdgeLevelGNN(config.n_ccy, node_dim=1, hidden_dim=64)
        e_model = train_edge_model(e_model, edge_train_loader, edge_index_edge, device, epochs=30)
        e_res = evaluate_edge_model(e_model, edge_test_loader, edge_index_edge, device)
        multi_results['Yoonsik-style'].append(e_res['hit_rate'])

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<25} {'RMSE':>10} {'Hit Rate':>12}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<25} {res['rmse']:>10.4f} {res['hit_rate']*100:>11.2f}%")

    print(f"\nMulti-seed Hit Rate:")
    for name in multi_results:
        mean = np.mean(multi_results[name])
        std = np.std(multi_results[name])
        print(f"  {name:<20}: {mean*100:.2f}% +/- {std*100:.2f}%")

    # Key insight
    our_mean = np.mean(multi_results['Ours (7 macro)'])
    edge_mean = np.mean(multi_results['Yoonsik-style'])
    improvement = (our_mean - edge_mean) / edge_mean * 100

    print("\n" + "=" * 70)
    print("KEY COMPARISON")
    print("=" * 70)
    print(f"""
| Model                | Target Level | Macro Input | Hit Rate |
|----------------------|--------------|-------------|----------|
| Yoonsik-style        | Edge (FX)    | IR only     | {edge_mean*100:.1f}%    |
| Ours (no macro)      | Node (ccy)   | None        | {our_zero_results['hit_rate']*100:.1f}%    |
| Ours (with macro)    | Node (ccy)   | 7 factors   | {our_mean*100:.1f}%    |

Our model advantage: +{improvement:.1f}% hit rate over Yoonsik-style
""")

    # Save
    output_dir = "exp8_yoonsik_comparison"
    os.makedirs(output_dir, exist_ok=True)

    output = {
        'timestamp': datetime.now().isoformat(),
        'results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
        'multi_seed': {k: [float(x) for x in v] for k, v in multi_results.items()},
        'improvement_pct': float(improvement),
    }

    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
