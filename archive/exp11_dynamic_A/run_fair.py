"""
Exp11: Dynamic A Matrix - Fair Comparison using same training infrastructure
"""

import sys
sys.path.append('/home/gdro/experiment/fx-graph-other')

import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv

from config import Config
from dataset import load_data, build_features, FXDataset, fully_connected_edge_index
from torch.utils.data import DataLoader
from train import loss_fn, Trainer

cfg = Config()
CURRENCIES = cfg.ccys
MACRO_FEATURES = [f.replace('Global_', '') for f in cfg.global_features]
VIX_IDX = 1


class FXStrengthGNNDynamicA(nn.Module):
    """FX Strength GNN with Dynamic A Matrix"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden = config.hidden
        self.n_ccy = config.n_ccy
        self.macro_dim = config.macro_dim
        self.usd_idx = config.usd_idx

        # Temporal encoding
        self.local_gru = nn.GRU(config.local_dim, config.hidden, batch_first=True)

        # Currency spillover (GAT)
        self.ccy_gnn = GATConv(config.hidden, config.hidden, heads=config.heads, concat=False)

        # Macro embedding
        self.macro_embed = nn.Linear(config.macro_dim, config.macro_dim * config.hidden, bias=False)

        # Dynamic A matrix
        self.A_base = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim))
        self.A_delta = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim))
        nn.init.normal_(self.A_base, mean=0.0, std=0.1)
        nn.init.normal_(self.A_delta, mean=0.0, std=0.05)

        # Regime gate from VIX
        self.regime_gate = nn.Sequential(
            nn.Linear(config.lookback, config.hidden // 2),
            nn.ReLU(),
            nn.Linear(config.hidden // 2, 1),
            nn.Sigmoid()
        )

        # Prediction head
        self.head = nn.Linear(config.hidden, 1)

    @property
    def A(self):
        return self.A_base

    def forward(self, xl, xm, edge_index_single):
        B, L, N, local_dim = xl.shape

        # 1. Local encoding
        x = xl.permute(0, 2, 1, 3).reshape(B * N, L, local_dim)
        _, h = self.local_gru(x)
        h = h.squeeze(0)

        # 2. GNN
        E = edge_index_single.size(1)
        edge_b = edge_index_single.repeat(1, B)
        offset = torch.arange(B, device=xl.device).repeat_interleave(E) * N
        edge_b = edge_b + offset.unsqueeze(0)

        z = self.ccy_gnn(h, edge_b)
        z_ccy = z.view(B, N, self.hidden)

        # 3. Macro embedding
        m_t = xm[:, -1, :]
        u = self.macro_embed(m_t).view(B, self.macro_dim, self.hidden)

        # 4. Dynamic A from VIX
        vix_seq = xm[:, :, VIX_IDX]
        regime_score = self.regime_gate(vix_seq)

        A_base_exp = self.A_base.unsqueeze(0).expand(B, -1, -1)
        A_delta_exp = self.A_delta.unsqueeze(0).expand(B, -1, -1)
        A_dynamic = A_base_exp + regime_score.unsqueeze(-1) * A_delta_exp

        # 5. Heterogeneous transmission
        A_exp = A_dynamic.unsqueeze(-1)
        u_exp = u.unsqueeze(1)
        m_msg = (A_exp * u_exp).sum(dim=2)

        # 6. Prediction
        z_total = z_ccy + m_msg
        ds = self.head(z_total).squeeze(-1)
        ds = ds - ds.mean(dim=1, keepdim=True)

        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]

        return rhat, ds, z_ccy, m_msg


def run_experiment():
    print("=" * 70)
    print("EXP11: DYNAMIC A MATRIX - FAIR COMPARISON")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = 'exp11_dynamic_A'

    # Load data (same as main.py)
    df = load_data(cfg)
    X_local_base, X_macro, Y = build_features(df, cfg)

    # Split
    n_total = len(X_local_base)
    split_idx = int(n_total * 0.8)

    train_local = X_local_base[:split_idx]
    train_macro = X_macro[:split_idx]

    # Normalize (same as main.py)
    local_mean = train_local.mean(axis=(0, 1), keepdims=True)
    local_std = train_local.std(axis=(0, 1), keepdims=True) + 1e-6
    macro_mean = train_macro.mean(axis=0, keepdims=True)
    macro_std = train_macro.std(axis=0, keepdims=True) + 1e-6

    X_local_scaled = (X_local_base - local_mean) / local_std
    X_macro_scaled = (X_macro - macro_mean) / macro_std

    # Create datasets
    train_ds = FXDataset(X_local_scaled[:split_idx], X_macro_scaled[:split_idx], Y[:split_idx], cfg)
    test_ds = FXDataset(X_local_scaled[split_idx:], X_macro_scaled[split_idx:], Y[split_idx:], cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    edge_index = fully_connected_edge_index(cfg.n_ccy).to(device)

    # Import baseline model
    from models import FXStrengthGNN

    # ======================================
    # Train Static A Model
    # ======================================
    print("\n>>> Training Static A Model...")
    static_model = FXStrengthGNN(cfg).to(device)
    static_trainer = Trainer(static_model, cfg, device)

    for epoch in range(cfg.epochs):
        static_trainer.train_epoch(train_loader, edge_index)

    # Evaluate static
    static_model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xl, xm, y in test_loader:
            xl, xm, y = xl.to(device), xm.to(device), y.to(device)
            rhat, ds, z_ccy, m_msg = static_model(xl, xm, edge_index)
            all_preds.append(ds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    static_preds = np.concatenate(all_preds, axis=0)
    static_targets = np.concatenate(all_targets, axis=0)
    static_hit = (np.sign(static_preds) == np.sign(static_targets)).mean()
    print(f"Static A Hit Rate: {static_hit:.1%}")

    # ======================================
    # Train Dynamic A Model
    # ======================================
    print("\n>>> Training Dynamic A Model...")
    dynamic_model = FXStrengthGNNDynamicA(cfg).to(device)
    optimizer = torch.optim.AdamW(dynamic_model.parameters(), lr=cfg.lr)

    dynamic_model.train()
    for epoch in range(cfg.epochs):
        losses = []
        for xl, xm, y in train_loader:
            xl, xm, y = xl.to(device), xm.to(device), y.to(device)
            rhat, ds, z_ccy, m_msg = dynamic_model(xl, xm, edge_index)

            # Use both A_base and A_delta for regularization
            A_combined = dynamic_model.A_base.abs().mean() + dynamic_model.A_delta.abs().mean() * 0.5
            loss = loss_fn(rhat, y, ds, dynamic_model.A_base, cfg)
            loss = loss + cfg.lambda_a_l1 * dynamic_model.A_delta.abs().mean() * 0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    # Evaluate dynamic
    dynamic_model.eval()
    all_preds, all_targets, all_regimes = [], [], []
    with torch.no_grad():
        for xl, xm, y in test_loader:
            xl, xm, y = xl.to(device), xm.to(device), y.to(device)
            rhat, ds, z_ccy, m_msg = dynamic_model(xl, xm, edge_index)

            # Get regime score
            vix_seq = xm[:, :, VIX_IDX]
            regime_score = dynamic_model.regime_gate(vix_seq)

            all_preds.append(ds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_regimes.append(regime_score.cpu().numpy())

    dyn_preds = np.concatenate(all_preds, axis=0)
    dyn_targets = np.concatenate(all_targets, axis=0)
    regime_scores = np.concatenate(all_regimes, axis=0)
    dyn_hit = (np.sign(dyn_preds) == np.sign(dyn_targets)).mean()
    print(f"Dynamic A Hit Rate: {dyn_hit:.1%}")

    # ======================================
    # Analysis
    # ======================================
    print("\n>>> Analyzing regime-dependent performance...")

    median_regime = np.median(regime_scores)
    regime_exp = np.repeat(regime_scores, dyn_preds.shape[1], axis=1)
    correct = (np.sign(dyn_preds) == np.sign(dyn_targets))

    risk_on_mask = regime_exp < median_regime
    risk_off_mask = ~risk_on_mask

    risk_on_hit = correct[risk_on_mask].mean()
    risk_off_hit = correct[risk_off_mask].mean()

    print(f"\n=== Regime-Dependent Performance ===")
    print(f"Risk-ON Hit Rate:  {risk_on_hit:.1%}")
    print(f"Risk-OFF Hit Rate: {risk_off_hit:.1%}")

    # Visualize A matrices
    A_base = dynamic_model.A_base.detach().cpu().numpy()
    A_delta = dynamic_model.A_delta.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmax = max(np.abs(A_base).max(), np.abs(A_base + A_delta).max())

    im1 = axes[0].imshow(A_base, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[0].set_title('A_base (Risk-ON)', fontsize=14)
    axes[0].set_xticks(range(len(MACRO_FEATURES)))
    axes[0].set_xticklabels(MACRO_FEATURES, rotation=45, ha='right')
    axes[0].set_yticks(range(len(CURRENCIES)))
    axes[0].set_yticklabels(CURRENCIES)
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(A_base + A_delta, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[1].set_title('A_base + A_delta (Risk-OFF)', fontsize=14)
    axes[1].set_xticks(range(len(MACRO_FEATURES)))
    axes[1].set_xticklabels(MACRO_FEATURES, rotation=45, ha='right')
    axes[1].set_yticks(range(len(CURRENCIES)))
    axes[1].set_yticklabels(CURRENCIES)
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(A_delta, cmap='RdBu_r', aspect='auto', vmin=-0.2, vmax=0.2)
    axes[2].set_title('A_delta (Regime Shift)', fontsize=14)
    axes[2].set_xticks(range(len(MACRO_FEATURES)))
    axes[2].set_xticklabels(MACRO_FEATURES, rotation=45, ha='right')
    axes[2].set_yticks(range(len(CURRENCIES)))
    axes[2].set_yticklabels(CURRENCIES)
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/A_matrix_regimes.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Top changes
    print("\n=== Top Regime-Dependent Sensitivities ===")
    diff_flat = A_delta.flatten()
    indices = np.argsort(np.abs(diff_flat))[::-1]

    top_changes = []
    for idx in indices[:10]:
        ccy_idx = idx // len(MACRO_FEATURES)
        macro_idx = idx % len(MACRO_FEATURES)
        ccy = CURRENCIES[ccy_idx]
        macro = MACRO_FEATURES[macro_idx]
        delta = A_delta[ccy_idx, macro_idx]
        base = A_base[ccy_idx, macro_idx]
        direction = "↑" if delta > 0 else "↓"
        print(f"  {ccy} ← {macro}: {base:.3f} → {base+delta:.3f} ({direction}{abs(delta):.3f})")
        top_changes.append({'currency': ccy, 'macro': macro, 'risk_on': float(base),
                           'risk_off': float(base + delta), 'delta': float(delta)})

    # Summary
    improvement = (dyn_hit - static_hit) / static_hit * 100

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n1. MODEL COMPARISON:")
    print(f"   • Static A (Baseline): {static_hit:.1%}")
    print(f"   • Dynamic A (Proposed): {dyn_hit:.1%}")
    print(f"   • Improvement: {improvement:+.1f}%")

    print(f"\n2. REGIME-DEPENDENT PERFORMANCE:")
    print(f"   • Risk-ON Hit Rate:  {risk_on_hit:.1%}")
    print(f"   • Risk-OFF Hit Rate: {risk_off_hit:.1%}")

    # Save results
    results = {
        'static_hit_rate': float(static_hit),
        'dynamic_hit_rate': float(dyn_hit),
        'improvement_pct': float(improvement),
        'risk_on_hit': float(risk_on_hit),
        'risk_off_hit': float(risk_off_hit),
        'top_changes': top_changes
    }

    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    run_experiment()
