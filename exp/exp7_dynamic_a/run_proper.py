"""
Exp11: Dynamic A Matrix - Using exact same pipeline as main.py
"""

import sys
sys.path.append('/home/gdro/experiment/fx-graph-other')

import random
import numpy as np
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv

from config import Config
from dataset import create_dataloaders, fully_connected_edge_index
from models import FXStrengthGNN
from train import Trainer, loss_fn

cfg = Config()
CURRENCIES = cfg.ccys
MACRO_FEATURES = [f.replace('Global_', '') for f in cfg.global_features]
VIX_IDX = 1  # VIX index in macro features


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FXStrengthGNNDynamicA(nn.Module):
    """
    FX Strength GNN with Dynamic A Matrix

    Compatible with existing Trainer class.
    Returns same outputs as FXStrengthGNN: (rhat, ds, z_ccy, m_msg)
    """

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

        # Dynamic A matrices
        self.A_base = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim))
        self.A_delta = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim))
        nn.init.normal_(self.A_base, mean=0.0, std=0.1)
        nn.init.normal_(self.A_delta, mean=0.0, std=0.05)

        # Regime gate
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
        """For compatibility with Trainer - returns A_base"""
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
        regime_score = self.regime_gate(vix_seq)  # [B, 1]

        # Store for analysis
        self._last_regime_score = regime_score.detach()

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

        # Return same format as FXStrengthGNN
        return rhat, ds, z_ccy, m_msg


def main():
    print("=" * 70)
    print("EXP11: DYNAMIC A MATRIX - PROPER COMPARISON")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    output_dir = 'exp11_dynamic_A'
    edge_index = fully_connected_edge_index(cfg.n_ccy).to(device)

    # ======================================
    # Train Static A (Baseline) - same as main.py
    # ======================================
    print("\n>>> Training Static A Model (Baseline)...")
    set_seed(42)
    train_loader, test_loader = create_dataloaders(cfg, macro_mode="real")
    static_model = FXStrengthGNN(cfg)
    static_trainer = Trainer(static_model, cfg, device)
    static_res = static_trainer.train(train_loader, test_loader, edge_index, label="STATIC A")

    # ======================================
    # Train Dynamic A Model
    # ======================================
    print("\n>>> Training Dynamic A Model...")
    set_seed(42)
    train_loader, test_loader = create_dataloaders(cfg, macro_mode="real")
    dynamic_model = FXStrengthGNNDynamicA(cfg)
    dynamic_trainer = Trainer(dynamic_model, cfg, device)
    dynamic_res = dynamic_trainer.train(train_loader, test_loader, edge_index, label="DYNAMIC A")

    # ======================================
    # Regime Analysis
    # ======================================
    print("\n>>> Analyzing regime-dependent performance...")

    dynamic_model.eval()
    all_regime_scores = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xl, xm, y in test_loader:
            xl, xm, y = xl.to(device), xm.to(device), y.to(device)
            rhat, ds, z_ccy, m_msg = dynamic_model(xl, xm, edge_index)
            all_regime_scores.append(dynamic_model._last_regime_score.cpu().numpy())
            all_preds.append(ds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    regime_scores = np.concatenate(all_regime_scores, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Regime-dependent hit rate
    median_regime = np.median(regime_scores)
    regime_exp = np.repeat(regime_scores, preds.shape[1], axis=1)
    correct = (np.sign(preds) == np.sign(targets))

    # Exclude USD (index 0)
    mask = np.ones(preds.shape[1], dtype=bool)
    mask[cfg.usd_idx] = False
    correct_masked = correct[:, mask]
    regime_exp_masked = regime_exp[:, mask]

    risk_on_mask = regime_exp_masked < median_regime
    risk_off_mask = ~risk_on_mask

    risk_on_hit = correct_masked[risk_on_mask].mean()
    risk_off_hit = correct_masked[risk_off_mask].mean()

    print(f"\n=== Regime-Dependent Performance ===")
    print(f"Median regime: {median_regime:.3f}")
    print(f"Risk-ON Hit:   {risk_on_hit:.1%}")
    print(f"Risk-OFF Hit:  {risk_off_hit:.1%}")

    # ======================================
    # Visualize A matrices
    # ======================================
    A_base = dynamic_model.A_base.detach().cpu().numpy()
    A_delta = dynamic_model.A_delta.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmax = max(np.abs(A_base).max(), np.abs(A_base + A_delta).max(), 0.3)

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
        top_changes.append({
            'currency': ccy, 'macro': macro,
            'risk_on': float(base), 'risk_off': float(base + delta),
            'delta': float(delta)
        })

    # ======================================
    # Summary
    # ======================================
    improvement = (dynamic_res['hit'] - static_res['hit']) / static_res['hit'] * 100

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n1. MODEL COMPARISON:")
    print(f"   • Static A:  {static_res['hit']:.1%}")
    print(f"   • Dynamic A: {dynamic_res['hit']:.1%}")
    print(f"   • Improvement: {improvement:+.1f}%")

    print(f"\n2. REGIME PERFORMANCE:")
    print(f"   • Risk-ON:  {risk_on_hit:.1%}")
    print(f"   • Risk-OFF: {risk_off_hit:.1%}")

    # Save results
    results = {
        'static_hit_rate': float(static_res['hit']),
        'dynamic_hit_rate': float(dynamic_res['hit']),
        'improvement_pct': float(improvement),
        'risk_on_hit': float(risk_on_hit),
        'risk_off_hit': float(risk_off_hit),
        'top_changes': top_changes
    }

    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()
