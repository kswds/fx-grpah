"""
Exp11: Dynamic A Matrix - Regime-Dependent Macro Sensitivity

Key Innovation:
- A matrix changes based on market regime (risk-on vs risk-off)
- Uses VIX as explicit regime indicator
- A(t) = A_base + σ(VIX_t) * A_delta

This extends the original FXStrengthGNN model.
"""

import sys
sys.path.append('/home/gdro/experiment/fx-graph-other')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv
from config import Config
from dataset import load_data, build_features, FXDataset, fully_connected_edge_index, create_dataloaders
from models import FXStrengthGNN

# Load config
cfg = Config()
CURRENCIES = cfg.ccys
MACRO_FEATURES = [f.replace('Global_', '') for f in cfg.global_features]

# VIX is index 1 in MACRO_FEATURES (Gold=0, VIX=1, Oil=2, ...)
VIX_IDX = 1


class FXStrengthGNNDynamicA(nn.Module):
    """
    FX Strength GNN with Dynamic A Matrix

    A(t) = A_base + gate(VIX_t) * A_delta

    The gate is learned from VIX: high VIX -> risk-off regime
    Different currencies respond differently to macros in different regimes.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden = config.hidden
        self.n_ccy = config.n_ccy
        self.macro_dim = config.macro_dim
        self.usd_idx = config.usd_idx

        # Step 1: Temporal encoding (GRU)
        self.local_gru = nn.GRU(config.local_dim, config.hidden, batch_first=True)

        # Step 2: Currency-currency spillover (GAT)
        self.ccy_gnn = GATConv(config.hidden, config.hidden, heads=config.heads, concat=False)

        # Step 3: Macro embedding
        self.macro_embed = nn.Linear(config.macro_dim, config.macro_dim * config.hidden, bias=False)

        # === DYNAMIC A MATRIX ===
        # Base A matrix (risk-on baseline)
        self.A_base = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim))
        nn.init.normal_(self.A_base, mean=0.0, std=0.1)

        # Delta A matrix (regime-dependent adjustment)
        self.A_delta = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim))
        nn.init.normal_(self.A_delta, mean=0.0, std=0.05)

        # Regime gate: VIX -> regime score [0, 1]
        # 0 = risk-on (low VIX), 1 = risk-off (high VIX)
        self.regime_gate = nn.Sequential(
            nn.Linear(config.lookback, config.hidden // 2),
            nn.ReLU(),
            nn.Linear(config.hidden // 2, 1),
            nn.Sigmoid()
        )

        # Step 4: Strength prediction head
        self.head = nn.Linear(config.hidden, 1)

    @property
    def A(self):
        """For compatibility - return A_base"""
        return self.A_base

    def get_regime_score(self, xm):
        """
        Compute regime score from VIX history

        Args:
            xm: [B, L, M] macro features

        Returns:
            regime_score: [B, 1] in [0, 1], higher = risk-off
        """
        # Extract VIX sequence
        vix_seq = xm[:, :, VIX_IDX]  # [B, L]

        # Compute regime score from VIX trajectory
        regime_score = self.regime_gate(vix_seq)  # [B, 1]

        return regime_score

    def get_dynamic_A(self, regime_score):
        """
        Compute dynamic A matrix

        Args:
            regime_score: [B, 1]

        Returns:
            A_dynamic: [B, N, M]
        """
        B = regime_score.shape[0]

        # Expand A matrices for batch
        A_base_exp = self.A_base.unsqueeze(0).expand(B, -1, -1)  # [B, N, M]
        A_delta_exp = self.A_delta.unsqueeze(0).expand(B, -1, -1)  # [B, N, M]

        # regime_score: [B, 1] -> [B, 1, 1]
        regime_exp = regime_score.unsqueeze(-1)

        # Dynamic A
        A_dynamic = A_base_exp + regime_exp * A_delta_exp  # [B, N, M]

        return A_dynamic

    def forward(self, xl, xm, edge_index_single):
        """
        Args:
            xl: Local features [B, L, N, local_dim]
            xm: Macro features [B, L, macro_dim]
            edge_index_single: Edge index [2, E]

        Returns:
            rhat: Predicted FX returns [B, N]
            ds: Currency strengths [B, N]
            regime_score: [B, 1]
            A_dynamic: [B, N, M]
        """
        B, L, N, local_dim = xl.shape

        # 1. Local encoding per currency
        x = xl.permute(0, 2, 1, 3).reshape(B * N, L, local_dim)
        _, h = self.local_gru(x)
        h = h.squeeze(0)  # [B*N, H]

        # 2. Currency-currency spillover via GNN
        E = edge_index_single.size(1)
        edge_b = edge_index_single.repeat(1, B)
        offset = torch.arange(B, device=xl.device).repeat_interleave(E) * N
        edge_b = edge_b + offset.unsqueeze(0)

        z = self.ccy_gnn(h, edge_b)
        z_ccy = z.view(B, N, self.hidden)

        # 3. Macro embedding
        m_t = xm[:, -1, :]  # [B, M]
        u = self.macro_embed(m_t).view(B, self.macro_dim, self.hidden)  # [B, M, H]

        # 4. Get regime score and dynamic A
        regime_score = self.get_regime_score(xm)  # [B, 1]
        A_dynamic = self.get_dynamic_A(regime_score)  # [B, N, M]

        # 5. Heterogeneous transmission via dynamic A matrix
        # A_dynamic: [B, N, M] -> [B, N, M, 1]
        A_exp = A_dynamic.unsqueeze(-1)
        # u: [B, M, H] -> [B, 1, M, H]
        u_exp = u.unsqueeze(1)
        # m_msg: [B, N, H]
        m_msg = (A_exp * u_exp).sum(dim=2)

        # 6. Integration & Prediction
        z_total = z_ccy + m_msg
        ds = self.head(z_total).squeeze(-1)  # [B, N]

        # Zero-mean normalization
        ds = ds - ds.mean(dim=1, keepdim=True)

        # Relative to USD
        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]

        return rhat, ds, regime_score, A_dynamic


def train_model(model, train_loader, edge_index, device, epochs=30, dynamic=False):
    """Train model with MSE loss"""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            xl, xm, y = batch
            xl = xl.to(device)
            xm = xm.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            if dynamic:
                rhat, ds, regime_score, A_dyn = model(xl, xm, edge_index)
            else:
                rhat, ds, z_ccy, m_msg = model(xl, xm, edge_index)

            loss = criterion(ds, y)

            # L1 regularization on A
            if dynamic:
                loss = loss + cfg.lambda_a_l1 * (model.A_base.abs().mean() + model.A_delta.abs().mean())
            else:
                loss = loss + cfg.lambda_a_l1 * model.A.abs().mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")

    return model


def evaluate_model(model, test_loader, edge_index, device, dynamic=False):
    """Evaluate and return predictions"""
    model.eval()

    all_preds = []
    all_targets = []
    all_regime_scores = []

    with torch.no_grad():
        for batch in test_loader:
            xl, xm, y = batch
            xl = xl.to(device)
            xm = xm.to(device)
            y = y.to(device)

            if dynamic:
                rhat, ds, regime_score, A_dyn = model(xl, xm, edge_index)
                all_regime_scores.append(regime_score.cpu().numpy())
            else:
                rhat, ds, z_ccy, m_msg = model(xl, xm, edge_index)

            all_preds.append(ds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    if dynamic:
        regime_scores = np.concatenate(all_regime_scores, axis=0)
        return preds, targets, regime_scores

    return preds, targets


def compute_hit_rate(preds, targets):
    """Compute directional accuracy"""
    correct = (np.sign(preds) == np.sign(targets))
    return correct.mean()


def analyze_regime_performance(preds, targets, regime_scores, output_dir):
    """Analyze performance by regime"""

    regime_flat = regime_scores.flatten()
    median_regime = np.median(regime_flat)

    # Expand to match predictions
    regime_exp = np.repeat(regime_scores, preds.shape[1], axis=1)

    correct = (np.sign(preds) == np.sign(targets))

    risk_on_mask = regime_exp < median_regime
    risk_off_mask = ~risk_on_mask

    risk_on_hit = correct[risk_on_mask].mean() if risk_on_mask.sum() > 0 else 0
    risk_off_hit = correct[risk_off_mask].mean() if risk_off_mask.sum() > 0 else 0

    print(f"\n=== Regime-Dependent Performance ===")
    print(f"Median regime score: {median_regime:.3f}")
    print(f"Risk-ON (regime < {median_regime:.3f}):  {risk_on_hit:.1%}")
    print(f"Risk-OFF (regime >= {median_regime:.3f}): {risk_off_hit:.1%}")

    return {
        'risk_on_hit': float(risk_on_hit),
        'risk_off_hit': float(risk_off_hit),
        'median_regime': float(median_regime)
    }


def visualize_A_matrices(model, output_dir):
    """Visualize A_base, A_delta, and their combination"""

    A_base = model.A_base.detach().cpu().numpy()
    A_delta = model.A_delta.detach().cpu().numpy()

    A_risk_on = A_base  # regime_score = 0
    A_risk_off = A_base + A_delta  # regime_score = 1

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    vmax = max(np.abs(A_risk_on).max(), np.abs(A_risk_off).max())

    # Risk-ON
    im1 = axes[0].imshow(A_risk_on, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[0].set_title('A Matrix (Risk-ON, Low VIX)', fontsize=14)
    axes[0].set_xlabel('Macro Factors')
    axes[0].set_ylabel('Currencies')
    axes[0].set_xticks(range(len(MACRO_FEATURES)))
    axes[0].set_xticklabels(MACRO_FEATURES, rotation=45, ha='right')
    axes[0].set_yticks(range(len(CURRENCIES)))
    axes[0].set_yticklabels(CURRENCIES)
    plt.colorbar(im1, ax=axes[0])

    # Risk-OFF
    im2 = axes[1].imshow(A_risk_off, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[1].set_title('A Matrix (Risk-OFF, High VIX)', fontsize=14)
    axes[1].set_xlabel('Macro Factors')
    axes[1].set_ylabel('Currencies')
    axes[1].set_xticks(range(len(MACRO_FEATURES)))
    axes[1].set_xticklabels(MACRO_FEATURES, rotation=45, ha='right')
    axes[1].set_yticks(range(len(CURRENCIES)))
    axes[1].set_yticklabels(CURRENCIES)
    plt.colorbar(im2, ax=axes[1])

    # Difference
    im3 = axes[2].imshow(A_delta, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
    axes[2].set_title('A_delta (Regime Shift)', fontsize=14)
    axes[2].set_xlabel('Macro Factors')
    axes[2].set_ylabel('Currencies')
    axes[2].set_xticks(range(len(MACRO_FEATURES)))
    axes[2].set_xticklabels(MACRO_FEATURES, rotation=45, ha='right')
    axes[2].set_yticks(range(len(CURRENCIES)))
    axes[2].set_yticklabels(CURRENCIES)
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/A_matrix_regimes.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Find top changes
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
            'currency': ccy,
            'macro': macro,
            'risk_on': float(base),
            'risk_off': float(base + delta),
            'delta': float(delta)
        })

    return top_changes


def visualize_regime_time(regime_scores, output_dir):
    """Visualize learned regime over time"""

    regime_flat = regime_scores.flatten()

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(regime_flat, alpha=0.7, linewidth=0.5, label='Raw')

    # Rolling average
    window = 20
    regime_rolling = pd.Series(regime_flat).rolling(window=window).mean()
    ax.plot(regime_rolling, color='blue', linewidth=2, label=f'{window}-day MA')

    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax.fill_between(range(len(regime_flat)), 0, 1,
                    where=(regime_flat > 0.5), alpha=0.2, color='red', label='Risk-OFF')
    ax.fill_between(range(len(regime_flat)), 0, 1,
                    where=(regime_flat <= 0.5), alpha=0.2, color='green', label='Risk-ON')

    ax.set_ylabel('Regime Score (0=Risk-ON, 1=Risk-OFF)')
    ax.set_xlabel('Test Sample Index')
    ax.set_title('Learned Market Regime from VIX')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/regime_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("EXP11: DYNAMIC A MATRIX - REGIME-DEPENDENT MACRO SENSITIVITY")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = 'exp11_dynamic_A'

    # Load data using existing infrastructure
    print("\n>>> Loading data...")
    train_loader, test_loader = create_dataloaders(cfg)

    edge_index = fully_connected_edge_index(cfg.n_ccy).to(device)

    # ======================================
    # Train Static A Model (Baseline)
    # ======================================
    print("\n>>> Training Static A Model (Baseline)...")
    static_model = FXStrengthGNN(cfg).to(device)
    static_model = train_model(static_model, train_loader, edge_index, device,
                               epochs=cfg.epochs, dynamic=False)

    static_preds, static_targets = evaluate_model(static_model, test_loader, edge_index, device, dynamic=False)
    static_hit = compute_hit_rate(static_preds, static_targets)
    print(f"\nStatic A Hit Rate: {static_hit:.1%}")

    # ======================================
    # Train Dynamic A Model
    # ======================================
    print("\n>>> Training Dynamic A Model (Regime-Dependent)...")
    dynamic_model = FXStrengthGNNDynamicA(cfg).to(device)
    dynamic_model = train_model(dynamic_model, train_loader, edge_index, device,
                                epochs=cfg.epochs, dynamic=True)

    dyn_preds, dyn_targets, regime_scores = evaluate_model(
        dynamic_model, test_loader, edge_index, device, dynamic=True
    )
    dyn_hit = compute_hit_rate(dyn_preds, dyn_targets)
    print(f"\nDynamic A Hit Rate: {dyn_hit:.1%}")

    # ======================================
    # Analysis
    # ======================================
    print("\n>>> Analyzing Results...")

    # Regime performance
    regime_results = analyze_regime_performance(dyn_preds, dyn_targets, regime_scores, output_dir)

    # Visualize A matrices
    top_changes = visualize_A_matrices(dynamic_model, output_dir)

    # Visualize regime
    visualize_regime_time(regime_scores, output_dir)

    # ======================================
    # Summary
    # ======================================
    improvement = (dyn_hit - static_hit) / static_hit * 100

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n1. MODEL COMPARISON:")
    print(f"   • Static A (Baseline): {static_hit:.1%}")
    print(f"   • Dynamic A (Proposed): {dyn_hit:.1%}")
    print(f"   • Improvement: {improvement:+.1f}%")

    print(f"\n2. REGIME-DEPENDENT PERFORMANCE:")
    print(f"   • Risk-ON Hit Rate:  {regime_results['risk_on_hit']:.1%}")
    print(f"   • Risk-OFF Hit Rate: {regime_results['risk_off_hit']:.1%}")

    print(f"\n3. KEY FINDINGS:")
    for i, change in enumerate(top_changes[:5]):
        direction = "↑ stronger" if change['delta'] > 0 else "↓ weaker"
        print(f"   {i+1}. {change['currency']} ← {change['macro']}: {direction} in Risk-OFF ({change['delta']:+.3f})")

    # Save results
    results = {
        'static_hit_rate': float(static_hit),
        'dynamic_hit_rate': float(dyn_hit),
        'improvement_pct': float(improvement),
        'regime_analysis': regime_results,
        'top_regime_changes': top_changes
    }

    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"  - A_matrix_regimes.png")
    print(f"  - regime_over_time.png")
    print(f"  - results.json")

    return results


if __name__ == '__main__':
    main()
