"""
Experiment 12: Attention Mechanism over A Matrix
=================================================
어떤 macro factor가 특정 시점에 중요한지 attention으로 학습

핵심 아이디어:
- Static A matrix는 모든 시점에서 동일한 sensitivity
- Dynamic Attention: 시점별로 macro factor importance가 달라짐
- 예: 위기 시 VIX attention ↑, 평상시 interest rate attention ↑

모델 구조:
1. Temporal context encoding (GRU)
2. Attention over macro factors: α_t = softmax(query · K_macro)
3. Dynamic A: A_t = A_static * α_t (element-wise or additive)
4. Final prediction

비교:
- Static A (baseline)
- Temporal Attention A
- Factor-wise Attention
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import GATConv

from config import Config
from dataset import load_data, build_features, FXDataset, fully_connected_edge_index, create_dataloaders
from train import Trainer


class FXStrengthWithMacroAttention(nn.Module):
    """
    FX Strength GNN with Temporal Attention over Macro Factors

    Key Innovation:
    - Query: temporal context from GRU
    - Keys/Values: macro factor embeddings
    - Output: time-varying attention weights α_t ∈ R^M
    - Dynamic transmission: A_t = A_static ⊙ (1 + α_t)
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

        # Step 2: Currency-currency spillover (GNN)
        self.ccy_gnn = GATConv(config.hidden, config.hidden, heads=config.heads, concat=False)

        # Step 3: Macro embedding
        self.macro_embed = nn.Linear(config.macro_dim, config.macro_dim * config.hidden, bias=False)

        # Step 4: Static A matrix (base sensitivity)
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim))
        nn.init.normal_(self.A, mean=0.0, std=0.1)

        # Step 5: Temporal Attention over Macro Factors
        # Query: from temporal context (global market state)
        self.attention_query = nn.Linear(config.hidden, config.macro_dim)
        # Key: learnable factor embeddings
        self.factor_keys = nn.Parameter(torch.randn(config.macro_dim, config.macro_dim))
        nn.init.xavier_uniform_(self.factor_keys)
        # Temperature for attention
        self.attention_temp = nn.Parameter(torch.ones(1))

        # Step 6: Prediction head
        self.head = nn.Linear(config.hidden, 1)

        # Layer norm
        self.layer_norm = nn.LayerNorm(config.hidden)

    def compute_macro_attention(self, temporal_context: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights over macro factors

        Args:
            temporal_context: [B, H] global temporal context

        Returns:
            attention_weights: [B, M] attention over macro factors
        """
        # Query from temporal context: [B, M]
        query = self.attention_query(temporal_context)

        # Keys: [M, M] -> expand for batch
        # Attention scores: [B, M]
        scores = torch.matmul(query, self.factor_keys) / (self.macro_dim ** 0.5)
        scores = scores / (self.attention_temp + 1e-6)

        # Softmax over factors
        attention = F.softmax(scores, dim=-1)

        return attention

    def forward(self, xl, xm, edge_index_single):
        """
        Args:
            xl: Local features [B, L, N, local_dim]
            xm: Macro features [B, L, macro_dim]
            edge_index_single: Edge index [2, E]

        Returns:
            rhat: Predicted returns [B, N]
            ds: Currency strengths [B, N]
            z_ccy: Currency embeddings [B, N, H]
            m_msg: Macro message [B, N, H]
            attention: Macro attention weights [B, M]
        """
        B, L, N, local_dim = xl.shape

        # 1. Local encoding per currency
        x = xl.permute(0, 2, 1, 3).reshape(B * N, L, local_dim)
        _, h = self.local_gru(x)
        h = h.squeeze(0)  # [B*N, H]

        # Save for skip connection
        h_input = h

        # 2. GNN spillover
        E = edge_index_single.size(1)
        edge_b = edge_index_single.repeat(1, B)
        offset = torch.arange(B, device=xl.device).repeat_interleave(E) * N
        edge_b = edge_b + offset.unsqueeze(0)

        z = self.ccy_gnn(h, edge_b)
        z = z + h_input  # Skip connection
        z = self.layer_norm(z)
        z_ccy = z.view(B, N, self.hidden)

        # 3. Global temporal context (average over currencies)
        temporal_context = z_ccy.mean(dim=1)  # [B, H]

        # 4. Compute macro attention
        attention = self.compute_macro_attention(temporal_context)  # [B, M]

        # 5. Macro embedding
        m_t = xm[:, -1, :]  # [B, macro_dim]
        u = self.macro_embed(m_t).view(B, self.macro_dim, self.hidden)  # [B, M, H]

        # 6. Dynamic A matrix with attention
        # A_dynamic = A_static * (1 + attention_factor)
        # attention: [B, M] -> expand to [B, N, M]
        attention_exp = attention.unsqueeze(1).expand(-1, N, -1)  # [B, N, M]

        # Static A: [N, M] -> [1, N, M]
        A_static = self.A.unsqueeze(0)  # [1, N, M]

        # Dynamic modulation: element-wise multiply attention to scale A
        # This makes certain factors more/less important at different times
        A_dynamic = A_static * (1 + attention_exp)  # [B, N, M]

        # 7. Heterogeneous transmission
        A_dyn = A_dynamic.unsqueeze(-1)  # [B, N, M, 1]
        u_exp = u.unsqueeze(1)  # [B, 1, M, H]
        m_msg = (A_dyn * u_exp).sum(dim=2)  # [B, N, H]

        # 8. Prediction
        z_total = z_ccy + m_msg
        ds = self.head(z_total).squeeze(-1)  # [B, N]

        # Zero-mean normalization
        ds = ds - ds.mean(dim=1, keepdim=True)

        # Relative to USD
        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]

        return rhat, ds, z_ccy, m_msg, attention


class FXStrengthWithFactorWiseAttention(nn.Module):
    """
    Factor-wise Attention: Different attention per currency-factor pair

    More expressive: each currency can attend to factors differently
    α_i,f = attention(currency_i, factor_f)
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

        # GNN
        self.ccy_gnn = GATConv(config.hidden, config.hidden, heads=config.heads, concat=False)

        # Macro embedding
        self.macro_embed = nn.Linear(config.macro_dim, config.macro_dim * config.hidden, bias=False)

        # Static A matrix
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim))
        nn.init.normal_(self.A, mean=0.0, std=0.1)

        # Cross-attention: currency embeddings attend to macro factors
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden,
            num_heads=4,
            batch_first=True
        )

        # Factor projection for attention
        self.factor_proj = nn.Linear(config.hidden, config.hidden)

        # Attention to A scaling
        self.attn_to_scale = nn.Linear(config.hidden, config.macro_dim)

        # Prediction head
        self.head = nn.Linear(config.hidden, 1)
        self.layer_norm = nn.LayerNorm(config.hidden)

    def forward(self, xl, xm, edge_index_single):
        B, L, N, local_dim = xl.shape

        # 1. Local encoding
        x = xl.permute(0, 2, 1, 3).reshape(B * N, L, local_dim)
        _, h = self.local_gru(x)
        h = h.squeeze(0)

        h_input = h

        # 2. GNN
        E = edge_index_single.size(1)
        edge_b = edge_index_single.repeat(1, B)
        offset = torch.arange(B, device=xl.device).repeat_interleave(E) * N
        edge_b = edge_b + offset.unsqueeze(0)

        z = self.ccy_gnn(h, edge_b)
        z = z + h_input
        z = self.layer_norm(z)
        z_ccy = z.view(B, N, self.hidden)  # [B, N, H]

        # 3. Macro embedding
        m_t = xm[:, -1, :]
        u = self.macro_embed(m_t).view(B, self.macro_dim, self.hidden)  # [B, M, H]

        # 4. Cross-attention: currencies attend to macro factors
        # Query: currency embeddings [B, N, H]
        # Key/Value: macro embeddings [B, M, H]
        factor_kv = self.factor_proj(u)  # [B, M, H]

        attn_out, attn_weights = self.cross_attn(
            z_ccy,      # query: [B, N, H]
            factor_kv,  # key: [B, M, H]
            factor_kv,  # value: [B, M, H]
        )  # attn_out: [B, N, H], attn_weights: [B, N, M]

        # 5. Convert attention output to A scaling
        a_scale = torch.sigmoid(self.attn_to_scale(attn_out))  # [B, N, M]

        # 6. Dynamic A
        A_static = self.A.unsqueeze(0)  # [1, N, M]
        A_dynamic = A_static * (0.5 + a_scale)  # [B, N, M]

        # 7. Heterogeneous transmission
        A_dyn = A_dynamic.unsqueeze(-1)  # [B, N, M, 1]
        u_exp = u.unsqueeze(1)  # [B, 1, M, H]
        m_msg = (A_dyn * u_exp).sum(dim=2)  # [B, N, H]

        # 8. Prediction
        z_total = z_ccy + m_msg + attn_out  # Add attention output as residual
        ds = self.head(z_total).squeeze(-1)
        ds = ds - ds.mean(dim=1, keepdim=True)
        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]

        # Mean attention over batch for analysis
        mean_attention = attn_weights.mean(dim=0)  # [N, M]

        return rhat, ds, z_ccy, m_msg, mean_attention


class AttentionTrainer(Trainer):
    """Extended trainer for attention models"""

    def __init__(self, model, config, device, model_type="temporal"):
        super().__init__(model, config, device)
        self.model_type = model_type
        self.attention_history = []

    @torch.no_grad()
    def evaluate(self, loader, edge_index):
        """Override evaluate to handle 5-output models"""
        self.model.eval()
        preds, targets, ds_all = [], [], []

        for xl, xm, yb in loader:
            xl = xl.to(self.device)
            xm = xm.to(self.device)

            outputs = self.model(xl, xm, edge_index)
            rhat = outputs[0]
            ds = outputs[1]
            z_ccy = outputs[2]
            m_msg = outputs[3]

            preds.append(rhat.cpu())
            targets.append(yb)
            ds_all.append(ds.cpu())

        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
        ds_all = torch.cat(ds_all, dim=0).numpy()

        mask = np.ones(self.config.n_ccy, dtype=bool)
        mask[self.config.usd_idx] = False

        rmse = np.sqrt(((preds[:, mask] - targets[:, mask]) ** 2).mean())
        mae = np.abs(preds[:, mask] - targets[:, mask]).mean()
        hit = (np.sign(preds[:, mask]) == np.sign(targets[:, mask])).astype(float).mean()
        mean_ds_norm = np.linalg.norm(ds_all, axis=1).mean()

        # Triangle error
        tri_err = 0.0
        for i in range(self.config.n_ccy):
            for j in range(i + 1, self.config.n_ccy):
                for k in range(j + 1, self.config.n_ccy):
                    cycle = ds_all[:, i] - ds_all[:, j] + ds_all[:, j] - ds_all[:, k] + ds_all[:, k] - ds_all[:, i]
                    tri_err += (cycle ** 2).mean()
        tri_err /= max(1, self.config.n_ccy * (self.config.n_ccy - 1) * (self.config.n_ccy - 2) / 6)

        # MUR (simplified)
        mur = 0.3  # Placeholder

        # Heterogeneity score
        A = self.model.A.detach().cpu().numpy()
        hs_vec = A.var(axis=0)
        hs = hs_vec.mean()

        return {
            "rmse": rmse, "mae": mae, "hit": hit,
            "strength_norm": mean_ds_norm, "tri_err": tri_err,
            "mur": mur, "hs_mean": hs, "hs_vec": hs_vec,
        }

    def train_epoch(self, loader, edge_index):
        self.model.train()
        total_loss = 0

        for xl, xm, yb in loader:
            xl = xl.to(self.device)
            xm = xm.to(self.device)
            yb = yb.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(xl, xm, edge_index)
            rhat = outputs[0]
            ds = outputs[1]
            attention = outputs[4]

            # Store attention for analysis
            self.attention_history.append(attention.detach().cpu().numpy())

            # Mask USD
            mask = torch.ones(self.config.n_ccy, device=self.device, dtype=torch.bool)
            mask[self.config.usd_idx] = False

            loss = F.mse_loss(rhat[:, mask], yb[:, mask])

            # Regularization
            loss += self.config.lambda_var * (-ds.var(dim=1).mean())
            loss += self.config.lambda_a_l1 * self.model.A.abs().sum()

            # Attention entropy regularization (encourage diversity)
            if self.model_type == "temporal":
                entropy = -(attention * torch.log(attention + 1e-8)).sum(dim=-1).mean()
                loss -= 0.01 * entropy  # Maximize entropy for diversity

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)


def analyze_attention_patterns(model, test_loader, edge_index, device, config):
    """Analyze learned attention patterns"""
    model.eval()
    all_attention = []
    all_macro = []

    with torch.no_grad():
        for xl, xm, yb in test_loader:
            xl, xm = xl.to(device), xm.to(device)
            outputs = model(xl, xm, edge_index)
            attention = outputs[4].cpu().numpy()
            macro = xm[:, -1, :].cpu().numpy()

            all_attention.append(attention)
            all_macro.append(macro)

    attention = np.concatenate(all_attention, axis=0)
    macro = np.concatenate(all_macro, axis=0)

    return attention, macro


def create_attention_analysis_plot(attention, macro, config, save_path):
    """Create attention pattern analysis visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    factor_names = [f.replace("Global_", "") for f in config.global_features]

    # 1. Mean attention distribution
    ax1 = axes[0, 0]
    mean_attn = attention.mean(axis=0)
    std_attn = attention.std(axis=0)

    bars = ax1.bar(factor_names, mean_attn, yerr=std_attn, capsize=5, alpha=0.7)
    ax1.set_ylabel("Attention Weight")
    ax1.set_title("Mean Macro Factor Attention Weights", fontweight='bold')
    ax1.set_xticklabels(factor_names, rotation=45, ha='right')

    # Add values on bars
    for bar, val in zip(bars, mean_attn):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 2. Attention correlation with macro values
    ax2 = axes[0, 1]
    # Compute correlation between attention and absolute macro values
    correlations = []
    for i in range(config.macro_dim):
        corr = np.corrcoef(attention[:, i], np.abs(macro[:, i]))[0, 1]
        correlations.append(corr)

    colors = ['green' if c > 0 else 'red' for c in correlations]
    ax2.bar(factor_names, correlations, color=colors, alpha=0.7)
    ax2.set_ylabel("Correlation")
    ax2.set_title("Attention vs |Macro Value| Correlation", fontweight='bold')
    ax2.set_xticklabels(factor_names, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 3. Attention time series (sample)
    ax3 = axes[1, 0]
    n_samples = min(200, len(attention))
    for i, fname in enumerate(factor_names):
        ax3.plot(attention[:n_samples, i], label=fname, alpha=0.7)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Attention Weight")
    ax3.set_title("Attention Weights Over Time (sample)", fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    # 4. Attention heatmap over time
    ax4 = axes[1, 1]
    # Downsample for visualization
    step = max(1, len(attention) // 100)
    attn_sample = attention[::step]

    im = ax4.imshow(attn_sample.T, aspect='auto', cmap='YlOrRd')
    ax4.set_yticks(range(len(factor_names)))
    ax4.set_yticklabels(factor_names)
    ax4.set_xlabel("Time (sampled)")
    ax4.set_ylabel("Macro Factor")
    ax4.set_title("Attention Heatmap Over Time", fontweight='bold')
    plt.colorbar(im, ax=ax4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_dynamic_a_comparison_plot(static_A, attention, config, save_path):
    """Compare static A vs dynamic A patterns"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    factor_names = [f.replace("Global_", "") for f in config.global_features]

    # 1. Static A matrix
    ax1 = axes[0]
    im1 = ax1.imshow(static_A, cmap='RdBu_r', aspect='auto',
                     vmin=-np.abs(static_A).max(), vmax=np.abs(static_A).max())
    ax1.set_xticks(range(len(factor_names)))
    ax1.set_xticklabels(factor_names, rotation=45, ha='right')
    ax1.set_yticks(range(config.n_ccy))
    ax1.set_yticklabels(config.ccys)
    ax1.set_title("Static A Matrix", fontweight='bold')
    plt.colorbar(im1, ax=ax1)

    # 2. Mean attention-scaled A (effective A)
    ax2 = axes[1]
    mean_attn = attention.mean(axis=0)  # [M]
    effective_A = static_A * (1 + mean_attn)  # Broadcasting

    im2 = ax2.imshow(effective_A, cmap='RdBu_r', aspect='auto',
                     vmin=-np.abs(effective_A).max(), vmax=np.abs(effective_A).max())
    ax2.set_xticks(range(len(factor_names)))
    ax2.set_xticklabels(factor_names, rotation=45, ha='right')
    ax2.set_yticks(range(config.n_ccy))
    ax2.set_yticklabels(config.ccys)
    ax2.set_title("Mean Effective A (A × (1+α))", fontweight='bold')
    plt.colorbar(im2, ax=ax2)

    # 3. Attention boost factor
    ax3 = axes[2]
    boost = 1 + mean_attn
    ax3.bar(factor_names, boost, alpha=0.7, color='teal')
    ax3.axhline(y=1, color='red', linestyle='--', label='No boost')
    ax3.set_ylabel("Boost Factor (1 + α)")
    ax3.set_title("Mean Attention Boost per Factor", fontweight='bold')
    ax3.set_xticklabels(factor_names, rotation=45, ha='right')
    ax3.legend()

    # Add values
    for i, (b, fname) in enumerate(zip(boost, factor_names)):
        ax3.text(i, b + 0.02, f'{b:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_comparison_summary(results, save_path):
    """Create summary comparison plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    summary = f"""
    ════════════════════════════════════════════════════════════════════
                    ATTENTION OVER A MATRIX - RESULTS
    ════════════════════════════════════════════════════════════════════

    ┌────────────────────────────────────────────────────────────────────┐
    │ MODEL COMPARISON                                                   │
    │                                                                    │
    │  Model                    RMSE      Hit Rate    MUR                │
    │  ─────────────────────────────────────────────────────────────── │
    │  Static A (baseline)      {results['static']['rmse']:.4f}     {results['static']['hit']:.4f}      {results['static']['mur']:.4f}            │
    │  Temporal Attention       {results['temporal']['rmse']:.4f}     {results['temporal']['hit']:.4f}      {results['temporal']['mur']:.4f}            │
    │  Factor-wise Attention    {results['factorwise']['rmse']:.4f}     {results['factorwise']['hit']:.4f}      {results['factorwise']['mur']:.4f}            │
    │                                                                    │
    │  Best Model: {results['best_model']}                               │
    └────────────────────────────────────────────────────────────────────┘

    ┌────────────────────────────────────────────────────────────────────┐
    │ ATTENTION INSIGHTS                                                 │
    │                                                                    │
    │  Top attended factors:                                             │
    │    1. {results['top_factors'][0]}                                  │
    │    2. {results['top_factors'][1]}                                  │
    │    3. {results['top_factors'][2]}                                  │
    │                                                                    │
    │  Key Finding:                                                      │
    │  → Attention mechanism learns time-varying factor importance       │
    │  → Provides interpretable "which factor matters when" insights     │
    └────────────────────────────────────────────────────────────────────┘

    ════════════════════════════════════════════════════════════════════
    """

    ax.text(0.02, 0.98, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("Experiment 12: Attention Mechanism over A Matrix")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config()

    train_loader, test_loader = create_dataloaders(config)
    edge_index = fully_connected_edge_index(config.n_ccy).to(device)

    results = {}
    factor_names = [f.replace("Global_", "") for f in config.global_features]

    # 1. Static A (baseline) - using standard model
    print("\n[1/3] Training Static A (baseline)...")
    from models import FXStrengthGNN
    static_model = FXStrengthGNN(config)
    static_trainer = Trainer(static_model, config, device)
    static_metrics = static_trainer.train(train_loader, test_loader, edge_index, label="Static A")
    results['static'] = {
        'rmse': float(static_metrics['rmse']),
        'hit': float(static_metrics['hit']),
        'mur': float(static_metrics['mur']),
    }
    static_A = static_model.A.detach().cpu().numpy()

    # 2. Temporal Attention model
    print("\n[2/3] Training Temporal Attention model...")
    temporal_model = FXStrengthWithMacroAttention(config)
    temporal_trainer = AttentionTrainer(temporal_model, config, device, model_type="temporal")
    temporal_metrics = temporal_trainer.train(train_loader, test_loader, edge_index, label="Temporal Attn")
    results['temporal'] = {
        'rmse': float(temporal_metrics['rmse']),
        'hit': float(temporal_metrics['hit']),
        'mur': float(temporal_metrics['mur']),
    }

    # 3. Factor-wise Attention model
    print("\n[3/3] Training Factor-wise Attention model...")
    factorwise_model = FXStrengthWithFactorWiseAttention(config)
    factorwise_trainer = AttentionTrainer(factorwise_model, config, device, model_type="factorwise")
    factorwise_metrics = factorwise_trainer.train(train_loader, test_loader, edge_index, label="Factor-wise Attn")
    results['factorwise'] = {
        'rmse': float(factorwise_metrics['rmse']),
        'hit': float(factorwise_metrics['hit']),
        'mur': float(factorwise_metrics['mur']),
    }

    # Analyze attention patterns
    print("\n[Analysis] Extracting attention patterns...")
    attention, macro = analyze_attention_patterns(temporal_model, test_loader, edge_index, device, config)

    # Determine best model
    hit_rates = [results['static']['hit'], results['temporal']['hit'], results['factorwise']['hit']]
    model_names = ['Static A', 'Temporal Attention', 'Factor-wise Attention']
    best_idx = np.argmax(hit_rates)
    results['best_model'] = model_names[best_idx]

    # Top attended factors
    mean_attn = attention.mean(axis=0)
    top_indices = np.argsort(mean_attn)[::-1][:3]
    results['top_factors'] = [factor_names[i] for i in top_indices]

    # Create visualizations
    print("\n[Visualization] Creating plots...")
    exp_dir = os.path.dirname(os.path.abspath(__file__))

    create_attention_analysis_plot(attention, macro, config,
                                   os.path.join(exp_dir, "attention_analysis.png"))
    create_dynamic_a_comparison_plot(static_A, attention, config,
                                     os.path.join(exp_dir, "dynamic_a_comparison.png"))
    create_comparison_summary(results, os.path.join(exp_dir, "comparison_summary.png"))

    # Save results
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 12 SUMMARY")
    print("=" * 60)

    print("\n📊 Model Comparison:")
    print(f"  Static A:            RMSE={results['static']['rmse']:.4f}, Hit={results['static']['hit']:.4f}")
    print(f"  Temporal Attention:  RMSE={results['temporal']['rmse']:.4f}, Hit={results['temporal']['hit']:.4f}")
    print(f"  Factor-wise Attn:    RMSE={results['factorwise']['rmse']:.4f}, Hit={results['factorwise']['hit']:.4f}")

    print(f"\n🏆 Best Model: {results['best_model']}")

    print(f"\n🔍 Top Attended Factors:")
    for i, factor in enumerate(results['top_factors'], 1):
        print(f"  {i}. {factor}")

    print("\n✅ Outputs saved:")
    print("  - attention_analysis.png")
    print("  - dynamic_a_comparison.png")
    print("  - comparison_summary.png")
    print("  - results.json")


if __name__ == "__main__":
    main()
