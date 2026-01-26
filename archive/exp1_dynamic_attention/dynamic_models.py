"""
Exp1: Dynamic Cross-Attention for Macro-Currency Transmission
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from config import Config


class DynamicMacroCurrencyAttention(nn.Module):
    """
    Cross-attention module for dynamic macro-to-currency transmission

    Instead of fixed A matrix, learns dynamic attention weights:
    A_t[i, f] = softmax(Q_i @ K_f / sqrt(d))

    where Q = currency embedding, K = macro embedding
    """
    def __init__(self, hidden_dim: int, macro_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.macro_dim = macro_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        # Query projection (currency -> query)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)

        # Key projection (macro -> key)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)

        # Value projection (macro -> value)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, currency_embed, macro_embed):
        """
        Args:
            currency_embed: [B, N, H] - currency embeddings (from GRU/GNN)
            macro_embed: [B, M, H] - macro factor embeddings

        Returns:
            m_msg: [B, N, H] - macro message to each currency
            attn_weights: [B, n_heads, N, M] - attention weights (dynamic A)
        """
        B, N, H = currency_embed.shape
        M = macro_embed.size(1)

        # Project to Q, K, V
        Q = self.W_q(currency_embed)  # [B, N, H]
        K = self.W_k(macro_embed)      # [B, M, H]
        V = self.W_v(macro_embed)      # [B, M, H]

        # Reshape for multi-head attention
        Q = Q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # [B, heads, N, head_dim]
        K = K.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)  # [B, heads, M, head_dim]
        V = V.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)  # [B, heads, M, head_dim]

        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, heads, N, M]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # [B, heads, N, head_dim]

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, H)  # [B, N, H]
        m_msg = self.W_o(attn_output)

        return m_msg, attn_weights


class FXStrengthDynamicA(nn.Module):
    """
    FX Strength GNN with Dynamic Cross-Attention A

    Key difference from FXStrengthGNN:
    - A matrix is not a fixed parameter
    - Instead, A_t is computed dynamically via cross-attention
    - Query: currency embedding, Key/Value: macro embedding
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

        # Step 2: Currency-currency spillover (GNN) - optional based on ablation
        if config.gnn_type == "gcn":
            self.ccy_gnn = GCNConv(config.hidden, config.hidden)
        elif config.gnn_type == "sage":
            self.ccy_gnn = SAGEConv(config.hidden, config.hidden)
        elif config.gnn_type == "gat":
            self.ccy_gnn = GATConv(config.hidden, config.hidden, heads=config.heads, concat=False)
        else:
            raise ValueError(f"Unknown GNN type: {config.gnn_type}")

        # Step 3: Macro embedding (project each macro factor to hidden dim)
        self.macro_proj = nn.Linear(1, config.hidden)  # Each macro factor -> hidden

        # Step 4: Dynamic Cross-Attention for macro-currency transmission
        self.macro_currency_attn = DynamicMacroCurrencyAttention(
            hidden_dim=config.hidden,
            macro_dim=config.macro_dim,
            n_heads=config.heads,
            dropout=0.1
        )

        # Step 5: Strength prediction head
        self.head = nn.Linear(config.hidden, 1)

        # Dummy A for compatibility with trainer (will store average attention)
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim), requires_grad=False)

        # Enhanced mode options
        self.use_skip_connection = config.use_skip_connection
        self.use_layer_norm = config.use_layer_norm

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden)
            self.macro_layer_norm = nn.LayerNorm(config.hidden)

    def forward(self, xl, xm, edge_index_single):
        """
        Args:
            xl: Local features [B, L, N, local_dim]
            xm: Macro features [B, L, macro_dim]
            edge_index_single: Edge index for single graph [2, E]

        Returns:
            rhat: Predicted FX returns relative to USD [B, N]
            ds: Currency strengths (zero-mean) [B, N]
            z_ccy: Currency embeddings after GNN [B, N, H]
            m_msg: Macro message to each currency [B, N, H]
        """
        B, L, N, local_dim = xl.shape

        # 1. Local encoding per currency
        x = xl.permute(0, 2, 1, 3).reshape(B * N, L, local_dim)
        _, h = self.local_gru(x)
        h = h.squeeze(0)  # [B*N, H]

        # Save for skip connection
        h_input = h if self.use_skip_connection else None

        # 2. Currency-currency spillover via GNN
        E = edge_index_single.size(1)
        edge_b = edge_index_single.repeat(1, B)
        offset = torch.arange(B, device=xl.device).repeat_interleave(E) * N
        edge_b = edge_b + offset.unsqueeze(0)

        z = self.ccy_gnn(h, edge_b)

        if self.use_skip_connection:
            z = z + h_input

        if self.use_layer_norm:
            z = self.layer_norm(z)

        z_ccy = z.view(B, N, self.hidden)  # [B, N, H]

        # 3. Macro embedding - project each factor separately
        m_t = xm[:, -1, :]  # [B, macro_dim] - use last timestep
        # Expand and project: [B, M] -> [B, M, 1] -> [B, M, H]
        macro_embed = self.macro_proj(m_t.unsqueeze(-1))  # [B, M, H]

        if self.use_layer_norm:
            macro_embed = self.macro_layer_norm(macro_embed)

        # 4. Dynamic cross-attention for macro-currency transmission
        m_msg, attn_weights = self.macro_currency_attn(z_ccy, macro_embed)
        # m_msg: [B, N, H], attn_weights: [B, heads, N, M]

        # Store average attention as A for interpretability
        # Average over heads and batch
        self.A.data = attn_weights.mean(dim=(0, 1)).detach()  # [N, M]

        # 5. Integration & Prediction
        z_total = z_ccy + m_msg
        ds = self.head(z_total).squeeze(-1)  # [B, N]

        # Zero-mean normalization
        ds = ds - ds.mean(dim=1, keepdim=True)

        # Relative to USD
        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]

        return rhat, ds, z_ccy, m_msg


class FXStrengthDynamicANoGNN(nn.Module):
    """
    Dynamic A without GNN (since ablation showed GNN doesn't help much)
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

        # No GNN

        # Step 2: Macro embedding
        self.macro_proj = nn.Linear(1, config.hidden)

        # Step 3: Dynamic Cross-Attention
        self.macro_currency_attn = DynamicMacroCurrencyAttention(
            hidden_dim=config.hidden,
            macro_dim=config.macro_dim,
            n_heads=config.heads,
            dropout=0.1
        )

        # Step 4: Strength prediction head
        self.head = nn.Linear(config.hidden, 1)

        # Dummy A for compatibility
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim), requires_grad=False)

        # Layer norm
        self.use_layer_norm = config.use_layer_norm
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden)
            self.macro_layer_norm = nn.LayerNorm(config.hidden)

    def forward(self, xl, xm, edge_index_single):
        B, L, N, local_dim = xl.shape

        # 1. Local encoding per currency
        x = xl.permute(0, 2, 1, 3).reshape(B * N, L, local_dim)
        _, h = self.local_gru(x)
        h = h.squeeze(0)

        z_ccy = h.view(B, N, self.hidden)

        if self.use_layer_norm:
            z_ccy = self.layer_norm(z_ccy)

        # 2. Macro embedding
        m_t = xm[:, -1, :]
        macro_embed = self.macro_proj(m_t.unsqueeze(-1))

        if self.use_layer_norm:
            macro_embed = self.macro_layer_norm(macro_embed)

        # 3. Dynamic cross-attention
        m_msg, attn_weights = self.macro_currency_attn(z_ccy, macro_embed)

        # Store attention as A
        self.A.data = attn_weights.mean(dim=(0, 1)).detach()

        # 4. Integration & Prediction
        z_total = z_ccy + m_msg
        ds = self.head(z_total).squeeze(-1)

        ds = ds - ds.mean(dim=1, keepdim=True)
        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]

        return rhat, ds, z_ccy, m_msg
