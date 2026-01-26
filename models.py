"""
FX Strength GNN Models
- Baseline: Original kswds implementation (GRU + GNN + Hetero Macro)
- Enhanced: With skip connections, layer norm, magnitude head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from config import Config


class FXStrengthGNN(nn.Module):
    """
    FX Strength GNN with Heterogeneous Macro-to-Currency Transmission

    Architecture:
    1. GRU: Temporal encoding per currency
    2. GNN: Multi-currency spillover (GCN/SAGE/GAT)
    3. Macro Embedding + Heterogeneous A matrix
    4. Strength prediction with zero-mean normalization
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
        if config.gnn_type == "gcn":
            self.ccy_gnn = GCNConv(config.hidden, config.hidden)
        elif config.gnn_type == "sage":
            self.ccy_gnn = SAGEConv(config.hidden, config.hidden)
        elif config.gnn_type == "gat":
            self.ccy_gnn = GATConv(config.hidden, config.hidden, heads=config.heads, concat=False)
        else:
            raise ValueError(f"Unknown GNN type: {config.gnn_type}")

        # Step 3: Heterogeneous macro-to-currency effects
        self.macro_embed = nn.Linear(config.macro_dim, config.macro_dim * config.hidden, bias=False)
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim))
        nn.init.normal_(self.A, mean=0.0, std=0.1)

        # Step 4: Strength prediction head
        self.head = nn.Linear(config.hidden, 1)

        # Enhanced mode options
        self.use_skip_connection = config.use_skip_connection
        self.use_layer_norm = config.use_layer_norm

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden)

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
        x = xl.permute(0, 2, 1, 3).reshape(B * N, L, local_dim)  # [B*N, L, F]
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

        # Skip connection
        if self.use_skip_connection:
            z = z + h_input

        # Layer norm
        if self.use_layer_norm:
            z = self.layer_norm(z)

        z_ccy = z.view(B, N, self.hidden)

        # 3. Macro embedding
        m_t = xm[:, -1, :]  # [B, macro_dim] - use last timestep
        u = self.macro_embed(m_t).view(B, self.macro_dim, self.hidden)  # [B, M, H]

        # 4. Heterogeneous transmission via A matrix
        A = self.A.unsqueeze(0).unsqueeze(-1)  # [1, N, M, 1]
        u_exp = u.unsqueeze(1)  # [B, 1, M, H]
        m_msg = (A * u_exp).sum(dim=2)  # [B, N, H]

        # 5. Integration & Prediction
        z_total = z_ccy + m_msg
        ds = self.head(z_total).squeeze(-1)  # [B, N]

        # Zero-mean normalization (numeraire-free)
        ds = ds - ds.mean(dim=1, keepdim=True)

        # Relative to USD
        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]

        return rhat, ds, z_ccy, m_msg


class FXStrengthGNNWithMagnitude(FXStrengthGNN):
    """
    Extended model with magnitude prediction head

    Predicts:
    - u_i,t: direction (zero-mean strength)
    - a_t: magnitude (market volatility scale)
    """

    def __init__(self, config: Config):
        super().__init__(config)

        # Magnitude head (predicts market volatility scale)
        self.magnitude_head = nn.Sequential(
            nn.Linear(config.hidden, config.hidden // 2),
            nn.ReLU(),
            nn.Linear(config.hidden // 2, 1)
        )
        nn.init.constant_(self.magnitude_head[-1].bias, 0.02)

    def forward(self, xl, xm, edge_index_single):
        rhat, ds, z_ccy, m_msg = super().forward(xl, xm, edge_index_single)

        # Global embedding for magnitude prediction
        B = xl.size(0)
        global_emb = z_ccy.mean(dim=1)  # [B, H] - average over currencies

        # Magnitude prediction
        raw_magnitude = self.magnitude_head(global_emb).squeeze(-1)  # [B]
        magnitude = F.softplus(raw_magnitude) + 1e-4

        return rhat, ds, z_ccy, m_msg, magnitude


def create_model(config: Config) -> nn.Module:
    """Factory function to create model based on config"""
    if config.use_magnitude_head:
        return FXStrengthGNNWithMagnitude(config)
    else:
        return FXStrengthGNN(config)


# ============================================================
# Baseline Models for Comparison
# ============================================================

class RandomWalkBaseline(nn.Module):
    """
    Random Walk baseline: predict 0 (no change)
    For normalized returns, this is equivalent to "tomorrow = today"
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_ccy = config.n_ccy
        self.usd_idx = config.usd_idx
        # Dummy parameter for optimizer
        self.dummy = nn.Parameter(torch.zeros(1))
        # Dummy A for compatibility
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim), requires_grad=False)

    def forward(self, xl, xm, edge_index_single):
        B = xl.size(0)
        # Predict zero (no change)
        rhat = torch.zeros(B, self.n_ccy, device=xl.device)
        ds = torch.zeros(B, self.n_ccy, device=xl.device)
        z_ccy = torch.zeros(B, self.n_ccy, self.config.hidden, device=xl.device)
        m_msg = torch.zeros(B, self.n_ccy, self.config.hidden, device=xl.device)
        return rhat, ds, z_ccy, m_msg


class LinearBaseline(nn.Module):
    """
    Linear baseline: simple linear projection from last timestep features
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_ccy = config.n_ccy
        self.usd_idx = config.usd_idx

        # Input: last timestep local features + macro features
        input_dim = config.local_dim + config.macro_dim
        self.linear = nn.Linear(input_dim, 1)

        # Dummy A for compatibility
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim), requires_grad=False)

    def forward(self, xl, xm, edge_index_single):
        B, L, N, local_dim = xl.shape

        # Use last timestep
        xl_last = xl[:, -1, :, :]  # [B, N, local_dim]
        xm_last = xm[:, -1, :]  # [B, macro_dim]

        # Concat macro to each currency
        xm_exp = xm_last.unsqueeze(1).expand(-1, N, -1)  # [B, N, macro_dim]
        x = torch.cat([xl_last, xm_exp], dim=-1)  # [B, N, local_dim + macro_dim]

        # Linear prediction
        ds = self.linear(x).squeeze(-1)  # [B, N]
        ds = ds - ds.mean(dim=1, keepdim=True)  # zero-mean

        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]

        z_ccy = torch.zeros(B, N, self.config.hidden, device=xl.device)
        m_msg = torch.zeros(B, N, self.config.hidden, device=xl.device)
        return rhat, ds, z_ccy, m_msg


class MLPBaseline(nn.Module):
    """
    MLP baseline: multi-layer perceptron on flattened features
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_ccy = config.n_ccy
        self.usd_idx = config.usd_idx

        # Input: last timestep local features + macro features
        input_dim = config.local_dim + config.macro_dim
        hidden = config.hidden

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

        # Dummy A for compatibility
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim), requires_grad=False)

    def forward(self, xl, xm, edge_index_single):
        B, L, N, local_dim = xl.shape

        # Use last timestep
        xl_last = xl[:, -1, :, :]  # [B, N, local_dim]
        xm_last = xm[:, -1, :]  # [B, macro_dim]

        # Concat macro to each currency
        xm_exp = xm_last.unsqueeze(1).expand(-1, N, -1)  # [B, N, macro_dim]
        x = torch.cat([xl_last, xm_exp], dim=-1)  # [B, N, local_dim + macro_dim]

        # MLP prediction
        ds = self.mlp(x).squeeze(-1)  # [B, N]
        ds = ds - ds.mean(dim=1, keepdim=True)  # zero-mean

        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]

        z_ccy = torch.zeros(B, N, self.config.hidden, device=xl.device)
        m_msg = torch.zeros(B, N, self.config.hidden, device=xl.device)
        return rhat, ds, z_ccy, m_msg


class LSTMBaseline(nn.Module):
    """
    LSTM baseline: LSTM per currency, no GNN, no macro structure
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_ccy = config.n_ccy
        self.usd_idx = config.usd_idx
        self.hidden = config.hidden

        # LSTM for temporal encoding (includes macro as input)
        input_dim = config.local_dim + config.macro_dim
        self.lstm = nn.LSTM(input_dim, config.hidden, batch_first=True)
        self.head = nn.Linear(config.hidden, 1)

        # Dummy A for compatibility
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim), requires_grad=False)

    def forward(self, xl, xm, edge_index_single):
        B, L, N, local_dim = xl.shape

        # Expand macro to each currency
        xm_exp = xm.unsqueeze(2).expand(-1, -1, N, -1)  # [B, L, N, macro_dim]
        x = torch.cat([xl, xm_exp], dim=-1)  # [B, L, N, local_dim + macro_dim]

        # Reshape for LSTM: [B*N, L, input_dim]
        x = x.permute(0, 2, 1, 3).reshape(B * N, L, -1)

        # LSTM encoding
        _, (h, _) = self.lstm(x)
        h = h.squeeze(0)  # [B*N, hidden]

        # Prediction
        ds = self.head(h).squeeze(-1)  # [B*N]
        ds = ds.view(B, N)  # [B, N]
        ds = ds - ds.mean(dim=1, keepdim=True)  # zero-mean

        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]

        z_ccy = h.view(B, N, self.hidden)
        m_msg = torch.zeros(B, N, self.hidden, device=xl.device)
        return rhat, ds, z_ccy, m_msg


class GRUOnlyBaseline(nn.Module):
    """
    GRU baseline: GRU per currency, no GNN, no structured macro
    Same as LSTM but with GRU
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_ccy = config.n_ccy
        self.usd_idx = config.usd_idx
        self.hidden = config.hidden

        # GRU for temporal encoding (includes macro as input)
        input_dim = config.local_dim + config.macro_dim
        self.gru = nn.GRU(input_dim, config.hidden, batch_first=True)
        self.head = nn.Linear(config.hidden, 1)

        # Dummy A for compatibility
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim), requires_grad=False)

    def forward(self, xl, xm, edge_index_single):
        B, L, N, local_dim = xl.shape

        # Expand macro to each currency
        xm_exp = xm.unsqueeze(2).expand(-1, -1, N, -1)  # [B, L, N, macro_dim]
        x = torch.cat([xl, xm_exp], dim=-1)  # [B, L, N, local_dim + macro_dim]

        # Reshape for GRU: [B*N, L, input_dim]
        x = x.permute(0, 2, 1, 3).reshape(B * N, L, -1)

        # GRU encoding
        _, h = self.gru(x)
        h = h.squeeze(0)  # [B*N, hidden]

        # Prediction
        ds = self.head(h).squeeze(-1)  # [B*N]
        ds = ds.view(B, N)  # [B, N]
        ds = ds - ds.mean(dim=1, keepdim=True)  # zero-mean

        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]

        z_ccy = h.view(B, N, self.hidden)
        m_msg = torch.zeros(B, N, self.hidden, device=xl.device)
        return rhat, ds, z_ccy, m_msg


# ============================================================
# SOTA Models
# ============================================================

class iTransformerBaseline(nn.Module):
    """
    iTransformer baseline (ICLR 2024 Spotlight)
    Inverted Transformer - treats each variate as a token
    """
    def __init__(self, config: Config):
        super().__init__()
        from iTransformer import iTransformer

        self.config = config
        self.n_ccy = config.n_ccy
        self.usd_idx = config.usd_idx
        self.hidden = config.hidden

        # Treat each currency as a variate
        # Input per variate: local_dim + macro_dim features over time
        input_dim = config.local_dim + config.macro_dim

        self.itransformer = iTransformer(
            num_variates=config.n_ccy,
            lookback_len=config.lookback,
            dim=config.hidden,
            depth=2,
            heads=4,
            dim_head=config.hidden // 4,
            pred_length=1,  # predict 1 step ahead
            num_tokens_per_variate=1,
            flash_attn=False,  # disable for compatibility
        )

        # Project input features to single value per timestep
        self.input_proj = nn.Linear(input_dim, 1)

        # Dummy A for compatibility
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim), requires_grad=False)

    def forward(self, xl, xm, edge_index_single):
        B, L, N, local_dim = xl.shape

        # Expand macro to each currency
        xm_exp = xm.unsqueeze(2).expand(-1, -1, N, -1)  # [B, L, N, macro_dim]
        x = torch.cat([xl, xm_exp], dim=-1)  # [B, L, N, local_dim + macro_dim]

        # Project to single value per timestep per currency
        x = self.input_proj(x).squeeze(-1)  # [B, L, N]
        # iTransformer expects [B, lookback_len, num_variates] = [B, L, N]
        # x is already in correct shape

        # iTransformer forward
        pred = self.itransformer(x)  # [B, pred_length, num_variates] = [B, 1, N]
        ds = pred.squeeze(1)  # [B, N]

        # Zero-mean normalization
        ds = ds - ds.mean(dim=1, keepdim=True)

        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]

        z_ccy = torch.zeros(B, N, self.hidden, device=xl.device)
        m_msg = torch.zeros(B, N, self.hidden, device=xl.device)
        return rhat, ds, z_ccy, m_msg


class PatchTSTBaseline(nn.Module):
    """
    PatchTST baseline (ICLR 2023)
    Patching + channel-independent Transformer
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_ccy = config.n_ccy
        self.usd_idx = config.usd_idx
        self.hidden = config.hidden

        input_dim = config.local_dim + config.macro_dim
        patch_len = 4
        stride = 2
        num_patches = (config.lookback - patch_len) // stride + 1

        # Patch embedding
        self.patch_len = patch_len
        self.stride = stride
        self.patch_embed = nn.Linear(patch_len * input_dim, config.hidden)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden,
            nhead=4,
            dim_feedforward=config.hidden * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Prediction head
        self.head = nn.Linear(config.hidden * num_patches, 1)

        # Dummy A for compatibility
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim), requires_grad=False)

    def forward(self, xl, xm, edge_index_single):
        B, L, N, local_dim = xl.shape

        # Expand macro to each currency
        xm_exp = xm.unsqueeze(2).expand(-1, -1, N, -1)  # [B, L, N, macro_dim]
        x = torch.cat([xl, xm_exp], dim=-1)  # [B, L, N, input_dim]
        input_dim = x.size(-1)

        # Reshape: [B*N, L, input_dim]
        x = x.permute(0, 2, 1, 3).reshape(B * N, L, input_dim)

        # Create patches: [B*N, num_patches, patch_len * input_dim]
        patches = []
        for i in range(0, L - self.patch_len + 1, self.stride):
            patch = x[:, i:i + self.patch_len, :].reshape(B * N, -1)
            patches.append(patch)
        patches = torch.stack(patches, dim=1)  # [B*N, num_patches, patch_len * input_dim]

        # Patch embedding
        patches = self.patch_embed(patches)  # [B*N, num_patches, hidden]

        # Transformer
        out = self.transformer(patches)  # [B*N, num_patches, hidden]

        # Flatten and predict
        out = out.reshape(B * N, -1)  # [B*N, num_patches * hidden]
        ds = self.head(out).squeeze(-1)  # [B*N]
        ds = ds.view(B, N)  # [B, N]

        # Zero-mean normalization
        ds = ds - ds.mean(dim=1, keepdim=True)

        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]

        z_ccy = torch.zeros(B, N, self.hidden, device=xl.device)
        m_msg = torch.zeros(B, N, self.hidden, device=xl.device)
        return rhat, ds, z_ccy, m_msg


class TimeMixerBaseline(nn.Module):
    """
    TimeMixer-style baseline (ICLR 2024)
    Simplified MLP-based multiscale mixing
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_ccy = config.n_ccy
        self.usd_idx = config.usd_idx
        self.hidden = config.hidden

        input_dim = config.local_dim + config.macro_dim
        L = config.lookback

        # Multi-scale decomposition (simplified)
        self.scales = [1, 2, 4]  # different temporal scales

        # Time mixing MLPs for each scale
        self.time_mixers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(L // s, config.hidden),
                nn.GELU(),
                nn.Linear(config.hidden, L // s)
            ) for s in self.scales
        ])

        # Feature mixing MLP
        self.feature_mixer = nn.Sequential(
            nn.Linear(input_dim, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, input_dim)
        )

        # Final projection
        self.proj = nn.Linear(input_dim * sum(L // s for s in self.scales), 1)

        # Dummy A for compatibility
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim), requires_grad=False)

    def forward(self, xl, xm, edge_index_single):
        B, L, N, local_dim = xl.shape

        # Expand macro to each currency
        xm_exp = xm.unsqueeze(2).expand(-1, -1, N, -1)
        x = torch.cat([xl, xm_exp], dim=-1)  # [B, L, N, input_dim]
        input_dim = x.size(-1)

        # Reshape: [B*N, L, input_dim]
        x = x.permute(0, 2, 1, 3).reshape(B * N, L, input_dim)

        # Multi-scale mixing
        outputs = []
        for i, scale in enumerate(self.scales):
            # Downsample
            x_scale = x[:, ::scale, :]  # [B*N, L//scale, input_dim]
            L_s = x_scale.size(1)

            # Time mixing
            x_t = x_scale.permute(0, 2, 1)  # [B*N, input_dim, L//scale]
            x_t = self.time_mixers[i](x_t)
            x_t = x_t.permute(0, 2, 1)  # [B*N, L//scale, input_dim]

            # Feature mixing
            x_f = self.feature_mixer(x_t)

            outputs.append(x_f.reshape(B * N, -1))  # [B*N, L//scale * input_dim]

        # Concatenate all scales
        out = torch.cat(outputs, dim=-1)  # [B*N, total_dim]

        # Final prediction
        ds = self.proj(out).squeeze(-1)  # [B*N]
        ds = ds.view(B, N)  # [B, N]

        # Zero-mean normalization
        ds = ds - ds.mean(dim=1, keepdim=True)

        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]

        z_ccy = torch.zeros(B, N, self.hidden, device=xl.device)
        m_msg = torch.zeros(B, N, self.hidden, device=xl.device)
        return rhat, ds, z_ccy, m_msg


# ============================================================
# Ablation Models
# ============================================================

class FXStrengthNoGNN(nn.Module):
    """
    Ablation: No GNN (GRU + Hetero Macro only)
    Tests the value of currency-currency spillover via GNN
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

        # No GNN - just pass through

        # Step 3: Heterogeneous macro-to-currency effects (same as full model)
        self.macro_embed = nn.Linear(config.macro_dim, config.macro_dim * config.hidden, bias=False)
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim))
        nn.init.normal_(self.A, mean=0.0, std=0.1)

        # Step 4: Strength prediction head
        self.head = nn.Linear(config.hidden, 1)

        # Layer norm
        self.use_layer_norm = config.use_layer_norm
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden)

    def forward(self, xl, xm, edge_index_single):
        B, L, N, local_dim = xl.shape

        # 1. Local encoding per currency
        x = xl.permute(0, 2, 1, 3).reshape(B * N, L, local_dim)
        _, h = self.local_gru(x)
        h = h.squeeze(0)  # [B*N, H]

        # No GNN - direct to currency embeddings
        z_ccy = h.view(B, N, self.hidden)

        if self.use_layer_norm:
            z_ccy = self.layer_norm(z_ccy)

        # 2. Macro embedding
        m_t = xm[:, -1, :]  # [B, macro_dim]
        u = self.macro_embed(m_t).view(B, self.macro_dim, self.hidden)

        # 3. Heterogeneous transmission via A matrix
        A = self.A.unsqueeze(0).unsqueeze(-1)
        u_exp = u.unsqueeze(1)
        m_msg = (A * u_exp).sum(dim=2)

        # 4. Integration & Prediction
        z_total = z_ccy + m_msg
        ds = self.head(z_total).squeeze(-1)

        # Zero-mean normalization
        ds = ds - ds.mean(dim=1, keepdim=True)

        # Relative to USD
        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]

        return rhat, ds, z_ccy, m_msg


class FXStrengthHomoA(nn.Module):
    """
    Ablation: Homogeneous A (same macro sensitivity for all currencies)
    Tests the value of heterogeneous macro transmission
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
        if config.gnn_type == "gcn":
            self.ccy_gnn = GCNConv(config.hidden, config.hidden)
        elif config.gnn_type == "sage":
            self.ccy_gnn = SAGEConv(config.hidden, config.hidden)
        elif config.gnn_type == "gat":
            self.ccy_gnn = GATConv(config.hidden, config.hidden, heads=config.heads, concat=False)
        else:
            raise ValueError(f"Unknown GNN type: {config.gnn_type}")

        # Step 3: Homogeneous macro embedding (SAME for all currencies)
        # Instead of A[N, M], use a_shared[1, M] that applies to all
        self.macro_embed = nn.Linear(config.macro_dim, config.macro_dim * config.hidden, bias=False)
        self.a_shared = nn.Parameter(torch.zeros(1, config.macro_dim))
        nn.init.normal_(self.a_shared, mean=0.0, std=0.1)

        # Dummy A for compatibility (derived from a_shared)
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim), requires_grad=False)

        # Step 4: Strength prediction head
        self.head = nn.Linear(config.hidden, 1)

        # Enhanced mode options
        self.use_skip_connection = config.use_skip_connection
        self.use_layer_norm = config.use_layer_norm

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden)

    def forward(self, xl, xm, edge_index_single):
        B, L, N, local_dim = xl.shape

        # 1. Local encoding per currency
        x = xl.permute(0, 2, 1, 3).reshape(B * N, L, local_dim)
        _, h = self.local_gru(x)
        h = h.squeeze(0)

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

        z_ccy = z.view(B, N, self.hidden)

        # 3. Macro embedding
        m_t = xm[:, -1, :]
        u = self.macro_embed(m_t).view(B, self.macro_dim, self.hidden)

        # 4. HOMOGENEOUS transmission - same a_shared for all currencies
        # Expand a_shared to all currencies
        A_homo = self.a_shared.expand(N, -1)  # [N, M], all rows identical
        A = A_homo.unsqueeze(0).unsqueeze(-1)  # [1, N, M, 1]
        u_exp = u.unsqueeze(1)  # [B, 1, M, H]
        m_msg = (A * u_exp).sum(dim=2)  # [B, N, H]

        # Update dummy A for metrics
        self.A.data = A_homo.detach()

        # 5. Integration & Prediction
        z_total = z_ccy + m_msg
        ds = self.head(z_total).squeeze(-1)

        # Zero-mean normalization
        ds = ds - ds.mean(dim=1, keepdim=True)

        # Relative to USD
        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]

        return rhat, ds, z_ccy, m_msg


class FXStrengthNoMacro(nn.Module):
    """
    Ablation: No Macro (GRU + GNN only, no macro information)
    Tests the value of macro-economic information
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden = config.hidden
        self.n_ccy = config.n_ccy
        self.usd_idx = config.usd_idx

        # Step 1: Temporal encoding (GRU)
        self.local_gru = nn.GRU(config.local_dim, config.hidden, batch_first=True)

        # Step 2: Currency-currency spillover (GNN)
        if config.gnn_type == "gcn":
            self.ccy_gnn = GCNConv(config.hidden, config.hidden)
        elif config.gnn_type == "sage":
            self.ccy_gnn = SAGEConv(config.hidden, config.hidden)
        elif config.gnn_type == "gat":
            self.ccy_gnn = GATConv(config.hidden, config.hidden, heads=config.heads, concat=False)
        else:
            raise ValueError(f"Unknown GNN type: {config.gnn_type}")

        # No macro embedding

        # Step 3: Strength prediction head
        self.head = nn.Linear(config.hidden, 1)

        # Dummy A for compatibility
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim), requires_grad=False)

        # Enhanced mode options
        self.use_skip_connection = config.use_skip_connection
        self.use_layer_norm = config.use_layer_norm

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden)

    def forward(self, xl, xm, edge_index_single):
        B, L, N, local_dim = xl.shape

        # 1. Local encoding per currency
        x = xl.permute(0, 2, 1, 3).reshape(B * N, L, local_dim)
        _, h = self.local_gru(x)
        h = h.squeeze(0)

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

        z_ccy = z.view(B, N, self.hidden)

        # No macro - direct prediction from GNN output
        ds = self.head(z_ccy).squeeze(-1)

        # Zero-mean normalization
        ds = ds - ds.mean(dim=1, keepdim=True)

        # Relative to USD
        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]

        m_msg = torch.zeros(B, N, self.hidden, device=xl.device)
        return rhat, ds, z_ccy, m_msg


BASELINE_MODELS = {
    "random_walk": RandomWalkBaseline,
    "linear": LinearBaseline,
    "mlp": MLPBaseline,
    "lstm": LSTMBaseline,
    "gru": GRUOnlyBaseline,
    "ours": FXStrengthGNN,
    # SOTA models
    "itransformer": iTransformerBaseline,
    "patchtst": PatchTSTBaseline,
    "timemixer": TimeMixerBaseline,
}

ABLATION_MODELS = {
    "no_gnn": FXStrengthNoGNN,
    "homo_a": FXStrengthHomoA,
    "no_macro": FXStrengthNoMacro,
    "full": FXStrengthGNN,
}
