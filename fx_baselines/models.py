import torch
import torch.nn as nn

from config import Config

# -------------------------
# Graph utilities
# -------------------------
def fully_connected_row_stochastic(N: int, device: torch.device) -> torch.Tensor:
    """Row-stochastic adjacency with zero diagonal."""
    W = torch.ones((N, N), device=device)
    W.fill_diagonal_(0.0)
    W = W / (W.sum(dim=-1, keepdim=True) + 1e-12)
    return W

def topk_row(W, k: int):
    """Keep top-k per row and renormalize (numpy array)."""
    import numpy as np
    if k <= 0 or k >= W.shape[1]:
        return W / (W.sum(axis=1, keepdims=True) + 1e-12)
    N = W.shape[0]
    out = np.zeros_like(W)
    for i in range(N):
        idx = np.argsort(W[i])[::-1][:k]
        out[i, idx] = W[i, idx]
    return out / (out.sum(axis=1, keepdims=True) + 1e-12)

def build_granger_proxy_graph(Y_raw, train_split_idx: int, config: Config):
    """
    Directed Granger-proxy graph (TRAIN-only):

      score_{i<-j} = |corr(y_{t-1,j}, y_{t,i})|,   t in train label range

    Then:
      - zero diagonal
      - optional threshold
      - top-k per row (incoming)
      - row-normalize -> W_G (row-stochastic)
    """
    import numpy as np

    N = config.n_ccy
    L = config.lookback
    y = Y_raw.astype(np.float64)

    y_curr = y[L:train_split_idx, :]          # [Ttr,N]
    y_lag  = y[L-1:train_split_idx-1, :]      # [Ttr,N]

    y_curr_z = (y_curr - y_curr.mean(axis=0, keepdims=True)) / (y_curr.std(axis=0, keepdims=True) + 1e-12)
    y_lag_z  = (y_lag  - y_lag.mean(axis=0, keepdims=True))  / (y_lag.std(axis=0, keepdims=True) + 1e-12)

    C = (y_curr_z.T @ y_lag_z) / (y_curr_z.shape[0] + 1e-12)  # [N,N] (i,j)

    if config.granger_use_abs_corr:
        C = np.abs(C)

    np.fill_diagonal(C, 0.0)

    if config.granger_min_weight > 0:
        C = np.where(C >= config.granger_min_weight, C, 0.0)

    W = topk_row(C.copy(), config.granger_topk)

    # fallback for all-zero rows
    for i in range(N):
        if W[i].sum() <= 1e-12:
            W[i] = 1.0
            W[i, i] = 0.0
            W[i] = W[i] / (W[i].sum() + 1e-12)

    return W.astype("float32")

# -------------------------
# Baseline: MLP
# -------------------------
class FXBaselineMLP(nn.Module):
    """
    MLP baseline (same style as provided MLPBaseline):

      Use only last timestep local features + last macro features
      -> concat per currency -> MLP -> ds -> USD-base return prediction rhat

    No GRU, no heterogeneous macro injection (A is dummy for compatibility).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_ccy = config.n_ccy
        self.usd_idx = config.usd_idx

        input_dim = config.local_dim + config.macro_dim
        hidden = config.hidden

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        # Dummy A for compatibility with other models (non-trainable)
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim), requires_grad=False)

    def forward(self, xl, xm, *_):
        """
        xl: [B, L, N, local_dim]
        xm: [B, L, macro_dim]
        returns:
          rhat: [B, N]
          ds  : [B, N]
          dummy/dummy: shape-compatible placeholders
        """
        B, L, N, local_dim = xl.shape

        # last timestep only
        xl_last = xl[:, -1, :, :]     # [B, N, local_dim]
        xm_last = xm[:, -1, :]        # [B, macro_dim]

        # concat macro to each currency
        xm_exp = xm_last.unsqueeze(1).expand(-1, N, -1)   # [B, N, macro_dim]
        x = torch.cat([xl_last, xm_exp], dim=-1)          # [B, N, local_dim + macro_dim]

        # MLP prediction
        ds = self.mlp(x).squeeze(-1)  # [B, N]

        # Optional: zero-mean gauge fixing (commented out to match your provided code)
        # ds = ds - ds.mean(dim=1, keepdim=True)

        # USD-base return prediction
        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]  # [B, N]

        dummy = torch.zeros((B, N, 1), device=xl.device)
        return rhat, ds, dummy, dummy

# -------------------------
# FC graph baseline
# -------------------------
class StaticFCGNNMacro(nn.Module):
    """
    Static FC graph baseline (row-stochastic W_fc):

      GRU(local) -> FC aggregation -> + macro injection -> predict
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.N = config.n_ccy
        self.H = config.hidden
        self.M = config.macro_dim

        self.local_gru = nn.GRU(config.local_dim, config.hidden, batch_first=True)

        self.h_proj = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.H, self.H),
        )

        self.macro_embed = nn.Linear(self.M, self.M * self.H, bias=False)
        self.A = nn.Parameter(torch.zeros(self.N, self.M))
        nn.init.normal_(self.A, mean=0.0, std=0.1)

        self.head = nn.Linear(self.H, 1)

    def forward(self, xl, xm, W_fc: torch.Tensor):
        B, L, N, D = xl.shape
        x = xl.permute(0, 2, 1, 3).reshape(B * N, L, D)
        _, h = self.local_gru(x)
        h = h.squeeze(0).view(B, N, self.H)

        z_ccy = torch.matmul(W_fc.unsqueeze(0), self.h_proj(h))  # [B,N,H]

        m_t = xm[:, -1, :]
        u = self.macro_embed(m_t).view(B, self.M, self.H)
        A = self.A.unsqueeze(0).unsqueeze(-1)
        m_msg = (A * u.unsqueeze(1)).sum(dim=2)

        z_total = z_ccy + m_msg
        ds = self.head(z_total).squeeze(-1)
        ds = ds - ds.mean(dim=1, keepdim=True)
        rhat = ds - ds[:, self.config.usd_idx:self.config.usd_idx + 1]
        return rhat, ds, z_ccy, m_msg

# -------------------------
# Granger + Shock propagation
# -------------------------
class GrangerShockPropagationGNN(nn.Module):
    """
    Granger-proxy + shock propagation model:

      GRU(local) -> propagate on Wg -> + macro injection -> propagate again -> predict
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.N = config.n_ccy
        self.H = config.hidden
        self.M = config.macro_dim

        self.local_gru = nn.GRU(config.local_dim, config.hidden, batch_first=True)

        self.h_proj = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.H, self.H),
        )

        self.macro_embed = nn.Linear(self.M, self.M * self.H, bias=False)
        self.A = nn.Parameter(torch.zeros(self.N, self.M))
        nn.init.normal_(self.A, mean=0.0, std=0.1)

        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.head = nn.Linear(self.H, 1)

    def forward(self, xl, xm, Wg: torch.Tensor):
        B, L, N, D = xl.shape
        x = xl.permute(0, 2, 1, 3).reshape(B * N, L, D)
        _, h = self.local_gru(x)
        h = h.squeeze(0).view(B, N, self.H)

        z1 = torch.matmul(Wg.unsqueeze(0), self.h_proj(h))

        m_t = xm[:, -1, :]
        u = self.macro_embed(m_t).view(B, self.M, self.H)
        A = self.A.unsqueeze(0).unsqueeze(-1)
        m_msg = (A * u.unsqueeze(1)).sum(dim=2)

        if self.config.shockprop_steps <= 1:
            z_total = z1 + m_msg
        else:
            z2 = torch.matmul(Wg.unsqueeze(0), self.h_proj(z1 + self.gamma * m_msg))
            z_total = z2 + m_msg

        ds = self.head(z_total).squeeze(-1)
        ds = ds - ds.mean(dim=1, keepdim=True)
        rhat = ds - ds[:, self.config.usd_idx:self.config.usd_idx + 1]
        return rhat, ds, z1, m_msg
