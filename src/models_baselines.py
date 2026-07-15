from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .models_ours import usd_pin
except ImportError:
    from models_ours import usd_pin


class SharedMLPBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden: int, lookback: int, n_ccy: int, dropout: float = 0.2):
        super().__init__()
        self.n_ccy = n_ccy
        self.lookback = lookback
        self.net = nn.Sequential(
            nn.Linear(input_dim * lookback, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, n, f = x.shape
        flat = x.permute(0, 2, 1, 3).reshape(b * n, l * f)
        ds = self.net(flat).reshape(b, n)
        return usd_pin(ds, 0)


class TemporalRNNBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden: int, n_ccy: int, rnn_type: str = "gru", dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.n_ccy = n_ccy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, n, f = x.shape
        z = self.input_proj(x).permute(0, 2, 1, 3).reshape(b * n, l, -1)
        _, state = self.rnn(z)
        h = state[0][-1] if isinstance(state, tuple) else state[-1]
        ds = self.head(h).reshape(b, n)
        return usd_pin(ds, 0)


class GNNTBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden: int, n_ccy: int, top_k: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_ccy = n_ccy
        self.top_k = max(1, min(top_k, n_ccy - 1))
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.adj_logits = nn.Parameter(torch.randn(n_ccy, n_ccy) * 0.02)
        self.msg_proj = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def _adj(self) -> torch.Tensor:
        logits = self.adj_logits
        eye = torch.eye(self.n_ccy, device=logits.device, dtype=torch.bool)
        masked = logits.masked_fill(eye, float("-inf"))
        vals, idx = torch.topk(masked, k=self.top_k, dim=-1)
        sparse = torch.full_like(masked, float("-inf"))
        sparse.scatter_(1, idx, vals)
        return torch.nan_to_num(torch.softmax(sparse, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, n, f = x.shape
        z = self.input_proj(x).permute(0, 2, 1, 3).reshape(b * n, l, -1)
        _, h = self.gru(z)
        h = h[-1].reshape(b, n, -1)
        adj = self._adj().unsqueeze(0).expand(b, -1, -1)
        msg = torch.einsum("bij,bjh->bih", adj, h)
        ds = self.head(h + self.msg_proj(msg)).squeeze(-1)
        return usd_pin(ds, 0)


def build_baseline_model(model_name: str, input_dim: int, hidden: int, n_ccy: int, lookback: int, top_k: int, dropout: float):
    name = model_name.lower()
    if name == "mlp":
        return SharedMLPBaseline(input_dim, hidden, lookback, n_ccy, dropout=dropout)
    if name == "gru":
        return TemporalRNNBaseline(input_dim, hidden, n_ccy, rnn_type="gru", dropout=dropout)
    if name == "lstm":
        return TemporalRNNBaseline(input_dim, hidden, n_ccy, rnn_type="lstm", dropout=dropout)
    if name == "gnn":
        return GNNTBaseline(input_dim, hidden, n_ccy, top_k=top_k, dropout=dropout)
    raise ValueError(f"Unsupported baseline model: {model_name}")
