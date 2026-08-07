from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .models_ours import usd_pin
except ImportError:
    from models_ours import usd_pin


def build_corrlstmgat_static_graph(train_targets_nonusd: np.ndarray, threshold: float = 0.7) -> dict[str, Any]:
    arr = np.asarray(train_targets_nonusd, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected [n_dates, n_nonusd], got shape={arr.shape}")
    corr = np.nan_to_num(np.corrcoef(arr, rowvar=False), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    n = int(corr.shape[0])
    support = corr > float(threshold)
    np.fill_diagonal(support, False)
    isolated_mask = support.sum(axis=1) == 0
    adjacency = support.astype(np.float32)
    np.fill_diagonal(adjacency, 1.0)
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if support[i, j] or support[j, i]:
                edge_list.append({"src_idx": i, "dst_idx": j, "corr": float(corr[i, j])})
    density = float(support.sum() / max(n * max(n - 1, 1), 1))
    return {
        "correlation_matrix": corr,
        "adjacency_matrix": adjacency,
        "edge_list": edge_list,
        "graph_density": density,
        "isolated_count_before_self_loops": int(isolated_mask.sum()),
    }


def bilateral_return_panel_from_usd_returns(usd_relative_returns: torch.Tensor) -> torch.Tensor:
    if usd_relative_returns.ndim != 3:
        raise ValueError(f"Expected [batch, lookback, n_ccy], got shape={tuple(usd_relative_returns.shape)}")
    return usd_relative_returns.unsqueeze(-1) - usd_relative_returns.unsqueeze(-2)


def _window_mean_features(seq: torch.Tensor, windows: Sequence[int]) -> list[torch.Tensor]:
    if seq.ndim < 2:
        raise ValueError(f"Expected at least 2 dims for time aggregation, got shape={tuple(seq.shape)}")
    out = []
    lookback = seq.size(1)
    for w in windows:
        eff = min(int(w), lookback)
        out.append(seq[:, -eff:].mean(dim=1))
    return out


class DenseGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 1, concat: bool = True, dropout: float = 0.2):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.heads = int(max(1, heads))
        self.concat = bool(concat)
        self.proj = nn.Linear(self.in_dim, self.out_dim * self.heads, bias=False)
        self.attn_src = nn.Parameter(torch.empty(self.heads, self.out_dim))
        self.attn_dst = nn.Parameter(torch.empty(self.heads, self.out_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        b, n, _ = h.shape
        proj = self.proj(h).reshape(b, n, self.heads, self.out_dim)
        proj_h = proj.permute(0, 2, 1, 3)
        src_score = (proj * self.attn_src.view(1, 1, self.heads, self.out_dim)).sum(dim=-1).permute(0, 2, 1)
        dst_score = (proj * self.attn_dst.view(1, 1, self.heads, self.out_dim)).sum(dim=-1).permute(0, 2, 1)
        logits = self.leaky_relu(src_score.unsqueeze(-1) + dst_score.unsqueeze(-2))
        if adjacency.ndim == 2:
            mask = adjacency <= 0
            logits = logits.masked_fill(mask.view(1, 1, n, n), float("-inf"))
        else:
            logits = logits.masked_fill(adjacency.unsqueeze(1) <= 0, float("-inf"))
        attn = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.dropout(attn)
        msg = torch.matmul(attn, proj_h).permute(0, 2, 1, 3)
        if self.concat:
            return msg.reshape(b, n, self.heads * self.out_dim)
        return msg.mean(dim=2)


class CorrLSTMGAT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_ccy: int,
        adjacency: torch.Tensor | np.ndarray,
        hidden: int = 32,
        dropout: float = 0.2,
        n_heads: int = 4,
        architecture_order: str = "lstm_then_gat",
        usd_idx: int = 0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_ccy = int(n_ccy)
        self.usd_idx = int(usd_idx)
        self.n_targets = self.n_ccy - 1
        self.hidden = int(hidden)
        self.architecture_order = str(architecture_order)
        adj = torch.as_tensor(adjacency, dtype=torch.float32)
        if adj.shape != (self.n_targets, self.n_targets):
            raise ValueError(f"CorrLSTMGAT expects adjacency of shape {(self.n_targets, self.n_targets)}, got {tuple(adj.shape)}")
        self.register_buffer("adjacency", adj)
        gat_head_dim = max(1, self.hidden // max(1, n_heads))
        if self.architecture_order == "lstm_then_gat":
            self.temporal_lstm = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
            )
        elif self.architecture_order == "gat_then_lstm":
            self.pre_gat_proj = nn.Sequential(nn.Linear(self.input_dim, self.hidden), nn.GELU(), nn.Dropout(dropout))
            self.temporal_lstm = nn.LSTM(
                input_size=self.hidden,
                hidden_size=self.hidden,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported architecture_order: {architecture_order}")
        self.gat1 = DenseGATLayer(self.hidden, gat_head_dim, heads=n_heads, concat=True, dropout=dropout)
        self.gat2 = DenseGATLayer(gat_head_dim * max(1, n_heads), self.hidden, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.hidden, 1)

    def _merge_nonusd_with_usd_anchor(self, ds_nonusd: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        b = ds_nonusd.size(0)
        usd_col = torch.zeros(b, 1, device=device, dtype=dtype)
        if self.usd_idx == 0:
            ds = torch.cat([usd_col, ds_nonusd], dim=1)
        elif self.usd_idx == self.n_ccy - 1:
            ds = torch.cat([ds_nonusd, usd_col], dim=1)
        else:
            ds = torch.cat([ds_nonusd[:, :self.usd_idx], usd_col, ds_nonusd[:, self.usd_idx:]], dim=1)
        return usd_pin(ds, self.usd_idx)

    def _run_gat_stack(self, h: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gat1(h, self.adjacency))
        h = self.dropout(h)
        h = F.relu(self.gat2(h, self.adjacency))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nonusd = torch.cat([x[:, :, :self.usd_idx], x[:, :, self.usd_idx + 1:]], dim=2)
        b, l, n, f = nonusd.shape
        if self.architecture_order == "lstm_then_gat":
            seq = nonusd.permute(0, 2, 1, 3).reshape(b * n, l, f)
            _, (h_n, _) = self.temporal_lstm(seq)
            h = h_n[-1].reshape(b, n, self.hidden)
            h = self._run_gat_stack(h)
        else:
            step_feats = self.pre_gat_proj(nonusd)
            gat_seq = []
            for t in range(l):
                gat_seq.append(self._run_gat_stack(step_feats[:, t]))
            seq = torch.stack(gat_seq, dim=2).reshape(b * n, l, self.hidden)
            _, (h_n, _) = self.temporal_lstm(seq)
            h = h_n[-1].reshape(b, n, self.hidden)
        ds_nonusd = self.head(h).squeeze(-1)
        return self._merge_nonusd_with_usd_anchor(ds_nonusd, x.dtype, x.device)


class FXRPSLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.use_activation = bool(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return self.activation(x) if self.use_activation else x


class FXRP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_ccy: int,
        hidden: int = 32,
        dropout: float = 0.2,
        num_layers: int = 3,
        usd_idx: int = 0,
        windows: Sequence[int] = (1, 3, 5, 10),
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_ccy = int(n_ccy)
        self.hidden = int(hidden)
        self.num_layers = int(max(1, num_layers))
        self.usd_idx = int(usd_idx)
        self.windows = tuple(int(w) for w in windows)
        self.node_summary_dim = self.input_dim * (1 + len(self.windows))
        self.edge_summary_dim = len(self.windows)
        self.node_layers = nn.ModuleList()
        self.edge_layers = nn.ModuleList()
        # The first layer directly consumes the raw adapted node/edge summaries.
        self.node_layers.append(FXRPSLP(2 * self.node_summary_dim + self.edge_summary_dim, self.hidden, activation=True))
        self.edge_layers.append(FXRPSLP(2 * self.hidden + self.edge_summary_dim, self.hidden, activation=True))
        for _ in range(1, self.num_layers):
            self.node_layers.append(FXRPSLP(3 * self.hidden, self.hidden, activation=True))
            self.edge_layers.append(FXRPSLP(3 * self.hidden, self.hidden, activation=True))
        # The paper's output head is an activation-free linear readout on the final edge state.
        self.edge_head = FXRPSLP(self.hidden, 1, activation=False)
        edge_mask = ~torch.eye(self.n_ccy, dtype=torch.bool)
        self.register_buffer("edge_mask", edge_mask)
        self.in_degree = float(max(self.n_ccy - 1, 1))

    def _node_summary(self, x: torch.Tensor) -> torch.Tensor:
        parts = [x[:, -1]]
        parts.extend(_window_mean_features(x, self.windows))
        return torch.cat(parts, dim=-1)

    def _edge_summary(self, x: torch.Tensor) -> torch.Tensor:
        usd_ret_hist = x[:, :, :, 0]
        bilateral_hist = bilateral_return_panel_from_usd_returns(usd_ret_hist)
        feats = _window_mean_features(bilateral_hist, self.windows)
        return torch.stack(feats, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n, _ = x.shape
        if n != self.n_ccy:
            raise ValueError(f"FXRP expected n_ccy={self.n_ccy}, got {n}")
        node_state: torch.Tensor | None = None
        edge_state: torch.Tensor | None = None
        node_raw = self._node_summary(x)
        edge_raw = self._edge_summary(x)
        mask = self.edge_mask.view(1, n, n, 1).to(edge_raw.dtype)
        edge_raw = edge_raw * mask
        for layer_idx, (node_layer, edge_layer) in enumerate(zip(self.node_layers, self.edge_layers)):
            prev_node = node_raw if layer_idx == 0 else node_state
            prev_edge = edge_raw if layer_idx == 0 else edge_state
            assert prev_node is not None
            assert prev_edge is not None
            src_prev = prev_node.unsqueeze(2).expand(-1, -1, n, -1)
            dst_prev = prev_node.unsqueeze(1).expand(-1, n, -1, -1)
            # Node update: average pairwise-transformed incoming messages phi(n_i, e_ji, n_j).
            incoming_input = torch.cat([dst_prev, prev_edge, src_prev], dim=-1)
            incoming_msg = node_layer(incoming_input) * mask
            node_state = incoming_msg.sum(dim=1) / self.in_degree
            src_new = node_state.unsqueeze(2).expand(-1, -1, n, -1)
            dst_new = node_state.unsqueeze(1).expand(-1, n, -1, -1)
            edge_input = torch.cat([src_new, prev_edge, dst_new], dim=-1)
            edge_state = edge_layer(edge_input) * mask
        assert edge_state is not None
        edge_scores = self.edge_head(edge_state).squeeze(-1)
        # Read the canonical directed edge i -> USD directly instead of combining reverse edges.
        ds = edge_scores[:, :, self.usd_idx]
        ds[:, self.usd_idx] = 0.0
        return ds


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


class GATBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden: int, n_ccy: int, top_k: int = 2, dropout: float = 0.2, n_heads: int = 4):
        super().__init__()
        self.n_ccy = n_ccy
        self.hidden = hidden
        self.top_k = max(1, min(top_k, n_ccy - 1))
        self.n_heads = max(1, n_heads)
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.out_proj = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def _multihead_attention(self, h: torch.Tensor) -> torch.Tensor:
        b, n, d = h.shape
        head_dim = max(1, d // self.n_heads)
        proj_dim = head_dim * self.n_heads
        q = self.q_proj(h)[..., :proj_dim].reshape(b, n, self.n_heads, head_dim).transpose(1, 2)
        k = self.k_proj(h)[..., :proj_dim].reshape(b, n, self.n_heads, head_dim).transpose(1, 2)
        v = self.v_proj(h)[..., :proj_dim].reshape(b, n, self.n_heads, head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / (float(head_dim) ** 0.5)
        eye = torch.eye(n, device=h.device, dtype=torch.bool).view(1, 1, n, n)
        logits = logits.masked_fill(eye, float("-inf"))
        if self.top_k < n:
            topk_vals, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)
            sparse = torch.full_like(logits, float("-inf"))
            sparse.scatter_(-1, topk_idx, topk_vals)
            logits = sparse
        attn = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        msg = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, proj_dim)
        if proj_dim < d:
            msg = F.pad(msg, (0, d - proj_dim))
        return msg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, n, _ = x.shape
        z = self.input_proj(x).permute(0, 2, 1, 3).reshape(b * n, l, -1)
        _, h_last = self.gru(z)
        h = h_last[-1].reshape(b, n, self.hidden)
        msg = self._multihead_attention(h)
        ds = self.head(h + self.out_proj(msg)).squeeze(-1)
        return usd_pin(ds, 0)


class TransformerBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden: int, lookback: int, n_ccy: int, dropout: float = 0.2, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.lookback = lookback
        self.hidden = hidden
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.pos_embed = nn.Parameter(torch.zeros(1, lookback, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=max(1, min(n_heads, hidden)),
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.time_attn = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, n, _ = x.shape
        z = self.input_proj(x).permute(0, 2, 1, 3).reshape(b * n, l, self.hidden)
        z = z + self.pos_embed[:, :l]
        h = self.encoder(z)
        attn = torch.nan_to_num(F.softmax(self.time_attn(h).squeeze(-1), dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        pooled = torch.sum(attn.unsqueeze(-1) * h, dim=1)
        ds = self.head(pooled).reshape(b, n)
        return usd_pin(ds, 0)


class ITransformerBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden: int, lookback: int, n_ccy: int, dropout: float = 0.25, n_heads: int = 4, n_layers: int = 2, d_ff: int | None = None):
        super().__init__()
        self.n_ccy = n_ccy
        self.input_dim = input_dim
        self.lookback = lookback
        self.hidden = hidden
        d_ff = int(d_ff or hidden * 2)
        self.history_proj = nn.Linear(lookback * input_dim, hidden)
        self.currency_embed = nn.Parameter(torch.zeros(1, n_ccy, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=max(1, min(n_heads, hidden)),
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, n, f = x.shape
        tokens = x.permute(0, 2, 1, 3).reshape(b, n, l * f)
        token_repr = self.history_proj(tokens) + self.currency_embed[:, :n]
        encoded = self.encoder(token_repr)
        ds = self.head(encoded).squeeze(-1)
        return usd_pin(ds, 0)


class TimeXerBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int,
        lookback: int,
        n_ccy: int,
        patch_len: int = 2,
        dropout: float = 0.25,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int | None = None,
    ):
        super().__init__()
        if lookback % patch_len != 0:
            raise ValueError(f"lookback={lookback} must be divisible by patch_len={patch_len}")
        self.n_ccy = n_ccy
        self.n_targets = n_ccy - 1
        self.input_dim = input_dim
        self.lookback = lookback
        self.patch_len = patch_len
        self.n_patches = lookback // patch_len
        self.hidden = hidden
        d_ff = int(d_ff or hidden * 2)
        self.endog_patch_proj = nn.Linear(patch_len, hidden)
        self.endog_pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, hidden))
        self.currency_embed = nn.Parameter(torch.zeros(1, self.n_targets, hidden))
        self.exog_hist_proj = nn.Linear(lookback * input_dim, hidden)
        self.exog_embed = nn.Parameter(torch.zeros(1, n_ccy, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=max(1, min(n_heads, hidden)),
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.endog_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.exog_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cross_attn = nn.MultiheadAttention(hidden, max(1, min(n_heads, hidden)), dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden)
        self.patch_pool = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, n, f = x.shape
        endog = x[:, :, 1:, 0].permute(0, 2, 1).reshape(b, self.n_targets, self.n_patches, self.patch_len)
        endog_tokens = self.endog_patch_proj(endog)
        endog_tokens = endog_tokens + self.endog_pos_embed[:, None, :, :] + self.currency_embed[:, :, None, :]
        endog_tokens = endog_tokens.reshape(b, self.n_targets * self.n_patches, self.hidden)
        endog_tokens = self.endog_encoder(endog_tokens)

        exog = x.clone()
        exog[:, :, 1:, 0] = 0.0
        exog_hist = exog.permute(0, 2, 1, 3).reshape(b, n, l * f)
        exog_tokens = self.exog_hist_proj(exog_hist) + self.exog_embed[:, :n]
        exog_tokens = self.exog_encoder(exog_tokens)

        cross_out, _ = self.cross_attn(endog_tokens, exog_tokens, exog_tokens, need_weights=False)
        endog_tokens = self.cross_norm(endog_tokens + cross_out)
        endog_tokens = endog_tokens.reshape(b, self.n_targets, self.n_patches, self.hidden)
        patch_attn = torch.nan_to_num(F.softmax(self.patch_pool(endog_tokens).squeeze(-1), dim=2), nan=0.0, posinf=0.0, neginf=0.0)
        pooled = torch.sum(patch_attn.unsqueeze(-1) * endog_tokens, dim=2)
        nonusd_ds = self.head(pooled).squeeze(-1)
        usd_ds = torch.zeros(b, 1, device=x.device, dtype=x.dtype)
        ds = torch.cat([usd_ds, nonusd_ds], dim=1)
        return usd_pin(ds, 0)


class ZeroReturnBaseline(nn.Module):
    def __init__(self, n_ccy: int):
        super().__init__()
        self.n_ccy = n_ccy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.size(0), self.n_ccy, device=x.device, dtype=x.dtype)


class RandomWalkBaseline(nn.Module):
    def __init__(self, n_ccy: int):
        super().__init__()
        self.n_ccy = n_ccy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The first local feature is the most recent USD-relative 1-day FX return.
        if x.size(-1) == 0:
            return torch.zeros(x.size(0), self.n_ccy, device=x.device, dtype=x.dtype)
        last_ret = x[:, -1, :, 0]
        return last_ret.clone()


def build_baseline_model(
    model_name: str,
    input_dim: int,
    hidden: int,
    n_ccy: int,
    lookback: int,
    top_k: int,
    dropout: float,
    patch_len: int | None = None,
    **model_kwargs,
):
    name = model_name.lower()
    if name == "mlp":
        return SharedMLPBaseline(input_dim, hidden, lookback, n_ccy, dropout=dropout)
    if name == "gru":
        return TemporalRNNBaseline(input_dim, hidden, n_ccy, rnn_type="gru", dropout=dropout)
    if name == "lstm":
        return TemporalRNNBaseline(input_dim, hidden, n_ccy, rnn_type="lstm", dropout=dropout)
    if name == "gnn":
        return GNNTBaseline(input_dim, hidden, n_ccy, top_k=top_k, dropout=dropout)
    if name == "gat":
        return GATBaseline(input_dim, hidden, n_ccy, top_k=top_k, dropout=dropout)
    if name == "transformer":
        return TransformerBaseline(input_dim, hidden, lookback, n_ccy, dropout=dropout)
    if name == "itransformer":
        return ITransformerBaseline(input_dim, hidden, lookback, n_ccy, dropout=dropout)
    if name == "timexer":
        return TimeXerBaseline(input_dim, hidden, lookback, n_ccy, patch_len=patch_len or 2, dropout=dropout)
    if name in {"corr_lstm_gat", "corrlstmgat"}:
        adjacency = model_kwargs.get("adjacency")
        if adjacency is None:
            raise ValueError("CorrLSTMGAT requires a precomputed training-only adjacency matrix via the `adjacency` keyword.")
        architecture_order = str(model_kwargs.get("architecture_order", "lstm_then_gat"))
        usd_idx = int(model_kwargs.get("usd_idx", 0))
        return CorrLSTMGAT(
            input_dim=input_dim,
            n_ccy=n_ccy,
            adjacency=adjacency,
            hidden=max(32, hidden),
            dropout=dropout,
            architecture_order=architecture_order,
            usd_idx=usd_idx,
        )
    if name in {"fxrp", "fxir_edge_gnn"}:
        usd_idx = int(model_kwargs.get("usd_idx", 0))
        num_layers = int(model_kwargs.get("num_layers", 3))
        return FXRP(input_dim, n_ccy, hidden=max(32, hidden), dropout=dropout, num_layers=num_layers, usd_idx=usd_idx)
    if name == "zero_return":
        return ZeroReturnBaseline(n_ccy)
    if name == "random_walk":
        return RandomWalkBaseline(n_ccy)
    raise ValueError(f"Unsupported baseline model: {model_name}")
