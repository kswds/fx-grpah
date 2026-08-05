from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def usd_pin(ds: torch.Tensor, usd_idx: int = 0) -> torch.Tensor:
    return ds - ds[:, usd_idx:usd_idx + 1]


def topk_softmax(logits: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    if logits.numel() == 0:
        return logits
    n = logits.size(dim)
    if n <= 1:
        return torch.zeros_like(logits)
    k = max(1, min(int(k), n - 1))
    topk_vals, topk_idx = torch.topk(logits, k=k, dim=dim)
    sparse = torch.full_like(logits, float("-inf"))
    sparse.scatter_(dim, topk_idx, topk_vals)
    probs = F.softmax(sparse, dim=dim)
    return torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)


def spectral_norm_batch(A: torch.Tensor, n_iter: int = 3) -> torch.Tensor:
    b, n, _ = A.shape
    if n == 0:
        return A.new_zeros((b,))
    u = F.normalize(torch.ones(b, n, 1, device=A.device, dtype=A.dtype), dim=1)
    for _ in range(max(1, n_iter)):
        v = F.normalize(torch.bmm(A.transpose(1, 2), u), dim=1)
        u = F.normalize(torch.bmm(A, v), dim=1)
    sigma = torch.bmm(u.transpose(1, 2), torch.bmm(A, v)).squeeze(-1).squeeze(-1)
    return sigma.abs()


class SequenceComponentEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.input_dim = int(max(0, input_dim))
        self.hidden = hidden
        if self.input_dim > 0:
            self.input_proj = nn.Sequential(nn.Linear(self.input_dim, hidden), nn.GELU(), nn.Dropout(dropout))
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.attn = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.input_dim == 0 or x.size(-1) == 0:
            b, _, n, _ = x.shape
            zeros = torch.zeros(b, n, self.hidden, device=x.device, dtype=x.dtype)
            return {"repr": zeros}
        b, l, n, _ = x.shape
        z = self.input_proj(x)
        seq = z.permute(0, 2, 1, 3).reshape(b * n, l, self.hidden)
        seq_out, _ = self.gru(seq)
        attn = torch.nan_to_num(F.softmax(self.attn(seq_out).squeeze(-1), dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        pooled = torch.sum(attn.unsqueeze(-1) * seq_out, dim=1)
        pooled = self.norm(pooled).reshape(b, n, self.hidden)
        return {"repr": pooled}


class GlobalSequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.input_dim = int(max(0, input_dim))
        self.hidden = hidden
        if self.input_dim > 0:
            self.input_proj = nn.Sequential(nn.Linear(self.input_dim, hidden), nn.GELU(), nn.Dropout(dropout))
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.attn = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.input_dim == 0 or x.size(-1) == 0:
            return {"repr": torch.zeros(x.size(0), self.hidden, device=x.device, dtype=x.dtype)}
        z = self.input_proj(x)
        seq_out, _ = self.gru(z)
        attn = torch.nan_to_num(F.softmax(self.attn(seq_out).squeeze(-1), dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        pooled = torch.sum(attn.unsqueeze(-1) * seq_out, dim=1)
        return {"repr": self.norm(pooled)}


class StableRelationalScoreGraphFX(nn.Module):
    def __init__(
        self,
        n_ccy: int,
        currency_names: List[str],
        usd_idx: int = 0,
        dims: Optional[Dict] = None,
        hidden: int = 48,
        top_k: int = 2,
        dropout: float = 0.25,
        use_rate: bool = True,
        use_equity: bool = True,
        use_macro: bool = True,
        use_countrymacro: bool = True,
        use_component_gates: bool = True,
        component_gate_type: str = "sigmoid",
        use_dynamic_graph: bool = True,
        use_static_graph: bool = True,
        use_graph_gate: bool = True,
        use_signed_edges: bool = True,
        use_low_rank_delta: bool = True,
        graph_rank: int = 8,
        edge_dropout: float = 0.05,
        spectral_bound: float = 1.0,
        use_graph: bool = True,
        use_node_attention_edges: bool = True,
        use_shared_global_encoder: bool = False,
        use_direct_usd_output: bool = False,
    ):
        super().__init__()
        self.n_ccy = n_ccy
        self.currency_names = list(currency_names)
        self.usd_idx = usd_idx
        self.hidden = hidden
        self.top_k = max(1, min(int(top_k), max(1, n_ccy - 1)))
        self.use_rate = use_rate
        self.use_equity = use_equity
        self.use_macro = use_macro
        self.use_countrymacro = use_countrymacro
        self.use_component_gates = use_component_gates
        self.component_gate_type = component_gate_type
        self.use_dynamic_graph = use_dynamic_graph
        self.use_static_graph = use_static_graph
        self.use_graph_gate = use_graph_gate
        self.use_signed_edges = use_signed_edges
        self.use_low_rank_delta = use_low_rank_delta
        self.graph_rank = max(1, int(graph_rank))
        self.spectral_bound = float(spectral_bound)
        self.use_graph = use_graph
        self.use_node_attention_edges = use_node_attention_edges
        self.use_shared_global_encoder = use_shared_global_encoder
        self.use_direct_usd_output = use_direct_usd_output
        dims = dims or {}

        self.local_encoder = SequenceComponentEncoder(int(dims.get("local", 0)), hidden, dropout)
        self.rate_encoder = SequenceComponentEncoder(int(dims.get("rate", 0)), hidden, dropout)
        self.equity_encoder = SequenceComponentEncoder(int(dims.get("equity", 0)), hidden, dropout)
        self.countrymacro_encoder = SequenceComponentEncoder(int(dims.get("countrymacro", 0)), hidden, dropout)
        self.global_encoder = GlobalSequenceEncoder(int(dims.get("global", 0)), hidden, dropout)
        self.macro_regime_encoder = None if use_shared_global_encoder else GlobalSequenceEncoder(int(dims.get("global", 0)), hidden, dropout)

        self.currency_embedding = nn.Parameter(torch.randn(n_ccy, hidden) * 0.02)
        self.currency_bias = nn.Parameter(torch.zeros(n_ccy))
        self.static_edge_logits = nn.Parameter(torch.randn(n_ccy, n_ccy) * 0.02)
        self.node_left_embedding = nn.Parameter(torch.randn(n_ccy, self.graph_rank) * 0.02)
        self.node_right_embedding = nn.Parameter(torch.randn(n_ccy, self.graph_rank) * 0.02)

        self.rate_proj = nn.Linear(hidden, hidden)
        self.equity_proj = nn.Linear(hidden, hidden)
        self.country_proj = nn.Linear(hidden, hidden)
        self.global_proj = nn.Linear(hidden, hidden)
        self.global_direct_head = nn.Identity() if not use_shared_global_encoder else nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.global_regime_head = nn.Identity() if not use_shared_global_encoder else nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.node_norm = nn.LayerNorm(hidden)
        self.rel_norm = nn.LayerNorm(hidden)
        self.final_norm = nn.LayerNorm(hidden)
        self.edge_dropout = nn.Dropout(edge_dropout)

        self.local_head = nn.Linear(hidden, 1)
        self.rate_head = nn.Linear(hidden, 1)
        self.equity_head = nn.Linear(hidden, 1)
        self.countrymacro_head = nn.Linear(hidden, 1)
        self.global_macro_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.rel_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.rel_value_head = nn.Linear(hidden, 1)
        self.direct_usd_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

        self.component_gate_mlp = nn.Sequential(nn.Linear(hidden * 7, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 5))
        self.graph_gate_mlp = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.rank_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, self.graph_rank))
        self.Wq = nn.Linear(hidden, hidden, bias=False)
        self.Wk = nn.Linear(hidden, hidden, bias=False)
        self.Wv = nn.Linear(hidden, hidden, bias=False)
        self.message_proj = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))

    def _normalize_static_adj(self) -> torch.Tensor:
        logits = self.static_edge_logits
        n = logits.size(0)
        eye = torch.eye(n, device=logits.device, dtype=torch.bool)
        masked = logits.masked_fill(eye, float("-inf"))
        if self.use_signed_edges:
            sign = torch.tanh(masked)
            mag_logits = logits.abs().masked_fill(eye, float("-inf"))
            return torch.nan_to_num(topk_softmax(mag_logits, self.top_k, -1) * sign, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.nan_to_num(topk_softmax(masked, self.top_k, -1), nan=0.0, posinf=0.0, neginf=0.0)

    def _component_gates(self, h_local, h_rate, h_equity, h_country, h_global_node, h_rel) -> Dict[str, torch.Tensor]:
        b, n, _ = h_local.shape
        if not self.use_component_gates:
            ones = torch.ones(b, n, device=h_local.device, dtype=h_local.dtype)
            return {"local": ones, "rate": ones, "equity": ones, "macro": ones, "rel": ones}
        gate_in = torch.cat([h_local, h_rate, h_equity, h_country, h_global_node, h_rel, self.currency_embedding.unsqueeze(0).expand(b, -1, -1)], dim=-1)
        logits = self.component_gate_mlp(gate_in)
        if self.component_gate_type == "softmax":
            gates = torch.nan_to_num(F.softmax(logits, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            gates = torch.sigmoid(logits)
        return dict(zip(["local", "rate", "equity", "macro", "rel"], gates.unbind(dim=-1)))

    def graph_regularization(self, A: torch.Tensor, A_prev: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        static_norm = self._normalize_static_adj()
        reg_smooth = A.new_tensor(0.0) if A_prev is None or tuple(A_prev.shape) != tuple(A.shape) else ((A - A_prev) ** 2).mean()
        reg_static = ((A - static_norm.unsqueeze(0)) ** 2).mean()
        reg_sparse = A.abs().mean() if self.use_signed_edges else A.mean()
        reg_spectral = torch.relu(spectral_norm_batch(A.abs()) - self.spectral_bound).pow(2).mean()
        return {"smoothness": reg_smooth, "static_deviation": reg_static, "sparsity": reg_sparse, "spectral": reg_spectral}

    def _build_graph(
        self,
        h: torch.Tensor,
        h_global: torch.Tensor,
        A_prev: Optional[torch.Tensor] = None,
        A_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, n, hdim = h.shape
        zeros_adj = torch.zeros(b, n, n, device=h.device, dtype=h.dtype)
        static_norm = self._normalize_static_adj()
        if not self.use_graph:
            reg = self.graph_regularization(zeros_adj, A_prev=A_prev)
            zeros_node = torch.zeros(b, n, self.hidden, device=h.device, dtype=h.dtype)
            zeros_score = torch.zeros(b, n, device=h.device, dtype=h.dtype)
            return {"adj": zeros_adj, "adj_static": static_norm, "adj_dynamic_delta": zeros_adj, "edge_logits": zeros_adj, "edge_contrib": zeros_adj, "graph_gate": zeros_score, "rel_repr": zeros_node, "rel_score": zeros_score, "regularization": reg}
        edge_logits = torch.zeros(b, n, n, device=h.device, dtype=h.dtype)
        if self.use_static_graph:
            edge_logits = edge_logits + self.static_edge_logits.unsqueeze(0)
        delta_A = zeros_adj
        if self.use_dynamic_graph:
            if self.use_low_rank_delta:
                rank_state = self.rank_mlp(h_global)
                delta_A = torch.einsum("ir,br,jr->bij", self.node_left_embedding, rank_state, self.node_right_embedding)
            if self.use_node_attention_edges:
                attn = torch.matmul(self.Wq(h), self.Wk(h).transpose(1, 2)) / math.sqrt(max(1, hdim))
                edge_logits = edge_logits + attn
            edge_logits = edge_logits + delta_A
        eye = torch.eye(n, device=h.device, dtype=torch.bool).unsqueeze(0)
        edge_logits = edge_logits.masked_fill(eye, float("-inf"))
        if self.use_signed_edges:
            edge_sign = torch.tanh(edge_logits)
            edge_mag = topk_softmax(edge_logits.masked_fill(eye, 0.0).abs().masked_fill(eye, float("-inf")), self.top_k, -1)
            A = edge_mag * edge_sign
        else:
            A = topk_softmax(edge_logits, self.top_k, -1)
        A = torch.nan_to_num(self.edge_dropout(A), nan=0.0, posinf=0.0, neginf=0.0)
        A_effective = A
        if A_override is not None and tuple(A_override.shape) == tuple(A.shape):
            A_effective = torch.nan_to_num(A_override.to(device=A.device, dtype=A.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        msg = torch.einsum("bij,bjh->bih", A_effective, self.Wv(h))
        h_rel = self.rel_norm(h + self.message_proj(msg))
        raw_c_rel = self.rel_head(h_rel).squeeze(-1)
        h_global_node = h_global.unsqueeze(1).expand(-1, n, -1)
        g_graph = torch.sigmoid(self.graph_gate_mlp(torch.cat([h, h_global_node, h_rel], dim=-1))).squeeze(-1) if self.use_graph_gate else torch.ones(b, n, device=h.device, dtype=h.dtype)
        c_rel = g_graph * raw_c_rel
        edge_contrib = A_effective * self.rel_value_head(h).squeeze(-1)[:, None, :]
        return {
            "adj": A_effective,
            "adj_current": A,
            "adj_static": static_norm,
            "adj_dynamic_delta": delta_A,
            "edge_logits": edge_logits,
            "edge_contrib": edge_contrib,
            "graph_gate": g_graph,
            "rel_repr": h_rel,
            "rel_score": c_rel,
            "regularization": self.graph_regularization(A_effective, A_prev=A_prev),
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_local = batch["x_local"]
        x_rate = batch["x_rate"]
        x_equity = batch["x_equity"]
        x_country = batch["x_countrymacro"]
        x_global = batch["x_global"]
        A_prev = batch.get("A_prev")
        A_override = batch.get("A_override")
        disable_relational_path = bool(batch.get("disable_relational_path", False))
        local_out = self.local_encoder(x_local)
        rate_out = self.rate_encoder(x_rate)
        equity_out = self.equity_encoder(x_equity)
        country_out = self.countrymacro_encoder(x_country)
        global_out = self.global_encoder(x_global)
        regime_out = None if self.use_shared_global_encoder else self.macro_regime_encoder(x_global)
        h_local = local_out["repr"]
        h_rate = rate_out["repr"] if self.use_rate else torch.zeros_like(h_local)
        h_equity = equity_out["repr"] if self.use_equity else torch.zeros_like(h_local)
        h_country = country_out["repr"] if (self.use_macro and self.use_countrymacro) else torch.zeros_like(h_local)
        h_global_base = global_out["repr"] if self.use_macro else torch.zeros(x_local.size(0), self.hidden, device=x_local.device, dtype=x_local.dtype)
        if self.use_macro and self.use_shared_global_encoder:
            h_global = self.global_direct_head(h_global_base)
            h_regime = self.global_regime_head(h_global_base)
        elif self.use_macro:
            h_global = self.global_direct_head(h_global_base)
            h_regime = regime_out["repr"]
        else:
            h_global = h_global_base
            h_regime = h_global_base
        h_global_node = h_global.unsqueeze(1).expand(-1, self.n_ccy, -1)
        h = h_local + self.currency_embedding.unsqueeze(0)
        if self.use_rate:
            h = h + self.rate_proj(h_rate)
        if self.use_equity:
            h = h + self.equity_proj(h_equity)
        if self.use_macro and self.use_countrymacro:
            h = h + self.country_proj(h_country)
        if self.use_macro:
            h = h + self.global_proj(h_global_node)
        h = self.node_norm(h)
        graph_out = self._build_graph(h, h_regime, A_prev=A_prev, A_override=A_override)
        if disable_relational_path:
            graph_out["rel_repr"] = torch.zeros_like(graph_out["rel_repr"])
            graph_out["rel_score"] = torch.zeros_like(graph_out["rel_score"])
            graph_out["edge_contrib"] = torch.zeros_like(graph_out["edge_contrib"])
        c_local = self.local_head(h_local).squeeze(-1)
        c_rate = self.rate_head(h_rate).squeeze(-1) if self.use_rate else torch.zeros_like(c_local)
        c_equity = self.equity_head(h_equity).squeeze(-1) if self.use_equity else torch.zeros_like(c_local)
        c_macro_country = self.countrymacro_head(h_country).squeeze(-1) if (self.use_macro and self.use_countrymacro) else torch.zeros_like(c_local)
        c_macro_global = self.global_macro_head(torch.cat([h_global_node, self.currency_embedding.unsqueeze(0).expand(x_local.size(0), -1, -1)], dim=-1)).squeeze(-1) if self.use_macro else torch.zeros_like(c_local)
        c_macro = c_macro_country + c_macro_global
        c_rel = graph_out["rel_score"] if self.use_graph else torch.zeros_like(c_local)
        gates = self._component_gates(h_local, h_rate, h_equity, h_country, h_global_node, graph_out["rel_repr"])
        ds = gates["local"] * c_local + gates["rate"] * c_rate + gates["equity"] * c_equity + gates["macro"] * c_macro + gates["rel"] * c_rel + self.currency_bias.unsqueeze(0)
        node_repr = self.final_norm(h + graph_out["rel_repr"])
        if self.use_direct_usd_output:
            rhat = self.direct_usd_head(node_repr).squeeze(-1)
            rhat[:, self.usd_idx] = 0.0
        else:
            rhat = usd_pin(ds, self.usd_idx)
        return {
            "rhat": rhat,
            "ds": ds,
            "components": {"local": c_local, "rate": c_rate, "equity": c_equity, "macro": c_macro, "macro_global": c_macro_global, "macro_country": c_macro_country, "rel": c_rel},
            "component_gates": gates,
            "adj": graph_out["adj"],
            "adj_current": graph_out.get("adj_current", graph_out["adj"]),
            "adj_static": graph_out["adj_static"],
            "adj_dynamic_delta": graph_out["adj_dynamic_delta"],
            "edge_logits": graph_out["edge_logits"],
            "edge_contrib": graph_out["edge_contrib"],
            "graph_gate": graph_out["graph_gate"],
            "node_repr": node_repr,
            "regularization": graph_out["regularization"],
        }


def stable_scoregraph_loss(out, y, non_usd_mask, lambda_dir=0.15, lambda_rank=0.05, lambda_component=1e-4, lambda_smooth=1e-3, lambda_static=1e-3, lambda_sparse=1e-4, lambda_spectral=1e-3, q80_abs_y=1.0):
    pred = out["rhat"][:, non_usd_mask]
    target = y[:, non_usd_mask]
    loss_mse = F.mse_loss(pred, target)
    w = torch.clamp(target.abs() / max(float(q80_abs_y), 1e-6), max=1.0)
    loss_dir = (w * F.softplus(-(pred * target))).mean()
    if pred.size(1) < 2:
        loss_rank = pred.new_tensor(0.0)
    else:
        loss_rank = F.softplus(-((pred[:, :, None] - pred[:, None, :]) * (target[:, :, None] - target[:, None, :]))).mean()
    loss_component = sum(v.pow(2).mean() for v in out["components"].values())
    reg = out["regularization"]
    loss = loss_mse + lambda_dir * loss_dir + lambda_rank * loss_rank + lambda_component * loss_component + lambda_smooth * reg["smoothness"] + lambda_static * reg["static_deviation"] + lambda_sparse * reg["sparsity"] + lambda_spectral * reg["spectral"]
    return loss


def small_return_cls_loss(out, y_norm, non_usd_mask_t, threshold: float, lambda_component=1e-4, lambda_smooth=1e-3, lambda_static=1e-3, lambda_sparse=1e-4, lambda_spectral=1e-3):
    pred = out["rhat"][:, non_usd_mask_t]
    target = y_norm[:, non_usd_mask_t]
    active = target.abs() >= threshold
    loss_cls = F.softplus(-(pred[active] * torch.sign(target[active]))).mean() if active.any() else F.mse_loss(pred, target)
    reg = out["regularization"]
    loss_component = sum(v.pow(2).mean() for v in out["components"].values())
    return loss_cls + lambda_component * loss_component + lambda_smooth * reg["smoothness"] + lambda_static * reg["static_deviation"] + lambda_sparse * reg["sparsity"] + lambda_spectral * reg["spectral"]


def create_relational_model(model_name: str, config: Dict) -> StableRelationalScoreGraphFX:
    name = model_name.lower()
    kwargs = dict(config)
    kwargs.update({"use_graph": True, "use_dynamic_graph": True, "use_static_graph": True, "use_signed_edges": True, "use_rate": True, "use_equity": True, "use_macro": True, "use_countrymacro": True, "use_component_gates": True})
    if name == "oursmain":
        pass
    elif name == "oursmain_directusd":
        kwargs["use_direct_usd_output"] = True
    elif name == "oursmain2":
        kwargs["use_low_rank_delta"] = False
        kwargs["use_node_attention_edges"] = True
    elif name == "oursmain_nographgate":
        kwargs["use_graph_gate"] = False
    elif name == "oursmain_sharedglobal":
        kwargs["use_shared_global_encoder"] = True
    elif name == "oursmain_nographgate_sharedglobal":
        kwargs["use_graph_gate"] = False
        kwargs["use_shared_global_encoder"] = True
    elif name == "oursmain_static_only":
        kwargs["use_dynamic_graph"] = False
    elif name == "oursmain_static_plus_state":
        kwargs["use_low_rank_delta"] = False
        kwargs["use_node_attention_edges"] = True
    elif name == "oursmain_static_plus_regime":
        kwargs["use_low_rank_delta"] = True
        kwargs["use_node_attention_edges"] = False
    elif name == "oursmain_full_graph":
        pass
    elif name == "oursmain_nographgate_static_plus_state":
        kwargs["use_graph_gate"] = False
        kwargs["use_low_rank_delta"] = False
        kwargs["use_node_attention_edges"] = True
    elif name == "oursmain_nographgate_static_plus_regime":
        kwargs["use_graph_gate"] = False
        kwargs["use_low_rank_delta"] = True
        kwargs["use_node_attention_edges"] = False
    elif name == "oursmain_nographgate_full_graph":
        kwargs["use_graph_gate"] = False
    elif name == "foundation_relational":
        pass
    elif name == "foundation_nograph":
        kwargs["use_graph"] = False
        kwargs["use_dynamic_graph"] = False
        kwargs["use_static_graph"] = False
        kwargs["use_graph_gate"] = False
        kwargs["use_macro"] = False
        kwargs["use_countrymacro"] = False
        kwargs["use_component_gates"] = False
        kwargs["use_direct_usd_output"] = False
    elif name == "foundation_static":
        kwargs["use_dynamic_graph"] = False
    else:
        raise ValueError(f"Unsupported relational model: {model_name}")
    return StableRelationalScoreGraphFX(**kwargs)
