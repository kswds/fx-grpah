"""
FX Strength GNN Models — NeurIPS/ICML Research Grade

Architecture family:
  StrengthGNN      (FXStrengthGNN)   : Original baseline (GRU → GNN → Hetero-Macro → USD-pin)
  MACROHyGraph     (Ours)            : [OURS] Currency-Specific Macro Cross-Attention hybrid graph
  FiLMHyGraph      (FiLMHyGraph)     : FiLM macro conditioning hybrid graph
  PureGraphFX      (PureGraphFX)     : Graph-only, no direct branch (ablation)

Ablation models:
  LocalMacroFX     (NoGraphFX)    : No GNN; local GRU + macro only
  PureGraphNoMacro (NoMacroFX)    : No macro conditioning
  StaticHyGraph    (StaticGraphFX): Static (non-dynamic) graph adjacency

Key design fixes:
  1. Identifiability: SINGLE mechanism — USD-pinning (ds_i - ds_USD)
  2. Dynamic graph: pair-specific macro conditioning via E_ij formula
  3. Hybrid decomposition: s = s_direct + s_rel (clear theory ↔ code match)
  4. Heterogeneous A matrix: m_i = A_i ⊙ macro_t (node conditioning)
  5. Global skip: input = concat(N*local_dim, macro_dim) — documented
  6. MACROHyGraph: Currency-Specific Macro Cross-Attention (CMCA) replaces FiLM
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from config import Config


# ============================================================
# Shared Utilities
# ============================================================

def _make_fc_edge_index(N, device):
    """Create fully-connected (no self-loop) edge index for N nodes."""
    edges = [(i, j) for i in range(N) for j in range(N) if i != j]
    return torch.tensor(edges, dtype=torch.long, device=device).T  # [2, N*(N-1)]


def _batched_edge_index(edge_index_single, B, N, device):
    """Expand single-graph edge_index to batched graph. Auto-creates FC if None."""
    if edge_index_single is None:
        edge_index_single = _make_fc_edge_index(N, device)
    E = edge_index_single.size(1)
    edge_b = edge_index_single.repeat(1, B)
    offset = torch.arange(B, device=device).repeat_interleave(E) * N
    return edge_b + offset.unsqueeze(0)


def _usd_pin(ds: torch.Tensor, usd_idx: int) -> torch.Tensor:
    """USD-pinning identifiability: rhat_i = ds_i - ds_USD."""
    return ds - ds[:, usd_idx:usd_idx + 1]


def _hetero_macro_message(macro_t: torch.Tensor, A: torch.Tensor,
                           macro_embed: nn.Linear, B: int, N: int, M: int, H: int):
    """
    Heterogeneous macro node conditioning.

        m_i = A_i ⊙ macro_t   (element-wise sensitivity per factor)
        m_msg[b, i, :] = sum_m A[i, m] * embed(macro_t)[b, m, :]

    A ∈ R^{N × M} — currency-specific macro sensitivity matrix.

    Returns:
        m_msg : [B, N, H]
    """
    u = macro_embed(macro_t).view(B, M, H)     # [B, M, H]
    A_ = A.unsqueeze(0).unsqueeze(-1)           # [1, N, M, 1]
    u_ = u.unsqueeze(1)                         # [B, 1, M, H]
    return (A_ * u_).sum(dim=2)                 # [B, N, H]


# ============================================================
# Original Baseline: FXStrengthGNN
# ============================================================

class StrengthGNN(nn.Module):
    """
    StrengthGNN — GNN-based FX strength baseline.

    Baseline model using a standard GNN for cross-currency spillover.
    Architecture:
      1. GRU per currency (temporal encoding)           → h_i [H]
      2. GNN currency-to-currency spillover              → z_i [H]
      3. Hetero A-matrix macro conditioning             → m_i [H]
      4. Strength head                                   → s_i [1]
      5. USD-pinning: rhat_i = s_i - s_USD
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config    = config
        self.hidden    = config.hidden
        self.n_ccy     = config.n_ccy
        self.macro_dim = config.macro_dim
        self.usd_idx   = config.usd_idx

        self.local_gru = nn.GRU(config.local_dim, config.hidden, batch_first=True)

        if config.gnn_type == "gcn":
            self.ccy_gnn = GCNConv(config.hidden, config.hidden)
        elif config.gnn_type == "sage":
            self.ccy_gnn = SAGEConv(config.hidden, config.hidden)
        elif config.gnn_type == "gat":
            self.ccy_gnn = GATConv(config.hidden, config.hidden,
                                    heads=config.heads, concat=False)
        else:
            raise ValueError(f"Unknown GNN type: {config.gnn_type}")

        # Hetero macro — A ∈ R^{N × M}
        self.macro_embed = nn.Linear(config.macro_dim, config.macro_dim * config.hidden, bias=False)
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim))
        nn.init.normal_(self.A, mean=0.0, std=0.1)

        self.head         = nn.Linear(config.hidden, 1)
        self.use_layer_norm = getattr(config, "use_layer_norm", False)
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden)

    def forward(self, xl, xm, edge_index_single):
        """
        Args:
            xl : [B, L, N, local_dim]
            xm : [B, L, macro_dim]
            edge_index_single : [2, E]
        Returns:
            rhat, ds, z_ccy, m_msg
        """
        B, L, N, D = xl.shape
        device      = xl.device

        # 1. Local temporal encoding
        x  = xl.permute(0, 2, 1, 3).reshape(B * N, L, D)
        _, h = self.local_gru(x)
        h  = h.squeeze(0)   # [B*N, H]

        # 2. GNN currency spillover
        edge_b = _batched_edge_index(edge_index_single, B, N, device)
        z      = self.ccy_gnn(h, edge_b)

        if self.use_layer_norm:
            z = self.layer_norm(z)
        z_ccy = z.view(B, N, self.hidden)   # [B, N, H]

        # 3. Hetero macro: m_i = A_i ⊙ macro_t
        macro_t = xm[:, -1, :]             # [B, M] — last timestep
        m_msg   = _hetero_macro_message(macro_t, self.A, self.macro_embed,
                                         B, N, self.macro_dim, self.hidden)

        # 4. Strength prediction
        z_total = z_ccy + m_msg
        ds      = self.head(z_total).squeeze(-1)   # [B, N]

        # 5. Identifiability: USD-pinning
        rhat = _usd_pin(ds, self.usd_idx)
        return rhat, ds, z_ccy, m_msg


# ============================================================
# Ablation: No GNN
# ============================================================

class LocalMacroFX(nn.Module):
    """
    LocalMacroFX — Local GRU + Macro only (No GNN).

    Ablation: removes the relational graph component entirely.
    Pipeline: GRU → Hetero-Macro → Head → USD-pin
    Demonstrates value of cross-currency graph component.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config    = config
        self.hidden    = config.hidden
        self.n_ccy     = config.n_ccy
        self.macro_dim = config.macro_dim
        self.usd_idx   = config.usd_idx

        self.local_gru   = nn.GRU(config.local_dim, config.hidden, batch_first=True)
        self.macro_embed = nn.Linear(config.macro_dim, config.macro_dim * config.hidden, bias=False)
        self.A           = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim))
        nn.init.normal_(self.A, mean=0.0, std=0.1)
        self.head        = nn.Linear(config.hidden, 1)

    def forward(self, xl, xm, edge_index_single=None):
        B, L, N, D = xl.shape
        x  = xl.permute(0, 2, 1, 3).reshape(B * N, L, D)
        _, h = self.local_gru(x)
        z_ccy = h.squeeze(0).view(B, N, self.hidden)   # [B, N, H]

        macro_t = xm[:, -1, :]
        m_msg   = _hetero_macro_message(macro_t, self.A, self.macro_embed,
                                         B, N, self.macro_dim, self.hidden)

        z_total = z_ccy + m_msg
        ds      = self.head(z_total).squeeze(-1)
        rhat    = _usd_pin(ds, self.usd_idx)
        return rhat, ds, z_ccy, m_msg


# ============================================================
# Ablation: No Macro
# ============================================================

class PureGraphNoMacro(nn.Module):
    """
    PureGraphNoMacro — Graph model without macro conditioning.

    Ablation: removes all macro information.
    Pipeline: GRU → GNN → Head → USD-pin
    Demonstrates the value of macro conditioning.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config  = config
        self.hidden  = config.hidden
        self.n_ccy   = config.n_ccy
        self.usd_idx = config.usd_idx

        self.local_gru = nn.GRU(config.local_dim, config.hidden, batch_first=True)

        if config.gnn_type == "gcn":
            self.ccy_gnn = GCNConv(config.hidden, config.hidden)
        elif config.gnn_type == "sage":
            self.ccy_gnn = SAGEConv(config.hidden, config.hidden)
        else:
            self.ccy_gnn = GATConv(config.hidden, config.hidden,
                                    heads=config.heads, concat=False)

        self.head = nn.Linear(config.hidden, 1)
        # Dummy A for API compatibility
        self.A    = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim),
                                  requires_grad=False)

    def forward(self, xl, xm, edge_index_single):
        B, L, N, D = xl.shape
        x  = xl.permute(0, 2, 1, 3).reshape(B * N, L, D)
        _, h = self.local_gru(x)
        h  = h.squeeze(0)

        edge_b = _batched_edge_index(edge_index_single, B, N, xl.device)
        z      = self.ccy_gnn(h, edge_b)
        z_ccy  = z.view(B, N, self.hidden)

        ds   = self.head(z_ccy).squeeze(-1)
        rhat = _usd_pin(ds, self.usd_idx)

        m_msg = torch.zeros_like(z_ccy)
        return rhat, ds, z_ccy, m_msg


# ============================================================
# Ablation: Static Graph
# ============================================================

class StaticHyGraph(nn.Module):
    """
    StaticHyGraph — Hybrid graph with static (non-dynamic) adjacency.

    Ablation: replaces the macro-conditioned dynamic graph with a
    learnable fixed adjacency matrix.
    Demonstrates the value of dynamic, macro-conditioned edge construction.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config    = config
        self.hidden    = getattr(config, "hybrid_hidden", config.hidden)
        self.n_ccy     = config.n_ccy
        self.macro_dim = config.macro_dim
        self.usd_idx   = config.usd_idx

        self.local_gru  = nn.GRU(config.local_dim, self.hidden, batch_first=True)
        self.ccy_emb    = nn.Parameter(torch.randn(config.n_ccy, self.hidden) * 0.02)

        # Static learnable adjacency
        self.E_static   = nn.Parameter(torch.zeros(config.n_ccy, config.n_ccy))
        self.v_proj     = nn.Linear(self.hidden, self.hidden, bias=False)
        self.msg_proj   = nn.Linear(self.hidden, self.hidden)
        self.node_norm  = nn.LayerNorm(self.hidden)

        self.macro_embed = nn.Linear(config.macro_dim, config.macro_dim * self.hidden, bias=False)
        self.A           = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim))
        nn.init.normal_(self.A, mean=0.0, std=0.1)

        self.head        = nn.Linear(self.hidden, 1)
        self.dropout     = nn.Dropout(getattr(config, "dropout", 0.1))

    def forward(self, xl, xm, edge_index_single=None):
        B, L, N, D = xl.shape

        x  = xl.permute(0, 2, 1, 3).reshape(B * N, L, D)
        _, h = self.local_gru(x)
        h_local = h.squeeze(0).view(B, N, self.hidden) + self.ccy_emb.unsqueeze(0)

        # Static adjacency (row-softmax, no self-loops)
        eye  = torch.eye(N, device=xl.device, dtype=torch.bool).unsqueeze(0)
        E    = self.E_static.unsqueeze(0).masked_fill(eye, float("-inf"))
        A_t  = F.softmax(E, dim=-1)                                # [1, N, N] → broadcast

        v     = self.v_proj(h_local)
        msg   = torch.matmul(A_t, v)
        msg   = self.msg_proj(self.dropout(msg))
        z_ccy = self.node_norm(h_local + msg)

        macro_t = xm[:, -1, :]
        m_msg   = _hetero_macro_message(macro_t, self.A, self.macro_embed,
                                         B, N, self.macro_dim, self.hidden)

        ds   = self.head(z_ccy + m_msg).squeeze(-1)
        rhat = _usd_pin(ds, self.usd_idx)
        return rhat, ds, z_ccy, m_msg


# ============================================================
# Main Model: FiLMHyGraph
# ============================================================

class FiLMHyGraph(nn.Module):
    """
    FiLMHyGraph — Hybrid Graph with FiLM Macro Conditioning.

    Applies Feature-wise Linear Modulation (FiLM) for node-level macro
    conditioning after message passing. Predecessor to MACROHyGraph (Ours).

    Decomposition:
        s_i = s_direct_i + s_rel_i

    Where:
        s_direct_i = W_skip · concat(xl_flat, xm_last)  (global linear branch)
        s_rel_i    = graph-decoded relational embedding   (dynamic graph branch)

    Macro conditioning via FiLM (shared across all currencies):
        z' = (1 + 0.1·tanh(γ(c))) ⊙ z + β(c)

    Dynamic graph construction (pair-specific):
        E_ij = E_static + gate · [(W_q h_i)^T (W_k h_j) / √H
                                   + 0.1 · (h_i − h_j)^T W_m c]

    Heterogeneous macro node conditioning:
        m_i = A_i ⊙ macro_t     (A ∈ R^{N×M})

    Final output: rhat_i = (s_direct_i + s_rel_i) − s_USD  (USD-pinning)
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config    = config
        self.H         = getattr(config, "hybrid_hidden", 64)
        self.n_ccy     = config.n_ccy
        self.local_dim = config.local_dim
        self.macro_dim = config.macro_dim
        self.usd_idx   = config.usd_idx

        top_k_default  = min(getattr(config, "top_k", 6), self.n_ccy - 1) # k 비교
        self.top_k     = max(1, top_k_default)
        self.dropout   = nn.Dropout(getattr(config, "dropout", 0.1))

        # ---- 1. Local temporal encoder (GRU per currency)
        self.local_gru = nn.GRU(self.local_dim, self.H, batch_first=True)
        self.ccy_emb   = nn.Parameter(torch.randn(self.n_ccy, self.H) * 0.02) #currency identity

        # ---- 2. Macro temporal encoder
        self.macro_gru = nn.GRU(self.macro_dim, self.H, batch_first=True)

        # ---- 3. Dynamic graph — pair-specific macro conditioning
        #
        # E_ij = (W_q h_i)^T (W_k h_j) / sqrt(H)
        #       + (h_i - h_j)^T W_m c_t
        #
        self.W_q   = nn.Linear(self.H, self.H, bias=False)  # query projection
        self.W_k   = nn.Linear(self.H, self.H, bias=False)  # key projection
        self.W_m   = nn.Linear(self.H, self.H, bias=False)  # macro diff projection
        self.E_static = nn.Parameter(torch.zeros(self.n_ccy, self.n_ccy)) #static graph backbone
        self.edge_gate = nn.Linear(self.H, 1)    # scalar gate from macro

        # ---- 4. Message passing
        self.W_v      = nn.Linear(self.H, self.H, bias=False)
        self.msg_proj = nn.Linear(self.H, self.H)
        self.node_norm = nn.LayerNorm(self.H)

        # ---- 5. Node-level macro conditioning (FiLM + Hetero-A)
        #
        # FiLM: z' = (1 + γ(c_t)) ⊙ z + β(c_t)
        #
        self.film_gamma    = nn.Linear(self.H, self.H)
        self.film_beta     = nn.Linear(self.H, self.H)

        # Hetero A-matrix: m_i = A_i ⊙ macro_t
        # A ∈ R^{N × M}
        self.A           = nn.Parameter(torch.zeros(self.n_ccy, self.macro_dim))
        nn.init.normal_(self.A, mean=0.0, std=0.1)
        self.macro_embed = nn.Linear(self.macro_dim, self.macro_dim * self.H, bias=False)

        # ---- 6. Graph (relational) branch decoder — currency-specific
        self.head_rel_w = nn.Parameter(torch.randn(self.n_ccy, self.H) * 0.02)
        self.head_rel_b = nn.Parameter(torch.zeros(self.n_ccy))

        # ---- 7. Direct (linear) branch — Global skip
        #
        # Input: concat(xl[:,-1].flatten, xm[:,-1])
        # Dim  : N * local_dim + macro_dim   (explicitly: n_ccy * local_dim + macro_dim)
        #
        global_in_dim = self.n_ccy * self.local_dim + self.macro_dim
        self.global_skip = nn.Linear(global_in_dim, self.n_ccy)
        nn.init.normal_(self.global_skip.weight, std=0.01)
        nn.init.zeros_(self.global_skip.bias)

        # Macro-gated skip scale
        self.skip_gate  = nn.Linear(self.H, self.n_ccy)
        self.skip_scale = nn.Parameter(torch.tensor(0.5))

        # For debugging / visualization
        self.last_adj = None

    # ----------------------------------------------------------

    def _encode_local(self, xl):
        """xl: [B,L,N,D] → h_local: [B,N,H]"""
        B, L, N, D = xl.shape
        x = xl.permute(0, 2, 1, 3).reshape(B * N, L, D)
        _, h = self.local_gru(x)
        h = h.squeeze(0).view(B, N, self.H)
        return h + self.ccy_emb.unsqueeze(0)

    def _encode_macro(self, xm):
        """xm: [B,L,M] → c_macro: [B,H]"""
        _, h = self.macro_gru(xm)
        return h.squeeze(0)

    def _build_dynamic_adj(self, h_local, c_macro):
        """
        Pair-specific macro-conditioned dynamic graph.

        E_ij = (W_q h_i)^T (W_k h_j) / sqrt(H)
              + (h_i - h_j)^T W_m c_t

        Args:
            h_local : [B, N, H]
            c_macro : [B, H]
        Returns:
            A_t : [B, N, N] — row-stochastic sparse adjacency
        """
        B, N, H = h_local.shape
        device  = h_local.device

        q   = self.W_q(h_local)   # [B, N, H]
        k   = self.W_k(h_local)   # [B, N, H]

        # Attention score
        attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(H)  # [B, N, N]

        # Pair-specific macro term: (h_i - h_j)^T W_m c_t
        diff_ij = h_local.unsqueeze(2) - h_local.unsqueeze(1)      # [B, N, N, H]
        c_proj  = self.W_m(c_macro)                                  # [B, H]
        c_exp   = c_proj.view(B, 1, 1, H).expand(B, N, N, H)
        macro_pair = (diff_ij * c_exp).sum(dim=-1)                  # [B, N, N]

        # Gate from macro (scalar per batch)
        gate = torch.sigmoid(self.edge_gate(c_macro)).view(B, 1, 1)  # [B,1,1]

        # Combine
        E_t  = self.E_static.unsqueeze(0) + gate * (attn + 0.1 * macro_pair)

        # Remove self-loops
        eye  = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        E_t  = E_t.masked_fill(eye, float("-inf"))

        # Top-k sparsification
        if self.top_k < N:
            _, topk_idx = torch.topk(E_t, k=self.top_k, dim=-1)
            sparse_mask = torch.zeros_like(E_t, dtype=torch.bool)
            sparse_mask.scatter_(-1, topk_idx, True)
            E_t = E_t.masked_fill(~sparse_mask, float("-inf"))

        A_t = F.softmax(E_t, dim=-1)
        A_t = torch.nan_to_num(A_t, nan=0.0)
        return A_t

    def _direct_branch(self, xl, xm, c_macro):
        """
        Direct (linear) branch: s_direct.

        Input : concat(xl_last_flat [B, N*local_dim], xm_last [B, M])
                Total dim = N * local_dim + macro_dim  <-- explicitly documented

        Returns: s_direct [B, N]
        """
        B = xl.size(0)
        xl_flat = xl[:, -1].reshape(B, self.n_ccy * self.local_dim)  # [B, N*D]
        xm_last = xm[:, -1]                                            # [B, M]
        feat    = torch.cat([xl_flat, xm_last], dim=-1)

        ds_raw  = self.global_skip(feat)                               # [B, N]
        gate    = torch.sigmoid(self.skip_gate(c_macro))               # [B, N]
        return self.skip_scale * gate * ds_raw

    def forward(self, xl, xm, edge_index_single=None):
        """
        Args:
            xl : [B, L, N, local_dim]
            xm : [B, L, macro_dim]

        Returns:
            rhat  : [B, N]  predicted USD-relative returns
            ds    : [B, N]  combined latent strength
            z_ccy : [B, N, H] final node embeddings
            m_msg : [B, N, H] macro conditioning messages
        """
        B, L, N, D = xl.shape
        assert N == self.n_ccy

        # --- Encode
        h_local = self._encode_local(xl)    # [B, N, H]
        c_macro = self._encode_macro(xm)    # [B, H]

        # --- Dynamic graph (pair-specific macro conditioning)
        A_t          = self._build_dynamic_adj(h_local, c_macro)
        self.last_adj = A_t.detach()

        # --- Message passing → relational residual
        v     = self.W_v(h_local)
        msg   = torch.matmul(A_t, v)                     # [B, N, H]
        msg   = self.msg_proj(self.dropout(msg))
        z     = self.node_norm(h_local + msg)             # [B, N, H]

        # --- Node-level macro conditioning (FiLM)
        gamma = 1.0 + 0.1 * torch.tanh(self.film_gamma(c_macro)).unsqueeze(1)
        beta  = self.film_beta(c_macro).unsqueeze(1)
        z_ccy = gamma * z + beta
        z_ccy = self.node_norm(z_ccy)                    # [B, N, H]

        # --- Hetero-A macro message: m_i = A_i ⊙ macro_t
        macro_t = xm[:, -1, :]
        m_msg   = _hetero_macro_message(macro_t, self.A, self.macro_embed,
                                         B, N, self.macro_dim, self.H)

        # --- Relational (graph) branch: s_rel
        z_out  = z_ccy + m_msg                                          # [B, N, H]
        s_rel  = (z_out * self.head_rel_w.unsqueeze(0)).sum(-1) + self.head_rel_b.unsqueeze(0)

        # --- Direct branch: s_direct (linear shortcut)
        s_direct = self._direct_branch(xl, xm, c_macro)

        # --- Hybrid combination: s = s_direct + s_rel
        ds    = s_direct + s_rel                         # [B, N]

        # --- Identifiability: USD-pinning (SINGLE mechanism)
        rhat  = _usd_pin(ds, self.usd_idx)
        return rhat, ds, z_ccy, m_msg

    def forward_with_aux(self, xl, xm, edge_index_single=None):
        """Extended forward returning auxiliary direction logits."""
        rhat, ds, z_ccy, m_msg = self.forward(xl, xm, edge_index_single)
        # Auxiliary directional prediction (for aux BCE loss)
        dir_logits = (z_ccy * self.head_rel_w.unsqueeze(0)).sum(-1)
        dir_logits = dir_logits - dir_logits.mean(dim=1, keepdim=True)
        return rhat, ds, z_ccy, m_msg, dir_logits


# ============================================================
# MACROHyGraph — Our Final Model (CMCA replaces FiLM)
# ============================================================

class MACROHyGraph(FiLMHyGraph):
    """
    MACROHyGraph — Macro-Conditioned Hybrid Graph with Currency-Specific
                   Macro Cross-Attention (CMCA).

    ** Our proposed final model. **

    Motivation
    ----------
    FiLMHyGraph applies the SAME global macro scale/shift to ALL currency nodes:
        z' = (1 + γ(c_macro)) ⊙ z + β(c_macro)
    This is problematic because:
      1. Homogeneity: AUD is commodity-sensitive, JPY is safe-haven.
         One shared transform cannot capture this heterogeneity.
      2. Redundancy: c_macro is already used in dynamic graph construction and
         the direct branch, causing gradient interference / overfitting.

    MACROHyGraph replaces FiLM with Currency-Specific Macro Cross-Attention
    (CMCA) in three steps:

    Step 1 — Per-factor macro encoding:
        Each of the M macro scalars is independently projected to H dimensions.
        mf_emb = GELU(W_f2 · GELU(W_f1 · macro_t_i))   [B, M, H]

    Step 2 — Currency-specific cross-attention:
        Each of N currency nodes computes a query over the M macro factor keys.
        Q = W_cq(z)              [B, N, H]  — node query
        K = W_ck(mf_emb)         [B, M, H]  — macro factor key
        V = W_cv(mf_emb)         [B, M, H]  — macro factor value
        attn = softmax(Q @ K.T / √H)   [B, N, M]   ← currency-specific!
        ctx  = attn @ V                [B, N, H]

    Step 3 — Adaptive gate + residual:
        gate  = sigmoid(W_gate(z))     [B, N, H]
        z_ccy = LayerNorm(z + gate ⊙ ctx)

    The gate suppresses macro conditioning in calm regimes and amplifies it
    during macro-driven regime changes — directly addressing the observation
    that FiLM hurts on average but helps in extreme tail events.

    All other components (dynamic graph, message passing, direct branch,
    hetero-A mechanism) are identical to FiLMHyGraph.

    Interpretability
    ----------------
    self.last_cmca_attn: [B, N, M] attention map after each forward pass.
    Rows = currencies, Cols = macro factors.
    High attn[i, f] → currency i responds strongly to macro factor f.
    """

    def __init__(self, config):
        super().__init__(config)
        # film_gamma / film_beta are inherited but unused.

        H = self.H
        M = self.macro_dim
        dropout = getattr(config, 'dropout', 0.1)

        # ── Step 1: per-factor macro MLP (scalar → H) ─────────────────────────
        self.macro_factor_proj = nn.Sequential(
            nn.Linear(1, H),
            nn.GELU(),
            nn.Linear(H, H),
        )

        # ── Step 2: cross-attention projections ───────────────────────────────
        self.cmca_q    = nn.Linear(H, H, bias=False)
        self.cmca_k    = nn.Linear(H, H, bias=False)
        self.cmca_v    = nn.Linear(H, H, bias=False)
        self.cmca_norm = nn.LayerNorm(H)
        self.cmca_drop = nn.Dropout(dropout)

        # ── Step 3: adaptive gate ─────────────────────────────────────────────
        self.cmca_gate = nn.Linear(H, H)
        nn.init.zeros_(self.cmca_gate.bias)    # start near 0.5 gate
        nn.init.normal_(self.cmca_gate.weight, std=0.02)

        # Diagnostic: store last attention weights
        self.last_cmca_attn = None

    # ------------------------------------------------------------------

    def _cmca(self, z, macro_t):
        """
        Currency-Specific Macro Cross-Attention.

        Args:
            z       : [B, N, H]  node embeddings after message passing
            macro_t : [B, M]     last-step macro factor values

        Returns:
            z_ccy   : [B, N, H]  macro-conditioned node embeddings
            attn    : [B, N, M]  interpretable attention map
        """
        B, N, H = z.shape

        # Step 1 — per-factor embedding: [B, M] → [B, M, H]
        mf_emb = self.macro_factor_proj(macro_t.unsqueeze(-1))   # [B, M, H]

        # Step 2 — cross-attention
        q     = self.cmca_q(z)                                    # [B, N, H]
        k     = self.cmca_k(mf_emb)                               # [B, M, H]
        v     = self.cmca_v(mf_emb)                               # [B, M, H]

        scale = math.sqrt(H)
        attn  = torch.softmax(
            torch.bmm(q, k.transpose(1, 2)) / scale, dim=-1)     # [B, N, M]
        attn  = self.cmca_drop(attn)
        ctx   = torch.bmm(attn, v)                                # [B, N, H]

        # Step 3 — adaptive gate + residual + LayerNorm
        gate  = torch.sigmoid(self.cmca_gate(z))                  # [B, N, H]
        z_ccy = self.cmca_norm(z + gate * ctx)                    # [B, N, H]

        return z_ccy, attn

    # ------------------------------------------------------------------

    def forward(self, xl, xm, edge_index_single=None):
        B, L, N, D = xl.shape
        assert N == self.n_ccy

        # Encode (same as FiLMHyGraph)
        h_local = self._encode_local(xl)    # [B, N, H]
        c_macro = self._encode_macro(xm)    # [B, H]

        # Dynamic graph (same as FiLMHyGraph)
        A_t           = self._build_dynamic_adj(h_local, c_macro)
        self.last_adj = A_t.detach()

        # Message passing (same as FiLMHyGraph)
        v   = self.W_v(h_local)
        msg = torch.matmul(A_t, v)
        msg = self.msg_proj(self.dropout(msg))
        z   = self.node_norm(h_local + msg)

        # ── Ours: Currency-Specific Macro Cross-Attention ────────────────────
        macro_t = xm[:, -1, :]                                    # [B, M]
        z_ccy, attn = self._cmca(z, macro_t)
        self.last_cmca_attn = attn.detach()
        # ─────────────────────────────────────────────────────────────────────

        # Hetero-A macro message (same as FiLMHyGraph)
        m_msg = _hetero_macro_message(macro_t, self.A, self.macro_embed,
                                      B, N, self.macro_dim, self.H)

        # Relational branch
        z_out  = z_ccy + m_msg
        s_rel  = (z_out * self.head_rel_w.unsqueeze(0)).sum(-1) + self.head_rel_b.unsqueeze(0)

        # Direct branch (same as FiLMHyGraph)
        s_direct = self._direct_branch(xl, xm, c_macro)

        # Hybrid combination + USD-pinning
        ds   = s_direct + s_rel
        rhat = _usd_pin(ds, self.usd_idx)
        return rhat, ds, z_ccy, m_msg


# ============================================================
# PureGraphFX — Graph-only (no direct branch) — ablation
# ============================================================

class PureGraphFX(nn.Module):
    """
    PureGraphFX — Dynamic graph model without the direct linear branch.

    Ablation of MACROHyGraph / FiLMHyGraph: removes the global skip
    (direct) branch so that s = s_rel only (purely relational).
    Demonstrates the empirical importance of the direct branch.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config    = config
        self.H         = getattr(config, "hybrid_hidden", 64)
        self.n_ccy     = config.n_ccy
        self.local_dim = config.local_dim
        self.macro_dim = config.macro_dim
        self.usd_idx   = config.usd_idx
        top_k          = min(getattr(config, "top_k", 4), self.n_ccy - 1)
        self.top_k     = max(1, top_k)
        self.dropout   = nn.Dropout(getattr(config, "dropout", 0.1))

        self.local_gru = nn.GRU(self.local_dim, self.H, batch_first=True)
        self.ccy_emb   = nn.Parameter(torch.randn(self.n_ccy, self.H) * 0.02)
        self.macro_gru = nn.GRU(self.macro_dim, self.H, batch_first=True)

        self.W_q       = nn.Linear(self.H, self.H, bias=False)
        self.W_k       = nn.Linear(self.H, self.H, bias=False)
        self.W_m       = nn.Linear(self.H, self.H, bias=False)
        self.E_static  = nn.Parameter(torch.zeros(self.n_ccy, self.n_ccy))
        self.edge_gate = nn.Linear(self.H, 1)

        self.W_v       = nn.Linear(self.H, self.H, bias=False)
        self.msg_proj  = nn.Linear(self.H, self.H)
        self.node_norm = nn.LayerNorm(self.H)

        self.film_gamma   = nn.Linear(self.H, self.H)
        self.film_beta    = nn.Linear(self.H, self.H)

        self.A            = nn.Parameter(torch.zeros(self.n_ccy, self.macro_dim))
        nn.init.normal_(self.A, mean=0.0, std=0.1)
        self.macro_embed  = nn.Linear(self.macro_dim, self.macro_dim * self.H, bias=False)

        self.strength_head = nn.Sequential(
            nn.Linear(self.H, self.H), nn.ReLU(),
            nn.Dropout(getattr(config, "dropout", 0.1)),
            nn.Linear(self.H, 1)
        )
        self.ccy_bias  = nn.Parameter(torch.zeros(self.n_ccy))
        self.last_adj  = None

    def _encode_local(self, xl):
        B, L, N, D = xl.shape
        x = xl.permute(0, 2, 1, 3).reshape(B * N, L, D)
        _, h = self.local_gru(x)
        return h.squeeze(0).view(B, N, self.H) + self.ccy_emb.unsqueeze(0)

    def _encode_macro(self, xm):
        _, h = self.macro_gru(xm)
        return h.squeeze(0)

    def _build_dynamic_adj(self, h_local, c_macro):
        B, N, H = h_local.shape
        device  = h_local.device
        q   = self.W_q(h_local)
        k   = self.W_k(h_local)
        attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(H)

        diff_ij    = h_local.unsqueeze(2) - h_local.unsqueeze(1)
        c_proj     = self.W_m(c_macro).view(B, 1, 1, H).expand(B, N, N, H)
        macro_pair = (diff_ij * c_proj).sum(-1)

        gate = torch.sigmoid(self.edge_gate(c_macro)).view(B, 1, 1)
        E_t  = self.E_static.unsqueeze(0) + gate * (attn + 0.1 * macro_pair)

        eye  = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        E_t  = E_t.masked_fill(eye, float("-inf"))

        if self.top_k < N:
            _, idx = torch.topk(E_t, k=self.top_k, dim=-1)
            mask   = torch.zeros_like(E_t, dtype=torch.bool)
            mask.scatter_(-1, idx, True)
            E_t    = E_t.masked_fill(~mask, float("-inf"))

        A_t = F.softmax(E_t, dim=-1)
        return torch.nan_to_num(A_t, nan=0.0)

    def forward(self, xl, xm, edge_index_single=None):
        B, L, N, D = xl.shape
        h_local = self._encode_local(xl)
        c_macro = self._encode_macro(xm)

        A_t           = self._build_dynamic_adj(h_local, c_macro)
        self.last_adj = A_t.detach()

        v    = self.W_v(h_local)
        msg  = self.msg_proj(self.dropout(torch.matmul(A_t, v)))
        z    = self.node_norm(h_local + msg)

        gamma = 1.0 + 0.1 * torch.tanh(self.film_gamma(c_macro)).unsqueeze(1)
        beta  = self.film_beta(c_macro).unsqueeze(1)
        z_ccy = self.node_norm(gamma * z + beta)

        macro_t = xm[:, -1, :]
        m_msg   = _hetero_macro_message(macro_t, self.A, self.macro_embed,
                                         B, N, self.macro_dim, self.H)

        ds   = self.strength_head(z_ccy + m_msg).squeeze(-1)
        ds   = ds + self.ccy_bias.unsqueeze(0)
        rhat = _usd_pin(ds, self.usd_idx)
        return rhat, ds, z_ccy, m_msg


# ============================================================
# Backward Compatibility Aliases
# ============================================================

# Version-code aliases (legacy scripts / result JSON keys)
Ours = MACROHyGraph
NoGraphFX       = LocalMacroFX
NoMacroFX       = PureGraphNoMacro
StaticGraphFX   = StaticHyGraph
FXStrengthGNN   = StrengthGNN


# ============================================================
# Baseline Models (for comparison)
# ============================================================

class RandomWalkBaseline(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config  = config
        self.usd_idx = config.usd_idx
        self.dummy   = nn.Parameter(torch.zeros(1))
        self.A       = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim),
                                     requires_grad=False)

    def forward(self, xl, xm, edge_index_single=None):
        B, N = xl.size(0), xl.size(2)
        z    = torch.zeros(B, N, self.config.hidden, device=xl.device)
        return (torch.zeros(B, N, device=xl.device),
                torch.zeros(B, N, device=xl.device),
                z, z)


class LinearBaseline(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config  = config
        self.usd_idx = config.usd_idx
        self.linear  = nn.Linear(config.local_dim + config.macro_dim, 1)
        self.A       = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim),
                                     requires_grad=False)

    def forward(self, xl, xm, edge_index_single=None):
        B, L, N, D = xl.shape
        xl_last = xl[:, -1]                         # [B, N, D]
        xm_last = xm[:, -1].unsqueeze(1).expand(-1, N, -1)
        x  = torch.cat([xl_last, xm_last], dim=-1)
        ds = self.linear(x).squeeze(-1)
        rhat = _usd_pin(ds, self.usd_idx)
        h = torch.zeros(B, N, self.config.hidden, device=xl.device)
        return rhat, ds, h, h


class MLPBaseline(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config  = config
        self.usd_idx = config.usd_idx
        D = config.local_dim + config.macro_dim
        self.mlp = nn.Sequential(
            nn.Linear(D, config.hidden), nn.ReLU(),
            nn.Linear(config.hidden, config.hidden // 2), nn.ReLU(),
            nn.Linear(config.hidden // 2, 1)
        )
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim),
                               requires_grad=False)

    def forward(self, xl, xm, edge_index_single=None):
        B, L, N, D = xl.shape
        xl_last = xl[:, -1]
        xm_last = xm[:, -1].unsqueeze(1).expand(-1, N, -1)
        x  = torch.cat([xl_last, xm_last], dim=-1)
        ds = self.mlp(x).squeeze(-1)
        rhat = _usd_pin(ds, self.usd_idx)
        h = torch.zeros(B, N, self.config.hidden, device=xl.device)
        return rhat, ds, h, h


class LSTMBaseline(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config  = config
        self.usd_idx = config.usd_idx
        self.hidden  = config.hidden
        D = config.local_dim + config.macro_dim
        self.lstm = nn.LSTM(D, config.hidden, batch_first=True)
        self.head = nn.Linear(config.hidden, 1)
        self.A    = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim),
                                  requires_grad=False)

    def forward(self, xl, xm, edge_index_single=None):
        B, L, N, D = xl.shape
        xm_exp = xm.unsqueeze(2).expand(-1, -1, N, -1)
        x = torch.cat([xl, xm_exp], dim=-1).permute(0, 2, 1, 3).reshape(B * N, L, -1)
        _, (h, _) = self.lstm(x)
        h  = h.squeeze(0)
        ds = self.head(h).squeeze(-1).view(B, N)
        z_ccy = h.view(B, N, self.hidden)
        rhat  = _usd_pin(ds, self.usd_idx)
        return rhat, ds, z_ccy, torch.zeros_like(z_ccy)


# ============================================================
# Model Registries
# ============================================================

BASELINE_MODELS = {
    "random_walk": RandomWalkBaseline,
    "linear":      LinearBaseline,
    "mlp":         MLPBaseline,
    "lstm":        LSTMBaseline,
    "ours":        StrengthGNN,
}

ABLATION_MODELS = {
    # ── Proposed model ───────────────────────────────────────────────────────
    "MACROHyGraph":     MACROHyGraph,      # [OURS] CMCA — Currency-Specific Macro Cross-Attention
    # ── Ablations of MACROHyGraph ────────────────────────────────────────────
    "FiLMHyGraph":      FiLMHyGraph,       # FiLM instead of CMCA (shared macro transform)
    "PureGraphFX":      PureGraphFX,       # No direct branch (graph-only)
    "LocalMacroFX":     LocalMacroFX,      # No GNN (local + macro only)
    "PureGraphNoMacro": PureGraphNoMacro,  # No macro conditioning
    "StaticHyGraph":    StaticHyGraph,     # Static adjacency (no dynamic graph)
    "StrengthGNN":      StrengthGNN,       # Original GNN baseline
    # ── Public model keys ────────────────────────────────────────────────────
    "Ours":        MACROHyGraph,
    "FiLMHyGraph": FiLMHyGraph,
    "PureGraphFX": PureGraphFX,
    "NoGraph":     LocalMacroFX,
    "NoMacro":     PureGraphNoMacro,
    "StaticGraph": StaticHyGraph,
    "GNN":         StrengthGNN,
}

ALL_MODELS = {**BASELINE_MODELS, **ABLATION_MODELS}


def create_model(name: str, config: Config) -> nn.Module:
    """Factory function. name must be in ALL_MODELS."""
    if name not in ALL_MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(ALL_MODELS.keys())}")
    return ALL_MODELS[name](config)
