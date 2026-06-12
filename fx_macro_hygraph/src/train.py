"""
Training Utilities — NeurIPS/ICML Research Grade

Loss functions:
  - VICReg-style variance constraint (replaces scale-exploding -Var)
  - Direction-aware BCE loss
  - Optional pairwise ranking loss

Trainer:
  - Train / Val / Test with early stopping (patience=10)
  - Best model checkpoint on validation loss

Metrics:
  - RMSE, MAE, Hit(ccy), Hit(55)
  - Information Coefficient (IC) — Spearman rank correlation
  - Long-short Sharpe ratio
  - Triangle consistency error
  - Macro Usage Ratio (MUR)
"""
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as scipy_stats
from typing import Dict, Tuple, Optional
from config import Config

logger = logging.getLogger(__name__)


# ============================================================
# Loss Functions
# ============================================================

def vicreg_variance_loss(ds: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    VICReg-style variance constraint.

    Instead of -Var(ds) (which encourages scale explosion),
    we penalize when std(ds) < gamma:

        loss_var = mean( relu(gamma - std(ds_per_sample)) )

    This encourages spread without unbounded growth.

    Args:
        ds    : [B, N] currency strengths
        gamma : target minimum std
    """
    std_per_sample = ds.std(dim=1, unbiased=False)          # [B]
    return F.relu(gamma - std_per_sample).mean()


def direction_bce_loss(rhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    BCE loss on sign prediction.

    sign_prob = sigmoid(rhat)
    y_bin     = (y > 0).float()
    loss      = BCE(sign_prob, y_bin)

    Uses logits form for numerical stability.
    """
    y_bin = (y > 0).float()
    return F.binary_cross_entropy_with_logits(rhat, y_bin)


def pairwise_ranking_loss(pred: torch.Tensor, target: torch.Tensor,
                          margin: float = 0.0) -> torch.Tensor:
    """
    Pairwise ranking loss over currency strengths.

    For each pair (i, j) where target[i] > target[j]:
        loss += relu(margin - (pred[i] - pred[j]))

    Args:
        pred   : [B, N]
        target : [B, N]
        margin : minimum required margin
    """
    B, N = pred.shape
    pred_i   = pred.unsqueeze(2).expand(B, N, N)
    pred_j   = pred.unsqueeze(1).expand(B, N, N)
    tgt_i    = target.unsqueeze(2).expand(B, N, N)
    tgt_j    = target.unsqueeze(1).expand(B, N, N)

    order_mask = (tgt_i > tgt_j).float()
    diff       = pred_i - pred_j
    violation  = F.relu(margin - diff)
    loss       = (violation * order_mask).sum() / (order_mask.sum() + 1e-10)
    return loss


def hybrid_loss(rhat, y, ds, A_param, config: Config) -> Tuple[torch.Tensor, Dict]:
    """
    Combined loss for hybrid FX model.

    L = MSE(non-USD) + λ_var * VICReg_var(ds) + λ_dir * BCE(sign) 
       + λ_rank * pairwise_rank + λ_l1 * |A|

    Args:
        rhat      : [B, N] predicted returns
        y         : [B, N] target returns
        ds        : [B, N] latent currency strengths
        A_param   : [N, M] heterogeneous sensitivity matrix
        config    : Config with lambda_* fields
    """
    mask = torch.ones(config.n_ccy, dtype=torch.bool, device=y.device)
    mask[config.usd_idx] = False

    pred   = rhat[:, mask]
    target = y[:, mask]

    # Primary MSE
    mse = ((pred - target) ** 2).mean()

    # VICReg variance constraint (non-USD)
    ds_non_usd = ds[:, mask]
    var_loss   = vicreg_variance_loss(ds_non_usd, gamma=config.vicreg_gamma)

    # Direction BCE (non-USD)
    dir_loss   = direction_bce_loss(pred, target)

    # Pairwise ranking (non-USD)
    rank_loss  = pairwise_ranking_loss(pred, target)

    # A matrix L1 sparsity
    l1_A = A_param.abs().mean() if A_param is not None else torch.tensor(0.0, device=y.device)

    total = (mse
             + config.lambda_var  * var_loss
             + config.lambda_dir  * dir_loss
             + config.lambda_rank * rank_loss
             + config.lambda_a_l1 * l1_A)

    return total, {
        "mse":      mse.item(),
        "var_loss": var_loss.item(),
        "dir_loss": dir_loss.item(),
        "rank_loss":rank_loss.item(),
        "l1_A":     l1_A.item(),
    }


# Legacy simple loss (used by FXStrengthGNN etc.)
def loss_fn(rhat, y, ds, A_param, config: Config) -> torch.Tensor:
    return hybrid_loss(rhat, y, ds, A_param, config)[0]


# ============================================================
# Diagnostic / Interpretability Metrics
# ============================================================

def triangle_error(ds_np: np.ndarray) -> float:
    """Triangle consistency: (s_i - s_j) + (s_j - s_k) + (s_k - s_i) == 0."""
    T, N = ds_np.shape
    err, cnt = 0.0, 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if i != j and j != k and i != k:
                    e = (ds_np[:, i] - ds_np[:, j]) + (ds_np[:, j] - ds_np[:, k]) + (ds_np[:, k] - ds_np[:, i])
                    err += np.abs(e).mean()
                    cnt += 1
    return err / max(cnt, 1)


def macro_usage_ratio(z_ccy: torch.Tensor, m_msg: torch.Tensor, eps: float = 1e-12) -> float:
    """MUR = ||m_msg|| / (||m_msg|| + ||z_ccy||)."""
    num = torch.norm(m_msg, dim=(1, 2)).mean()
    den = (torch.norm(m_msg, dim=(1, 2)) + torch.norm(z_ccy, dim=(1, 2)) + eps).mean()
    return (num / den).item()


def heterogeneity_score(A: torch.Tensor) -> Tuple[float, np.ndarray]:
    """Variance of currency sensitivities per macro factor."""
    var_f = A.var(dim=0, unbiased=False)
    return var_f.mean().item(), var_f.detach().cpu().numpy()


# ============================================================
# Economic / Financial Metrics
# ============================================================

def information_coefficient(pred: np.ndarray, target: np.ndarray,
                            mask: Optional[np.ndarray] = None) -> float:
    """
    Information Coefficient (IC) — Spearman rank correlation
    averaged over time steps.

    Args:
        pred   : [T, N]
        target : [T, N]
        mask   : [N] boolean — exclude USD column

    Returns:
        mean IC over all time steps
    """
    if mask is not None:
        pred   = pred[:, mask]
        target = target[:, mask]

    ics = []
    for t in range(len(pred)):
        if np.std(pred[t]) < 1e-10 or np.std(target[t]) < 1e-10:
            continue
        rho, _ = scipy_stats.spearmanr(pred[t], target[t])
        ics.append(rho)
    return float(np.mean(ics)) if ics else 0.0


def long_short_sharpe(pred: np.ndarray, returns: np.ndarray,
                      k: int = 3, mask: Optional[np.ndarray] = None,
                      annualize: bool = True) -> Dict:
    """
    Long-short portfolio: long top-k, short bottom-k currencies.

    Args:
        pred    : [T, N] predicted scores
        returns : [T, N] actual returns (raw, not normalized)
        k       : number of long / short positions
        mask    : [N] boolean — subset of currencies to consider (excl USD)

    Returns:
        dict with cumulative_return, sharpe, annualized_sharpe
    """
    if mask is not None:
        pred    = pred[:, mask]
        returns = returns[:, mask]

    T, N = pred.shape
    port_rets = []

    for t in range(T):
        ranks = np.argsort(pred[t])   # ascending
        longs  = ranks[-k:]           # top-k predicted
        shorts = ranks[:k]            # bottom-k predicted

        long_ret  = returns[t, longs].mean()
        short_ret = returns[t, shorts].mean()
        port_ret  = long_ret - short_ret
        port_rets.append(port_ret)

    port_rets = np.array(port_rets)
    cum_ret   = np.prod(1 + port_rets) - 1

    mu  = port_rets.mean()
    sig = port_rets.std(ddof=1) + 1e-12
    sr  = mu / sig
    if annualize:
        sr = sr * np.sqrt(252)

    return {
        "cumulative_return": float(cum_ret),
        "sharpe":            float(sr),
        "mean_daily_ret":    float(mu),
        "std_daily_ret":     float(sig - 1e-12),
    }


def hit_rates(pred: np.ndarray, target: np.ndarray, n_ccy: int, usd_idx: int) -> Tuple[float, float]:
    """
    Hit(ccy) : directional accuracy per currency (exclude USD)
    Hit(55)  : directional accuracy for all cross-currency pairs
    """
    from itertools import combinations

    mask = np.ones(n_ccy, dtype=bool)
    mask[usd_idx] = False

    hit_ccy = float(np.mean(np.sign(pred[:, mask]) == np.sign(target[:, mask])))

    edges        = list(combinations(range(n_ccy), 2))
    pred_pairs   = np.stack([pred[:, i] - pred[:, j]   for i, j in edges], axis=1)
    target_pairs = np.stack([target[:, i] - target[:, j] for i, j in edges], axis=1)
    hit_55 = float(np.mean(np.sign(pred_pairs) == np.sign(target_pairs)))

    return hit_ccy, hit_55


# ============================================================
# Trainer
# ============================================================

class Trainer:
    """
    Trainer for FX Strength GNN variants.

    Supports:
    - Train / Val / Test splits
    - Early stopping on validation loss
    - Best-model checkpointing
    """

    def __init__(self, model: nn.Module, config: Config, device: str):
        self.model     = model.to(device)
        self.config    = config
        self.device    = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5, verbose=False
        )
        self._best_val_loss = float("inf")
        self._best_state    = None
        self._no_improve    = 0

    # ----------------------------------------------------------

    def _get_A(self) -> Optional[torch.Tensor]:
        """Safely retrieve model.A if it exists."""
        return getattr(self.model, "A", None)

    def _forward(self, xl, xm, edge_index):
        """Unified forward — returns (rhat, ds, z_ccy, m_msg)."""
        out = self.model(xl, xm, edge_index)
        # Models return 4 or 5 values; always take first 4
        return out[0], out[1], out[2], out[3]

    # ----------------------------------------------------------

    def train_epoch(self, train_loader, edge_index) -> float:
        """One training epoch. Returns mean loss."""
        self.model.train()
        losses = []

        for xl, xm, yb in train_loader:
            xl, xm, yb = xl.to(self.device), xm.to(self.device), yb.to(self.device)
            rhat, ds, z_ccy, m_msg = self._forward(xl, xm, edge_index)

            A_param = self._get_A()
            loss, _ = hybrid_loss(rhat, yb, ds, A_param, self.config)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            losses.append(loss.item())

        return float(np.mean(losses))

    @torch.no_grad()
    def eval_epoch(self, loader, edge_index) -> float:
        """Evaluate and return mean loss (used for early stopping)."""
        self.model.eval()
        losses = []
        for xl, xm, yb in loader:
            xl, xm, yb = xl.to(self.device), xm.to(self.device), yb.to(self.device)
            rhat, ds, _, _ = self._forward(xl, xm, edge_index)
            A_param = self._get_A()
            loss, _ = hybrid_loss(rhat, yb, ds, A_param, self.config)
            losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("inf")

    # ----------------------------------------------------------

    def train(self, train_loader, val_loader, test_loader, edge_index, label: str = ""):
        """
        Full training loop with early stopping and best-model selection.

        Returns final evaluation metrics dict.
        """
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader, edge_index)
            val_loss   = self.eval_epoch(val_loader,   edge_index)
            self.scheduler.step(val_loss)

            log_flag = (epoch == 1 or epoch == self.config.epochs or epoch % 10 == 0)
            if log_flag:
                logger.info(f"[{label}] epoch {epoch:03d}/{self.config.epochs}"
                            f" | train={train_loss:.4f} | val={val_loss:.4f}")

            # Early stopping
            if val_loss < self._best_val_loss - 1e-6:
                self._best_val_loss = val_loss
                self._best_state    = copy.deepcopy(self.model.state_dict())
                self._no_improve    = 0
            else:
                self._no_improve += 1

            if self._no_improve >= self.config.early_stopping_patience:
                logger.info(f"[{label}] Early stopping at epoch {epoch} "
                            f"(best val={self._best_val_loss:.4f})")
                break

        # Restore best checkpoint
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            logger.info(f"[{label}] Restored best model (val={self._best_val_loss:.4f})")

        metrics = self.evaluate(test_loader, edge_index)
        self._print_metrics(label, metrics)
        return metrics

    # ----------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader, edge_index) -> Dict:
        """Full evaluation on a split. Returns comprehensive metrics."""
        self.model.eval()

        rhat_all, y_all, ds_all, z_all, m_all = [], [], [], [], []

        for xl, xm, yb in loader:
            xl, xm, yb = xl.to(self.device), xm.to(self.device), yb.to(self.device)
            rhat, ds, z_ccy, m_msg = self._forward(xl, xm, edge_index)

            rhat_all.append(rhat.cpu())
            y_all.append(yb.cpu())
            ds_all.append(ds.cpu())
            z_all.append(z_ccy.cpu())
            m_all.append(m_msg.cpu())

        rhat  = torch.cat(rhat_all, dim=0)
        y     = torch.cat(y_all,    dim=0)
        ds    = torch.cat(ds_all,   dim=0)
        z_ccy = torch.cat(z_all,    dim=0)
        m_msg = torch.cat(m_all,    dim=0)

        rhat_np = rhat.numpy()
        y_np    = y.numpy()
        ds_np   = ds.numpy()

        mask = np.ones(self.config.n_ccy, dtype=bool)
        mask[self.config.usd_idx] = False

        # Standard regression metrics
        rmse = float(np.sqrt(((rhat_np[:, mask] - y_np[:, mask]) ** 2).mean()))
        mae  = float(np.abs(rhat_np[:, mask] - y_np[:, mask]).mean())

        # Directional metrics
        hit_ccy, hit_55 = hit_rates(rhat_np, y_np, self.config.n_ccy, self.config.usd_idx)

        # IC
        ic = information_coefficient(rhat_np, y_np, mask=mask)

        # Long-short Sharpe
        ls = long_short_sharpe(rhat_np, y_np, k=self.config.top_k_portfolio, mask=mask)

        # Graph / latent diagnostics
        strength_norm = float(torch.norm(ds, dim=1).mean().item())
        tri_err       = triangle_error(ds_np)
        mur           = macro_usage_ratio(z_ccy, m_msg)

        A_param = self._get_A()
        if A_param is not None:
            hs_mean, hs_vec = heterogeneity_score(A_param.detach().cpu())
        else:
            hs_mean, hs_vec = 0.0, np.zeros(self.config.macro_dim)

        return {
            "rmse":             rmse,
            "mae":              mae,
            "hit_ccy":          hit_ccy,
            "hit_55":           hit_55,
            "ic":               ic,
            "sharpe":           ls["sharpe"],
            "cum_return":       ls["cumulative_return"],
            "strength_norm":    strength_norm,
            "tri_err":          tri_err,
            "mur":              mur,
            "hs_mean":          hs_mean,
            "hs_vec":           hs_vec,
        }

    # ----------------------------------------------------------

    def _print_metrics(self, label: str, m: Dict):
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
        print(f"  RMSE           : {m['rmse']:.6f}")
        print(f"  MAE            : {m['mae']:.6f}")
        print(f"  Hit(ccy)       : {m['hit_ccy']:.4f}")
        print(f"  Hit(55)        : {m['hit_55']:.4f}")
        print(f"  IC (Spearman)  : {m['ic']:.4f}")
        print(f"  Sharpe (ann.)  : {m['sharpe']:.4f}")
        print(f"  Cum. Return    : {m['cum_return']:.4f}")
        print(f"  ||Strength||   : {m['strength_norm']:.4f}")
        print(f"  Triangle Err   : {m['tri_err']:.2e}")
        print(f"  MUR            : {m['mur']:.4f}")
        print(f"  Hetero Score   : {m['hs_mean']:.6f}")
        print(f"{'=' * 60}\n")
