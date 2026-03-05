"""
Training utilities for FX Strength GNN
Matches original fx_train.py loss and metrics exactly
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from config import Config


def loss_fn(rhat, y, ds, A_param, config: Config):
    """
    Loss function matching original implementation

    L = MSE(pred, target) + λ_var * (-Var(ds)) + λ_l1 * |A|

    Args:
        rhat: Predicted returns [B, N]
        y: Target returns [B, N]
        ds: Currency strengths [B, N]
        A_param: Heterogeneous transmission matrix [N, M]
        config: Configuration
    """
    mask = torch.ones(config.n_ccy, dtype=torch.bool, device=y.device)
    mask[config.usd_idx] = False

    # MSE loss (exclude USD)
    mse = ((rhat[:, mask] - y[:, mask]) ** 2).mean()

    # Variance term (prevent latent collapse)
    var_term = -ds.var(dim=1).mean()

    # L1 sparsity on A
    l1_A = A_param.abs().mean()

    return mse + config.lambda_var * var_term + config.lambda_a_l1 * l1_A


def confidence_loss_fn(rhat, y, ds, A_param, confidence, direction, magnitude, config: Config,
                        lambda_dir: float = 0.5, lambda_mag: float = 0.3,
                        lambda_conf: float = 0.1, lambda_var: float = 0.005):
    """
    Loss function for Confidence-Weighted Sparse Graph model

    L = L_total + λ_dir * L_direction + λ_mag * L_magnitude + λ_conf * L_confidence

    Where:
    - L_total: MSE on final prediction (for overall fit)
    - L_direction: Sign accuracy loss (soft cross-entropy on direction)
    - L_magnitude: Magnitude matching loss (L1 on |prediction| vs |target|)
    - L_confidence: Confidence calibration (high conf should be more accurate)

    Args:
        rhat: Predicted returns [B, N]
        y: Target returns [B, N]
        ds: Currency strengths [B, N]
        A_param: Heterogeneous transmission matrix [N, M]
        confidence: Per-currency confidence [B, N]
        direction: Direction prediction [-1, 1] [B, N]
        magnitude: Magnitude prediction [B, N]
        config: Configuration
        lambda_dir: Weight for direction loss
        lambda_mag: Weight for magnitude loss
        lambda_conf: Weight for confidence calibration
        lambda_var: Weight for variance regularization
    """
    mask = torch.ones(config.n_ccy, dtype=torch.bool, device=y.device)
    mask[config.usd_idx] = False

    pred = rhat[:, mask]
    target = y[:, mask]
    conf = confidence[:, mask]
    dir_pred = direction[:, mask]
    mag_pred = magnitude[:, mask]

    # 1. Total MSE loss
    mse = ((pred - target) ** 2).mean()

    # 2. Direction loss: Soft sign matching
    # target_sign in {-1, +1}, dir_pred in [-1, 1]
    target_sign = torch.sign(target)
    # Use smooth L1 on sign matching: (dir_pred - target_sign)^2 when close, L1 when far
    dir_loss = F.smooth_l1_loss(dir_pred, target_sign)

    # 3. Magnitude loss: Match |prediction| to |target|
    target_mag = torch.abs(target)
    mag_loss = F.l1_loss(mag_pred, target_mag)

    # 4. Confidence calibration: High confidence should mean correct direction
    # is_correct: 1 if sign matches, 0 otherwise
    is_correct = (torch.sign(pred) == target_sign).float()
    # Confidence should predict accuracy: BCE(conf, is_correct)
    conf_loss = F.binary_cross_entropy(conf, is_correct)

    # 5. Variance term (prevent latent collapse)
    var_term = -ds.var(dim=1).mean()

    # 6. L1 sparsity on A
    l1_A = A_param.abs().mean()

    total_loss = (mse +
                  lambda_dir * dir_loss +
                  lambda_mag * mag_loss +
                  lambda_conf * conf_loss +
                  lambda_var * var_term +
                  config.lambda_a_l1 * l1_A)

    return total_loss, {
        'mse': mse.item(),
        'dir_loss': dir_loss.item(),
        'mag_loss': mag_loss.item(),
        'conf_loss': conf_loss.item()
    }


# ============================================================
# Rank-based Loss Functions
# ============================================================

def listMLE_loss(pred, target, eps=1e-10):
    """
    ListMLE: Listwise Learning to Rank Loss

    Maximizes likelihood of observing the correct ranking.
    P(ranking) = prod_i [ exp(y_i) / sum_{j>=i} exp(y_j) ]

    Args:
        pred: Predicted scores [B, N]
        target: Target values [B, N] (used to determine ground truth ranking)

    Returns:
        loss: Scalar loss value
    """
    B, N = pred.shape

    # Get ground truth ranking (descending order by target value)
    # sorted_idx[b, i] = index of i-th largest target in batch b
    sorted_idx = torch.argsort(target, dim=1, descending=True)

    # Reorder predictions by ground truth ranking
    pred_sorted = torch.gather(pred, 1, sorted_idx)  # [B, N]

    # Compute ListMLE loss
    # For each position i, compute log(softmax) of position i given positions i:N
    loss = 0.0
    for i in range(N - 1):
        # log(exp(pred[i]) / sum_{j>=i} exp(pred[j]))
        log_softmax = pred_sorted[:, i] - torch.logsumexp(pred_sorted[:, i:], dim=1)
        loss = loss - log_softmax.mean()

    return loss / (N - 1)


def pairwise_margin_loss(pred, target, margin=0.1):
    """
    Pairwise Margin Loss

    For each pair (i, j) where target[i] > target[j],
    enforce pred[i] > pred[j] + margin

    Args:
        pred: Predicted scores [B, N]
        target: Target values [B, N]
        margin: Minimum margin between pairs

    Returns:
        loss: Scalar loss value
    """
    B, N = pred.shape

    # Create all pairs
    # pred_i[b, i, j] = pred[b, i], pred_j[b, i, j] = pred[b, j]
    pred_i = pred.unsqueeze(2).expand(B, N, N)  # [B, N, N]
    pred_j = pred.unsqueeze(1).expand(B, N, N)  # [B, N, N]

    target_i = target.unsqueeze(2).expand(B, N, N)
    target_j = target.unsqueeze(1).expand(B, N, N)

    # Mask: 1 if target_i > target_j (i should rank higher than j)
    mask = (target_i > target_j).float()

    # Hinge loss: max(0, margin - (pred_i - pred_j)) when target_i > target_j
    diff = pred_i - pred_j  # [B, N, N]
    hinge = F.relu(margin - diff)  # [B, N, N]

    # Apply mask and average
    loss = (hinge * mask).sum() / (mask.sum() + 1e-10)

    return loss


def topk_focused_loss(pred, target, k=3, alpha=2.0):
    """
    Top-K Focused Loss

    Extra weight on getting the top-K and bottom-K correct.

    Args:
        pred: Predicted scores [B, N]
        target: Target values [B, N]
        k: Number of top/bottom positions to focus on
        alpha: Weight multiplier for extreme positions

    Returns:
        loss: Scalar loss value
    """
    B, N = pred.shape

    # Get ranks (0 = largest, N-1 = smallest)
    target_ranks = torch.argsort(torch.argsort(target, dim=1, descending=True), dim=1)

    # Weight: alpha for top-k and bottom-k, 1 for middle
    weights = torch.ones_like(target, dtype=torch.float)
    weights[target_ranks < k] = alpha  # Top-k
    weights[target_ranks >= N - k] = alpha  # Bottom-k

    # Weighted MSE on the difference
    mse = ((pred - target) ** 2) * weights

    return mse.mean()


def rank_loss_fn(rhat, y, ds, A_param, config,
                 lambda_mse=1.0, lambda_list=0.5, lambda_pair=0.3, lambda_topk=0.2):
    """
    Combined Rank-aware Loss Function

    L = MSE + λ_list * ListMLE + λ_pair * Pairwise + λ_topk * TopK

    Args:
        rhat: Predicted returns [B, N]
        y: Target returns [B, N]
        ds: Currency strengths [B, N]
        A_param: A matrix
        config: Config
        lambda_*: Loss weights
    """
    mask = torch.ones(config.n_ccy, dtype=torch.bool, device=y.device)
    mask[config.usd_idx] = False

    pred = rhat[:, mask]
    target = y[:, mask]

    # MSE loss
    mse = ((pred - target) ** 2).mean()

    # Rank losses
    list_loss = listMLE_loss(pred, target)
    pair_loss = pairwise_margin_loss(pred, target)
    topk_loss = topk_focused_loss(pred, target, k=2)

    # Variance term
    var_term = -ds.var(dim=1).mean()

    # L1 on A
    l1_A = A_param.abs().mean()

    total = (lambda_mse * mse +
             lambda_list * list_loss +
             lambda_pair * pair_loss +
             lambda_topk * topk_loss +
             config.lambda_var * var_term +
             config.lambda_a_l1 * l1_A)

    return total, {
        'mse': mse.item(),
        'list': list_loss.item(),
        'pair': pair_loss.item(),
        'topk': topk_loss.item()
    }


def triangle_error(ds_np: np.ndarray) -> float:
    """
    Compute triangle consistency error
    For all i,j,k: (s_i - s_j) + (s_j - s_k) + (s_k - s_i) should be 0
    """
    T, N = ds_np.shape
    err = 0.0
    cnt = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if i != j and j != k and i != k:
                    e = (ds_np[:, i] - ds_np[:, j]) + (ds_np[:, j] - ds_np[:, k]) + (ds_np[:, k] - ds_np[:, i])
                    err += np.abs(e).mean()
                    cnt += 1
    return err / max(cnt, 1)


def macro_usage_ratio(z_ccy: torch.Tensor, m_msg: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Compute ratio of macro message contribution

    MUR = ||m_msg|| / (||m_msg|| + ||z_ccy||)
    """
    num = torch.norm(m_msg, dim=(1, 2)).mean()
    den = (torch.norm(m_msg, dim=(1, 2)) + torch.norm(z_ccy, dim=(1, 2)) + eps).mean()
    return (num / den).item()


def heterogeneity_score(A: torch.Tensor) -> Tuple[float, np.ndarray]:
    """
    Compute heterogeneity score of A matrix

    Measures variance of currency sensitivities per macro factor
    """
    var_f = A.var(dim=0, unbiased=False)
    return var_f.mean().item(), var_f.detach().cpu().numpy()


class Trainer:
    """Trainer class for FX Strength GNN"""

    def __init__(self, model: nn.Module, config: Config, device: str):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    def train_epoch(self, train_loader, edge_index) -> float:
        """Train for one epoch"""
        self.model.train()
        losses = []

        for xl, xm, yb in train_loader:
            xl, xm, yb = xl.to(self.device), xm.to(self.device), yb.to(self.device)

            if self.config.use_magnitude_head:
                rhat, ds, z_ccy, m_msg, magnitude = self.model(xl, xm, edge_index)
                # Scale prediction by magnitude
                rhat = rhat * magnitude.unsqueeze(1)
            else:
                rhat, ds, z_ccy, m_msg = self.model(xl, xm, edge_index)
            loss = loss_fn(rhat, yb, ds, self.model.A, self.config)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            losses.append(loss.item())

        return np.mean(losses)

    @torch.no_grad()
    def evaluate(self, test_loader, edge_index) -> Dict:
        """Evaluate model on test set"""
        self.model.eval()

        rhat_all, y_all, ds_all = [], [], []
        z_all, m_all = [], []

        for xl, xm, yb in test_loader:
            xl, xm, yb = xl.to(self.device), xm.to(self.device), yb.to(self.device)

            if self.config.use_magnitude_head:
                rhat, ds, z_ccy, m_msg, magnitude = self.model(xl, xm, edge_index)
                rhat = rhat * magnitude.unsqueeze(1)
            else:
                rhat, ds, z_ccy, m_msg = self.model(xl, xm, edge_index)

            rhat_all.append(rhat.cpu())
            y_all.append(yb.cpu())
            ds_all.append(ds.cpu())
            z_all.append(z_ccy.cpu())
            m_all.append(m_msg.cpu())

        rhat = torch.cat(rhat_all, dim=0)
        y = torch.cat(y_all, dim=0)
        ds = torch.cat(ds_all, dim=0)
        z_ccy = torch.cat(z_all, dim=0)
        m_msg = torch.cat(m_all, dim=0)

        # Metrics (exclude USD)
        mask = torch.ones(self.config.n_ccy, dtype=torch.bool)
        mask[self.config.usd_idx] = False

        rmse = torch.sqrt(((rhat[:, mask] - y[:, mask]) ** 2).mean()).item()
        mae = torch.abs(rhat[:, mask] - y[:, mask]).mean().item()
        hit = ((torch.sign(rhat[:, mask]) == torch.sign(y[:, mask])).float()).mean().item()

        strength_norm = torch.norm(ds, dim=1).mean().item()
        tri_err = triangle_error(ds.numpy())

        mur = macro_usage_ratio(z_ccy, m_msg)
        hs_mean, hs_vec = heterogeneity_score(self.model.A.detach().cpu())

        return {
            "rmse": rmse,
            "mae": mae,
            "hit": hit,
            "strength_norm": strength_norm,
            "tri_err": tri_err,
            "mur": mur,
            "hs_mean": hs_mean,
            "hs_vec": hs_vec,
        }

    def train(self, train_loader, test_loader, edge_index, label: str = ""):
        """Full training loop"""
        for epoch in range(1, self.config.epochs + 1):
            loss = self.train_epoch(train_loader, edge_index)

            if epoch in (1, self.config.epochs) or epoch % 5 == 0:
                print(f"[{label}] epoch {epoch:02d}/{self.config.epochs} | mean loss = {loss:.4f}")

        # Final evaluation
        metrics = self.evaluate(test_loader, edge_index)

        print(f"\n===== {label} =====")
        print(f"RMSE               : {metrics['rmse']:.4f}")
        print(f"MAE                : {metrics['mae']:.4f}")
        print(f"Directional Acc.   : {metrics['hit']:.4f}")
        print(f"Mean ||Δs||        : {metrics['strength_norm']:.4f}")
        print(f"Triangle Error     : {metrics['tri_err']:.6e}")
        print(f"Macro Usage Ratio  : {metrics['mur']:.4f}   (macro message share)")
        print(f"Heterogeneity Score: {metrics['hs_mean']:.6f} (mean Var_i(a_i,f))")
        print(f" per-factor Var    : {np.array2string(metrics['hs_vec'], precision=6)}")

        return metrics
