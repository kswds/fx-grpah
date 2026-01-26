"""
Training utilities for FX Strength GNN
Matches original fx_train.py loss and metrics exactly
"""
import numpy as np
import torch
import torch.nn as nn
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
