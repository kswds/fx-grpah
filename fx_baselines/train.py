import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import FXDataset
from models import (
    FXBaselineMLP,
    StaticFCGNNMacro,
    GrangerShockPropagationGNN,
)
from models import fully_connected_row_stochastic

def strength_loss(rhat, y, ds, A_param, config: Config) -> torch.Tensor:
    """
    Strength-based loss used across MLP/graph models for fair comparison.
    """
    mask = torch.ones(config.n_ccy, dtype=torch.bool, device=y.device)
    mask[config.usd_idx] = False

    mse = ((rhat[:, mask] - y[:, mask]) ** 2).mean()
    var_term = -ds.var(dim=1).mean()
    l1_A = A_param.abs().mean()
    return mse + config.lambda_var * var_term + config.lambda_a_l1 * l1_A

def weighted_hit(rhat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Weight sign accuracy by |y| to emphasize larger moves.
    """
    y_m = y[:, mask]
    r_m = rhat[:, mask]
    w = torch.abs(y_m)
    correct = (torch.sign(r_m) == torch.sign(y_m)).float()
    return float((w * correct).sum() / (w.sum() + 1e-12))

def train_and_evaluate(
    config: Config,
    X_local_raw: np.ndarray,
    X_macro_raw: np.ndarray,
    Y_raw: np.ndarray,
    model_name: str,
    macro_mode: str,
    label: str,
    W_graph: torch.Tensor | None = None,
):
    """
    model_name: "mlp" | "static_fc" | "granger_shockprop"
    macro_mode: "real" | "zero"
    """
    total_len = len(X_local_raw)
    valid_len = total_len - config.lookback
    train_split_idx = int(valid_len * 0.8) + config.lookback

    # ---- Train-only scaling for inputs ----
    train_local = X_local_raw[:train_split_idx]  # [T_train,N,3]
    train_macro = X_macro_raw[:train_split_idx]  # [T_train,M]

    local_mean = train_local.mean(axis=(0, 1), keepdims=True)
    local_std  = train_local.std(axis=(0, 1), keepdims=True) + 1e-6
    macro_mean = train_macro.mean(axis=0, keepdims=True)
    macro_std  = train_macro.std(axis=0, keepdims=True) + 1e-6

    X_local_scaled = (X_local_raw - local_mean) / local_std
    X_macro_scaled = (X_macro_raw - macro_mean) / macro_std

    dataset = FXDataset(X_local_scaled, X_macro_scaled, Y_raw, config, macro_mode=macro_mode)

    n = len(dataset)
    split = int(n * 0.8)
    train_ds = torch.utils.data.Subset(dataset, list(range(0, split)))
    test_ds  = torch.utils.data.Subset(dataset, list(range(split, n)))

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    if model_name == "mlp":
        model = FXBaselineMLP(config).to(config.device)
    elif model_name == "static_fc":
        model = StaticFCGNNMacro(config).to(config.device)
        assert W_graph is not None
    elif model_name == "granger_shockprop":
        model = GrangerShockPropagationGNN(config).to(config.device)
        assert W_graph is not None
    else:
        raise ValueError(model_name)

    optim = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # ---- Train ----
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for xl, xm, yb, t_idx in train_loader:
            xl = xl.to(config.device)
            xm = xm.to(config.device)
            yb = yb.to(config.device)

            if model_name == "mlp":
                rhat, ds, _, _ = model(xl, xm)
            else:
                rhat, ds, _, _ = model(xl, xm, W_graph)

            loss = strength_loss(rhat, yb, ds, model.A, config)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(float(loss.item()))

        if epoch in (1, 5, 10, 15, 20, 25, config.epochs):
            print(f"[{label}] epoch {epoch:02d}/{config.epochs} | mean loss = {np.mean(losses):.6f}")

    # ---- Eval ----
    model.eval()
    rhat_all, y_all, t_all = [], [], []
    with torch.no_grad():
        for xl, xm, yb, t_idx in test_loader:
            xl = xl.to(config.device)
            xm = xm.to(config.device)
            yb = yb.to(config.device)

            if model_name == "mlp":
                rhat, ds, _, _ = model(xl, xm)
            else:
                rhat, ds, _, _ = model(xl, xm, W_graph)

            rhat_all.append(rhat.detach().cpu())
            y_all.append(yb.detach().cpu())
            t_all.append(t_idx.detach().cpu())

    rhat_test = torch.cat(rhat_all, dim=0)
    y_test    = torch.cat(y_all, dim=0)
    t_test    = torch.cat(t_all, dim=0)

    mask = torch.ones(config.n_ccy, dtype=torch.bool)
    mask[config.usd_idx] = False

    rmse = float(torch.sqrt(((rhat_test[:, mask] - y_test[:, mask]) ** 2).mean()).item())
    mae  = float(torch.abs(rhat_test[:, mask] - y_test[:, mask]).mean().item())
    hit  = float((torch.sign(rhat_test[:, mask]) == torch.sign(y_test[:, mask])).float().mean().item())
    w_hit = weighted_hit(rhat_test, y_test, mask)

    pos_rate = float((y_test[:, mask] > 0).float().mean().item())
    naive = max(pos_rate, 1 - pos_rate)

    # Extreme threshold from TRAIN distribution of RAW |y|
    y_raw_all = torch.tensor(Y_raw, dtype=torch.float32)
    train_y_raw = y_raw_all[config.lookback:train_split_idx, :][:, mask]
    thr = float(torch.quantile(torch.abs(train_y_raw), config.extreme_percentile).item())

    is_ext = torch.abs(y_test[:, mask]) >= thr
    is_norm = ~is_ext

    def safe_mean(x: torch.Tensor) -> float:
        return float(x.mean().item()) if x.numel() > 0 else float("nan")

    hit_norm = safe_mean((torch.sign(rhat_test[:, mask][is_norm]) == torch.sign(y_test[:, mask][is_norm])).float())
    hit_ext  = safe_mean((torch.sign(rhat_test[:, mask][is_ext])  == torch.sign(y_test[:, mask][is_ext])).float())

    rmse_norm = safe_mean(torch.sqrt(((rhat_test[:, mask][is_norm] - y_test[:, mask][is_norm]) ** 2)).view(-1)) \
        if is_norm.sum() > 0 else float("nan")
    rmse_ext = safe_mean(torch.sqrt(((rhat_test[:, mask][is_ext] - y_test[:, mask][is_ext]) ** 2)).view(-1)) \
        if is_ext.sum() > 0 else float("nan")

    print(f"===== {label} =====")
    print(f"Target space: RAW FXRet (log return)   [NO Y normalization]")
    print(f"RMSE: {rmse:.6f} | MAE: {mae:.6f} | Hit: {hit:.4f} | W-Hit: {w_hit:.4f}")
    print(f"Pos-rate(test): {pos_rate:.3f} | Naive(best-const-sign): {naive:.3f}")
    print(f"Extreme thr(|y|)@q={config.extreme_percentile:.2f}: {thr:.6f}")
    print(f"Normal : n={int(is_norm.sum())} | Hit={hit_norm:.4f} | RMSE={rmse_norm:.6f}")
    print(f"Extreme: n={int(is_ext.sum())} | Hit={hit_ext:.4f} | RMSE={rmse_ext:.6f}\n")

    return {
        "rmse": rmse,
        "mae": mae,
        "hit": hit,
        "weighted_hit": w_hit,
        "pos_rate": pos_rate,
        "naive": naive,
        "thr": thr,
        "hit_norm": hit_norm,
        "hit_ext": hit_ext,
        "rmse_norm": rmse_norm,
        "rmse_ext": rmse_ext,
    }
