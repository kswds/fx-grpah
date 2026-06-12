import copy
import json
import os
import random
import sys
from dataclasses import asdict
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats as scipy_stats
from sklearn.linear_model import Ridge as SklearnRidge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from config import Config
from dataset import build_features, load_data, normalize_data
from models import create_model
from train import information_coefficient, long_short_sharpe

ALL_MACRO_FEATURES = [
    "Global_VIX",
    "Global_Gold",
    "Global_Oil",
    "Global_Copper",
    "Global_US2Y",
    "Global_IronOre",
    "Global_US10Y",
    "Global_Shanghai",
    "Global_SP500",
]

DEFAULT_DISPLAY_NAMES = {
    "MLP": "MLP",
    "LSTM": "LSTM",
    "GRU": "GRU",
    "Transformer": "Transformer",
    "GAT": "GAT-StrengthGNN",
    "NoGraph": "NoGraphFX",
    "NoMacro": "NoMacroFX",
    "StaticGraph": "StaticGraphFX",
    "PureGraphFX": "PureGraphFX",
    "FiLMHyGraph": "FiLMHyGraph",
    "Ours": "Ours",
}

MODEL_ALIASES = {
    "MLP": "mlp",
    "LSTM": "lstm",
    "GRU": "GRU",
    "Transformer": "Transformer",
    "GAT": "GNN",
    "NoGraph": "NoGraph",
    "NoMacro": "NoMacro",
    "StaticGraph": "StaticGraph",
    "PureGraphFX": "PureGraphFX",
    "FiLMHyGraph": "FiLMHyGraph",
    "Ours": "Ours",
}


class FlatMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class GRUBaseline(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_ccy: int, usd_idx: int):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)
        self.n_ccy = n_ccy
        self.usd_idx = usd_idx

    def forward(self, xl, xm):
        B, L, N, _ = xl.shape
        xm_exp = xm.unsqueeze(2).expand(-1, -1, N, -1)
        x = torch.cat([xl, xm_exp], dim=-1).permute(0, 2, 1, 3).reshape(B * N, L, -1)
        _, h = self.gru(x)
        ds = self.head(h.squeeze(0)).squeeze(-1).view(B, N)
        return ds - ds[:, self.usd_idx : self.usd_idx + 1]


class TransformerBaseline(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_ccy: int, usd_idx: int, n_heads: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, min(n_heads, hidden_dim // 8 if hidden_dim >= 8 else 1)),
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(hidden_dim, 1)
        self.n_ccy = n_ccy
        self.usd_idx = usd_idx

    def forward(self, xl, xm):
        B, L, N, _ = xl.shape
        xm_exp = xm.unsqueeze(2).expand(-1, -1, N, -1)
        x = torch.cat([xl, xm_exp], dim=-1).permute(0, 2, 1, 3).reshape(B * N, L, -1)
        h = self.enc(self.in_proj(x))
        ds = self.head(h[:, -1]).squeeze(-1).view(B, N)
        return ds - ds[:, self.usd_idx : self.usd_idx + 1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lookback_k(mode: Any, L: int) -> int:
    if mode in ("last", 1, "1"):
        return 1
    if mode in ("full", L, str(L)):
        return L
    return int(mode)


def compute_rv_numpy(fxret: np.ndarray, window: int) -> np.ndarray:
    LL, N = fxret.shape
    rv = np.zeros((LL, N), dtype=np.float32)
    for t in range(LL):
        s = max(0, t - window + 1)
        seg = fxret[s : t + 1]
        if len(seg) >= 2:
            rv[t] = seg.std(axis=0)
    return rv


def load_or_create_results_dir(base_dir: str | Path, subdirs: list[str]) -> dict[str, Path]:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    out = {"base": base}
    for s in subdirs:
        p = base / s
        p.mkdir(parents=True, exist_ok=True)
        out[s] = p
    return out


def save_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_convert)


def save_csv(path: str | Path, df: pd.DataFrame) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def save_predictions(path: str | Path, df: pd.DataFrame) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def _json_convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def prepare_data_split(
    config: Config,
    split_mode: str = "602020",
    data_path: str | None = None,
    macro_features: list[str] | None = None,
) -> dict:
    if split_mode != "602020":
        raise ValueError(f"Only split_mode='602020' is supported. got={split_mode}")

    cfg = copy.deepcopy(config)
    if data_path is not None:
        cfg.file_path = data_path

    df = load_data(cfg)
    dates = pd.to_datetime(df["Date"]).copy()
    X_local_base, X_macro_full, Y = build_features(df, cfg)

    macro_features = macro_features or ALL_MACRO_FEATURES
    missing = [f for f in macro_features if f not in cfg.global_features]
    if missing:
        raise ValueError(
            f"Missing macro features in config.global_features: {missing}. "
            f"Available: {cfg.global_features}"
        )
    macro_idx = [cfg.global_features.index(f) for f in macro_features]
    X_macro = X_macro_full[:, macro_idx]

    n_total = len(X_local_base)
    train_raw = int(n_total * cfg.train_ratio)
    X_local_s, X_macro_s, Y_raw, stats = normalize_data(X_local_base, X_macro, Y, train_idx=train_raw)

    L = cfg.lookback
    n = n_total - L
    train_end, val_end = cfg.get_split_indices(n)
    feat_dates = dates.iloc[1:].reset_index(drop=True)
    if len(feat_dates) != n_total:
        feat_dates = feat_dates.iloc[:n_total]

    return {
        "config": cfg,
        "X_local": X_local_s,
        "X_macro": X_macro_s,
        "Y": Y_raw,
        "stats": stats,
        "feature_dates": feat_dates,
        "L": L,
        "n": n,
        "train_end": train_end,
        "val_end": val_end,
        "n_ccy": cfg.n_ccy,
        "usd_idx": cfg.usd_idx,
        "currency_names": list(cfg.ccys),
        "macro_features": list(macro_features),
    }


def build_windows(
    data_bundle: dict,
    lookback_mode: Any,
    split: str,
    add_rv: bool = False,
    return_dates: bool = True,
):
    X_local = data_bundle["X_local"]
    X_macro = data_bundle["X_macro"]
    Y = data_bundle["Y"]
    dates = data_bundle["feature_dates"]
    L = data_bundle["L"]
    n = data_bundle["n"]
    train_end = data_bundle["train_end"]
    val_end = data_bundle["val_end"]
    k = lookback_k(lookback_mode, L)

    ranges = {"train": range(0, train_end), "val": range(train_end, val_end), "test": range(val_end, n)}
    if split not in ranges:
        raise ValueError(f"unknown split={split}")
    indices = ranges[split]

    Xl_list, Xm_list, Y_list, recs = [], [], [], []
    for idx in indices:
        wl = X_local[idx : idx + L][-k:]
        wm = X_macro[idx : idx + L][-k:]
        if add_rv:
            rv = compute_rv_numpy(wl[:, :, 0], window=k)
            wl = np.concatenate([wl, rv[:, :, np.newaxis]], axis=-1)
        Xl_list.append(wl)
        Xm_list.append(wm)
        Y_list.append(Y[idx + L])
        if return_dates:
            recs.append(
                {
                    "sample_idx": idx,
                    "input_end_date": pd.Timestamp(dates.iloc[idx + L - 1]),
                    "target_date": pd.Timestamp(dates.iloc[idx + L]),
                }
            )
    return (
        np.stack(Xl_list).astype(np.float32),
        np.stack(Xm_list).astype(np.float32),
        np.stack(Y_list).astype(np.float32),
        pd.DataFrame(recs) if return_dates else None,
    )


def resolve_model_key(name: str) -> str:
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]
    if name in MODEL_ALIASES.values():
        return name
    raise ValueError(f"Unsupported model '{name}'. Supported={list(MODEL_ALIASES.keys())}")


def _predict_tensor(model, Xl, Xm, model_key: str, device: torch.device):
    model.eval()
    with torch.no_grad():
        if model_key == "Ridge":
            raise RuntimeError("Ridge path should not call _predict_tensor")
        if model_key == "MLP":
            inp = np.concatenate([Xl.reshape(len(Xl), -1), Xm.reshape(len(Xm), -1)], axis=1)
            xt = torch.tensor(inp, dtype=torch.float32, device=device)
            return model(xt).cpu().numpy()
        out = model(
            torch.tensor(Xl, dtype=torch.float32, device=device),
            torch.tensor(Xm, dtype=torch.float32, device=device),
            None,
        )
        return out[0].detach().cpu().numpy()


def train_model(
    model_name: str,
    data_bundle: dict,
    lookback_mode: Any,
    seed: int,
    train_params: dict,
    device: torch.device | None = None,
):
    set_seed(seed)
    device = device or get_device()

    n_ccy = data_bundle["n_ccy"]
    usd_idx = data_bundle["usd_idx"]
    model_key = resolve_model_key(model_name)
    add_rv = model_name in {"NoGraph", "NoMacro", "StaticGraph", "PureGraphFX", "FiLMHyGraph", "Ours", "LSTM", "GAT", "GRU", "Transformer"}
    Xl_tr, Xm_tr, Y_tr, _ = build_windows(data_bundle, lookback_mode, "train", add_rv=add_rv)
    Xl_vl, Xm_vl, Y_vl, _ = build_windows(data_bundle, lookback_mode, "val", add_rv=add_rv)

    y_mean = Y_tr.mean(axis=0, keepdims=True)
    y_std = Y_tr.std(axis=0, keepdims=True) + 1e-8
    ytr_n = (Y_tr - y_mean) / y_std
    yvl_n = (Y_vl - y_mean) / y_std

    if model_name == "MLP":
        xtr = np.concatenate([Xl_tr.reshape(len(Xl_tr), -1), Xm_tr.reshape(len(Xm_tr), -1)], axis=1)
        xvl = np.concatenate([Xl_vl.reshape(len(Xl_vl), -1), Xm_vl.reshape(len(Xm_vl), -1)], axis=1)
        model = FlatMLP(xtr.shape[1], int(train_params["hidden"]), n_ccy).to(device)
        opt = optim.Adam(model.parameters(), lr=float(train_params["lr"]), weight_decay=float(train_params["weight_decay"]))
        xt = torch.tensor(xtr, dtype=torch.float32, device=device)
        yt = torch.tensor(ytr_n, dtype=torch.float32, device=device)
        xv = torch.tensor(xvl, dtype=torch.float32, device=device)
        yv = torch.tensor(yvl_n, dtype=torch.float32, device=device)
        best_state, best_val, best_epoch, no_imp = None, float("inf"), -1, 0
        for epoch in range(int(train_params["epochs"])):
            model.train()
            perm = np.random.permutation(len(xt))
            bs = int(train_params["batch_size"])
            for i in range(0, len(perm), bs):
                idx = perm[i : i + bs]
                opt.zero_grad()
                loss = ((model(xt[idx]) - yt[idx]) ** 2).mean()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_mse = ((model(xv) - yv) ** 2).mean().item()
            if val_mse < best_val - 1e-9:
                best_val, best_epoch, no_imp = val_mse, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                no_imp += 1
            if no_imp >= int(train_params["patience"]):
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, {
            "best_val_mse": float(best_val),
            "best_epoch": int(best_epoch),
            "y_mean": y_mean,
            "y_std": y_std,
            "add_rv": add_rv,
            "model_key": model_key,
        }

    if model_name in {"GRU", "Transformer"}:
        in_dim = Xl_tr.shape[-1] + Xm_tr.shape[-1]
        if model_name == "GRU":
            model = GRUBaseline(in_dim, int(train_params["hidden"]), n_ccy, usd_idx).to(device)
        else:
            model = TransformerBaseline(in_dim, int(train_params["hidden"]), n_ccy, usd_idx).to(device)
        opt = optim.AdamW(model.parameters(), lr=float(train_params["lr"]), weight_decay=float(train_params["weight_decay"]))
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=max(1, int(train_params["patience"]) // 2), factor=0.5)
        xt = torch.tensor(Xl_tr, dtype=torch.float32, device=device)
        mt = torch.tensor(Xm_tr, dtype=torch.float32, device=device)
        yt = torch.tensor(ytr_n, dtype=torch.float32, device=device)
        xv = torch.tensor(Xl_vl, dtype=torch.float32, device=device)
        mv = torch.tensor(Xm_vl, dtype=torch.float32, device=device)
        yv = torch.tensor(yvl_n, dtype=torch.float32, device=device)
        mask = torch.ones(n_ccy, dtype=torch.bool, device=device)
        mask[usd_idx] = False
        best_state, best_val, best_epoch, no_imp = None, float("inf"), -1, 0
        bs = int(train_params["batch_size"])
        for epoch in range(int(train_params["epochs"])):
            model.train()
            perm = np.random.permutation(len(xt))
            for i in range(0, len(perm), bs):
                idx = perm[i : i + bs]
                opt.zero_grad()
                pred = model(xt[idx], mt[idx])
                loss = ((pred[:, mask] - yt[idx][:, mask]) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                val_pred = model(xv, mv)
                val_mse = ((val_pred[:, mask] - yv[:, mask]) ** 2).mean().item()
            sch.step(val_mse)
            if val_mse < best_val - 1e-9:
                best_val, best_epoch, no_imp = val_mse, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                no_imp += 1
            if no_imp >= int(train_params["patience"]):
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, {
            "best_val_mse": float(best_val),
            "best_epoch": int(best_epoch),
            "y_mean": y_mean,
            "y_std": y_std,
            "add_rv": add_rv,
            "model_key": model_name,
        }

    if model_name == "Ridge":
        xtr = np.concatenate([Xl_tr.reshape(len(Xl_tr), -1), Xm_tr.reshape(len(Xm_tr), -1)], axis=1)
        model = SklearnRidge(alpha=1.0)
        model.fit(xtr, Y_tr)
        return model, {
            "best_val_mse": float("nan"),
            "best_epoch": -1,
            "y_mean": y_mean,
            "y_std": y_std,
            "add_rv": add_rv,
            "model_key": "Ridge",
        }

    cfg = copy.deepcopy(data_bundle["config"])
    if model_name == "GAT":
        cfg.gnn_type = "gat"
    cfg.hidden = int(train_params["hidden"])
    cfg.hybrid_hidden = int(train_params["hidden"])
    cfg.top_k = int(train_params["top_k"])
    model = create_model(model_key, cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=float(train_params["lr"]), weight_decay=float(train_params["weight_decay"]))
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=max(1, int(train_params["patience"]) // 2), factor=0.5)

    xt = torch.tensor(Xl_tr, dtype=torch.float32, device=device)
    mt = torch.tensor(Xm_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr_n, dtype=torch.float32, device=device)
    xv = torch.tensor(Xl_vl, dtype=torch.float32, device=device)
    mv = torch.tensor(Xm_vl, dtype=torch.float32, device=device)
    yv = torch.tensor(yvl_n, dtype=torch.float32, device=device)
    mask = torch.ones(n_ccy, dtype=torch.bool, device=device)
    mask[usd_idx] = False

    best_state, best_val, best_epoch, no_imp = None, float("inf"), -1, 0
    bs = int(train_params["batch_size"])
    for epoch in range(int(train_params["epochs"])):
        model.train()
        perm = np.random.permutation(len(xt))
        for i in range(0, len(perm), bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            pred = model(xt[idx], mt[idx], None)[0]
            loss = ((pred[:, mask] - yt[idx][:, mask]) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(xv, mv, None)[0]
            val_mse = ((val_pred[:, mask] - yv[:, mask]) ** 2).mean().item()
        sch.step(val_mse)
        if val_mse < best_val - 1e-9:
            best_val, best_epoch, no_imp = val_mse, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_imp += 1
        if no_imp >= int(train_params["patience"]):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_val_mse": float(best_val),
        "best_epoch": int(best_epoch),
        "y_mean": y_mean,
        "y_std": y_std,
        "add_rv": add_rv,
        "model_key": model_key,
    }


def predict_model(model, train_artifacts: dict, data_bundle: dict, lookback_mode: Any, device: torch.device | None = None):
    device = device or get_device()
    add_rv = bool(train_artifacts["add_rv"])
    model_key = str(train_artifacts["model_key"])
    y_mean = train_artifacts["y_mean"]
    y_std = train_artifacts["y_std"]

    Xl_te, Xm_te, Y_te, dates_te = build_windows(data_bundle, lookback_mode, "test", add_rv=add_rv, return_dates=True)
    if model_key == "Ridge":
        xte = np.concatenate([Xl_te.reshape(len(Xl_te), -1), Xm_te.reshape(len(Xm_te), -1)], axis=1)
        pred = model.predict(xte)
        return pred.astype(np.float32), Y_te.astype(np.float32), dates_te

    if isinstance(model, (GRUBaseline, TransformerBaseline)):
        model.eval()
        with torch.no_grad():
            pred_norm = (
                model(
                    torch.tensor(Xl_te, dtype=torch.float32, device=device),
                    torch.tensor(Xm_te, dtype=torch.float32, device=device),
                )
                .detach()
                .cpu()
                .numpy()
            )
    else:
        pred_norm = _predict_tensor(model, Xl_te, Xm_te, "MLP" if isinstance(model, FlatMLP) else model_key, device)
    pred = pred_norm * y_std + y_mean
    return pred.astype(np.float32), Y_te.astype(np.float32), dates_te


def compute_pairwise_hit(pred: np.ndarray, target: np.ndarray, n_ccy: int, usd_idx: int) -> float:
    from itertools import combinations

    edges = list(combinations(range(n_ccy), 2))
    pd_ = np.stack([pred[:, i] - pred[:, j] for i, j in edges], axis=1)
    td_ = np.stack([target[:, i] - target[:, j] for i, j in edges], axis=1)
    return float(np.mean(np.sign(pd_) == np.sign(td_)))


def compute_extreme_metrics(pred: np.ndarray, target: np.ndarray, n_ccy: int, usd_idx: int, q: float = 0.90) -> dict:
    mask = np.ones(n_ccy, dtype=bool)
    mask[usd_idx] = False
    p = pred[:, mask].reshape(-1)
    t = target[:, mask].reshape(-1)
    th = np.quantile(np.abs(t), q)
    sel = np.abs(t) >= th
    if not np.any(sel):
        return {"extreme_rmse": float("nan"), "extreme_hit": float("nan")}
    pp = p[sel]
    tt = t[sel]
    return {
        "extreme_rmse": float(np.sqrt(np.mean((pp - tt) ** 2))),
        "extreme_hit": float(np.mean(np.sign(pp) == np.sign(tt))),
    }


def compute_metrics(pred: np.ndarray, target: np.ndarray, n_ccy: int, usd_idx: int, q: float = 0.90) -> dict:
    mask = np.ones(n_ccy, dtype=bool)
    mask[usd_idx] = False
    rmse = float(np.sqrt(((pred[:, mask] - target[:, mask]) ** 2).mean()))
    mae = float(np.abs(pred[:, mask] - target[:, mask]).mean())
    hit_ccy = float(np.mean(np.sign(pred[:, mask]) == np.sign(target[:, mask])))
    hit_pair = compute_pairwise_hit(pred, target, n_ccy, usd_idx)
    ext = compute_extreme_metrics(pred, target, n_ccy, usd_idx, q=q)
    ic = information_coefficient(pred, target, mask=mask)
    ls = long_short_sharpe(pred, target, k=3, mask=mask)
    return {
        "rmse": rmse,
        "mae": mae,
        "hit_ccy": hit_ccy,
        "hit_pair": hit_pair,
        "extreme_rmse": float(ext["extreme_rmse"]),
        "extreme_hit": float(ext["extreme_hit"]),
        "ic": float(ic),
        "sharpe": float(ls["sharpe"]),
    }


def aggregate_metrics(df: pd.DataFrame, metric_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
    recs = []
    for keys, g in df.groupby(group_cols):
        out = {}
        if isinstance(keys, tuple):
            for k, v in zip(group_cols, keys):
                out[k] = v
        else:
            out[group_cols[0]] = keys
        for c in metric_cols:
            out[f"{c}_mean"] = float(g[c].mean())
            out[f"{c}_std"] = float(g[c].std(ddof=0))
        recs.append(out)
    return pd.DataFrame(recs)


def significance_tests(
    pred_records: dict[tuple, tuple[np.ndarray, np.ndarray]],
    comparisons: list[tuple[str, str]],
) -> pd.DataFrame:
    rows = []
    for (left, right) in comparisons:
        common_keys = sorted([k for k in pred_records if k[0] == left and (right, k[1], k[2]) in pred_records])
        for ck in common_keys:
            rk = (right, ck[1], ck[2])
            p1, t = pred_records[ck]
            p2, _ = pred_records[rk]
            e1 = ((p1 - t) ** 2).mean(axis=1)
            e2 = ((p2 - t) ** 2).mean(axis=1)
            d = e1 - e2
            if len(d) < 2:
                continue
            mu = float(np.mean(d))
            se = float(np.std(d, ddof=1) / np.sqrt(len(d)) + 1e-12)
            stat = mu / se
            pval = 2 * (1 - scipy_stats.norm.cdf(abs(stat)))
            sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else "n.s."))
            rows.append(
                {
                    "comparison": f"{left}_vs_{right}",
                    "metric": "paired_sq_error",
                    "test_stat": float(stat),
                    "p_value": float(pval),
                    "significance": sig,
                    "lookback": ck[1],
                    "seed": ck[2],
                }
            )
    return pd.DataFrame(rows)


def config_to_dict(cfg: Config) -> dict:
    try:
        return asdict(cfg)
    except Exception:
        return dict(cfg.__dict__)
