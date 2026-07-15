from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from .data_pipeline import (
        MultiBlockSequenceDataset,
        PreparedData,
        collate_batch,
        create_splits,
        inverse_transform,
        set_seed,
    )
    from .metrics import compute_metrics, save_dataframe
    from .models_baselines import build_baseline_model
    from .models_ours import create_relational_model, small_return_cls_loss, stable_scoregraph_loss
except ImportError:
    from data_pipeline import (
        MultiBlockSequenceDataset,
        PreparedData,
        collate_batch,
        create_splits,
        inverse_transform,
        set_seed,
    )
    from metrics import compute_metrics, save_dataframe
    from models_baselines import build_baseline_model
    from models_ours import create_relational_model, small_return_cls_loss, stable_scoregraph_loss


def build_selection_score(selection_metric: str, val_loss: float, metrics: Dict[str, float], hit_alpha: float) -> float:
    if selection_metric == "mse":
        return val_loss
    if selection_metric == "hit":
        return -metrics["hit_ratio"]
    if selection_metric == "sharpe":
        return -metrics["long_short_sharpe"]
    return val_loss - hit_alpha * metrics["hit_ratio"]


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor, non_usd_mask: torch.Tensor) -> torch.Tensor:
    pred = pred[:, non_usd_mask]
    target = target[:, non_usd_mask]
    if pred.size(1) < 2:
        return pred.new_tensor(0.0)
    losses = []
    for i in range(pred.size(1)):
        for j in range(i + 1, pred.size(1)):
            losses.append(F.softplus(-((pred[:, i] - pred[:, j]) * (target[:, i] - target[:, j]))))
    return torch.stack(losses, dim=0).mean()


def baseline_loss(pred_norm: torch.Tensor, y_norm: torch.Tensor, non_usd_mask: torch.Tensor, lambda_dir: float, lambda_rank: float) -> torch.Tensor:
    pred = pred_norm[:, non_usd_mask]
    target = y_norm[:, non_usd_mask]
    return F.mse_loss(pred, target) + lambda_dir * F.softplus(-(pred * target)).mean() + lambda_rank * pairwise_rank_loss(pred_norm, y_norm, non_usd_mask)


def build_combined_tensor(prepared: PreparedData) -> np.ndarray:
    t = prepared.x_local.shape[0]
    n = prepared.x_local.shape[1]
    x_global = np.repeat(prepared.x_global[:, None, :], n, axis=1)
    return np.concatenate([prepared.x_local, prepared.x_rate, prepared.x_equity, prepared.x_countrymacro, x_global], axis=-1).astype(np.float32)


class SequenceBaselineDataset(Dataset):
    def __init__(self, x_all: np.ndarray, y_norm: np.ndarray, y_raw: np.ndarray, dates: pd.Series, lookback: int):
        self.x_all = x_all
        self.y_norm = y_norm
        self.y_raw = y_raw
        self.dates = pd.to_datetime(dates).reset_index(drop=True)
        self.lookback = int(lookback)

    def __len__(self) -> int:
        return len(self.y_raw) - self.lookback

    def __getitem__(self, idx: int):
        end = idx + self.lookback
        x = torch.tensor(self.x_all[idx:end], dtype=torch.float32)
        target = {"y_norm": torch.tensor(self.y_norm[end], dtype=torch.float32), "y_raw": torch.tensor(self.y_raw[end], dtype=torch.float32)}
        meta = {"input_end_date": str(self.dates.iloc[end - 1].date()), "target_date": str(self.dates.iloc[end].date())}
        return x, target, meta


def collate_sequence(batch):
    xs, targets, metas = zip(*batch)
    x = torch.stack(xs, dim=0)
    y_norm = torch.stack([t["y_norm"] for t in targets], dim=0)
    y_raw = torch.stack([t["y_raw"] for t in targets], dim=0)
    return x, {"y_norm": y_norm, "y_raw": y_raw}, list(metas)


@dataclass
class TrainResult:
    raw_metrics: Dict[str, float]
    prediction_df: pd.DataFrame
    component_df: pd.DataFrame
    edge_df: pd.DataFrame


def compute_small_return_threshold(prepared: PreparedData, split: Sequence[float], quantile: float) -> float:
    train_end = int(len(prepared.merged) * split[0])
    vals = np.abs(prepared.y_norm[:train_end, 1:].reshape(-1))
    vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, quantile)) if len(vals) else 0.0


def train_relational_model(model_name: str, prepared: PreparedData, args, output_dir: Path, device: str) -> TrainResult:
    dataset = MultiBlockSequenceDataset(prepared, args.lookback)
    train_ds, val_ds, test_ds = create_splits(dataset, args.split)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    non_usd_mask_np = np.array([c != "USD" for c in prepared.currency_names], dtype=bool)
    non_usd_mask_t = torch.tensor(non_usd_mask_np, dtype=torch.bool, device=device)
    config = {
        "n_ccy": len(prepared.currency_names),
        "currency_names": prepared.currency_names,
        "usd_idx": 0,
        "dims": prepared.dims,
        "hidden": args.hidden,
        "top_k": args.top_k,
        "dropout": args.dropout,
        "graph_rank": args.graph_rank,
        "edge_dropout": args.edge_dropout,
        "spectral_bound": args.spectral_bound,
        "component_gate_type": args.component_gate_type,
    }
    model = create_relational_model(model_name, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    threshold = compute_small_return_threshold(prepared, args.split, args.small_return_quantile)
    best_state = None
    best_score = float("inf")
    no_improve = 0
    prev_adj = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch, target, _ in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if prev_adj is not None:
                batch["A_prev"] = prev_adj
            y_norm = target["y_norm"].to(device)
            out = model(batch)
            if model_name == "oursmain":
                loss = small_return_cls_loss(out, y_norm, non_usd_mask_t, threshold, args.lambda_component, args.lambda_smooth, args.lambda_static, args.lambda_sparse, args.lambda_spectral)
            else:
                loss = stable_scoregraph_loss(out, y_norm, non_usd_mask_t, args.lambda_dir, args.lambda_rank, args.lambda_component, args.lambda_smooth, args.lambda_static, args.lambda_sparse, args.lambda_spectral, prepared.q80_abs_y_train)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            prev_adj = out["adj"].detach()
        val_loss, val_metrics = evaluate_relational(model, val_loader, device, prepared, non_usd_mask_t, args, threshold, model_name)
        scheduler.step(val_loss)
        score = build_selection_score(args.selection_metric, val_loss, val_metrics, args.hit_alpha)
        if score < best_score - 1e-8:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    pred_norm_np, y_raw_np, pred_df, comp_df, edge_df = collect_relational_outputs(model, test_loader, device, prepared, model_name, args)
    pred_raw_np = inverse_transform(pred_norm_np, prepared.y_mean, prepared.y_std)
    metrics = compute_metrics(pred_raw_np, y_raw_np, non_usd_mask_np)
    metrics.update({"model": model_name, "universe": args.universe, "lookback": args.lookback, "seed": args.seed})
    save_dataframe(pred_df, output_dir / "predictions" / f"{model_name}_predictions.parquet")
    save_dataframe(comp_df, output_dir / "explanations" / f"{model_name}_components.parquet")
    save_dataframe(edge_df, output_dir / "explanations" / f"{model_name}_top_edges.parquet")
    return TrainResult(metrics, pred_df, comp_df, edge_df)


def evaluate_relational(model, loader, device, prepared, non_usd_mask_t, args, threshold: float, model_name: str) -> Tuple[float, Dict[str, float]]:
    losses = []
    pred_norm_list = []
    y_raw_list = []
    prev_adj = None
    model.eval()
    with torch.no_grad():
        for batch, target, _ in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if prev_adj is not None:
                batch["A_prev"] = prev_adj
            y_norm = target["y_norm"].to(device)
            out = model(batch)
            loss = small_return_cls_loss(out, y_norm, non_usd_mask_t, threshold, args.lambda_component, args.lambda_smooth, args.lambda_static, args.lambda_sparse, args.lambda_spectral) if model_name == "oursmain" else stable_scoregraph_loss(out, y_norm, non_usd_mask_t, args.lambda_dir, args.lambda_rank, args.lambda_component, args.lambda_smooth, args.lambda_static, args.lambda_sparse, args.lambda_spectral, prepared.q80_abs_y_train)
            losses.append(float(loss.item()))
            pred_norm_list.append(out["rhat"].detach().cpu().numpy())
            y_raw_list.append(target["y_raw"].numpy())
            prev_adj = out["adj"].detach()
    pred_raw = inverse_transform(np.concatenate(pred_norm_list, axis=0), prepared.y_mean, prepared.y_std)
    y_raw = np.concatenate(y_raw_list, axis=0)
    return float(np.mean(losses)), compute_metrics(pred_raw, y_raw, non_usd_mask_t.cpu().numpy())


def collect_relational_outputs(model, loader, device, prepared: PreparedData, model_name: str, args):
    pred_norm_all, y_raw_all, pred_rows, comp_rows, edge_rows = [], [], [], [], []
    prev_adj = None
    model.eval()
    with torch.no_grad():
        for batch, target, metas in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if prev_adj is not None:
                batch["A_prev"] = prev_adj
            out = model(batch)
            pred_norm = out["rhat"].detach().cpu().numpy()
            pred_raw = inverse_transform(pred_norm, prepared.y_mean, prepared.y_std)
            y_raw = target["y_raw"].numpy()
            pred_norm_all.append(pred_norm)
            y_raw_all.append(y_raw)
            ds_np = out["ds"].detach().cpu().numpy()
            adj_np = out["adj"].detach().cpu().numpy()
            logit_np = out["edge_logits"].detach().cpu().numpy()
            contrib_np = out["edge_contrib"].detach().cpu().numpy()
            comp_np = {k: v.detach().cpu().numpy() for k, v in out["components"].items()}
            gate_np = {k: v.detach().cpu().numpy() for k, v in out["component_gates"].items()}
            graph_gate_np = out["graph_gate"].detach().cpu().numpy()
            for b_idx, meta in enumerate(metas):
                for c_idx, ccy in enumerate(prepared.currency_names):
                    pred_rows.append({"target_date": meta["target_date"], "input_end_date": meta["input_end_date"], "model": model_name, "currency": ccy, "pred": pred_raw[b_idx, c_idx], "target": y_raw[b_idx, c_idx], "seed": args.seed})
                    comp_rows.append({"target_date": meta["target_date"], "input_end_date": meta["input_end_date"], "model": model_name, "currency": ccy, "pred": pred_raw[b_idx, c_idx], "target": y_raw[b_idx, c_idx], "ds": ds_np[b_idx, c_idx], "usd_ds": ds_np[b_idx, 0], "c_local": comp_np["local"][b_idx, c_idx], "c_rate": comp_np["rate"][b_idx, c_idx], "c_equity": comp_np["equity"][b_idx, c_idx], "c_macro": comp_np["macro"][b_idx, c_idx], "c_macro_global": comp_np["macro_global"][b_idx, c_idx], "c_macro_country": comp_np["macro_country"][b_idx, c_idx], "c_rel": comp_np["rel"][b_idx, c_idx], "g_local": gate_np["local"][b_idx, c_idx], "g_rate": gate_np["rate"][b_idx, c_idx], "g_equity": gate_np["equity"][b_idx, c_idx], "g_macro": gate_np["macro"][b_idx, c_idx], "g_rel": gate_np["rel"][b_idx, c_idx], "g_graph": graph_gate_np[b_idx, c_idx], "seed": args.seed})
                for target_idx, tgt_ccy in enumerate(prepared.currency_names):
                    order = np.argsort(np.abs(adj_np[b_idx, target_idx]))[::-1]
                    edge_rank = 0
                    for source_idx in order:
                        if source_idx == target_idx or abs(adj_np[b_idx, target_idx, source_idx]) <= 0:
                            continue
                        edge_rank += 1
                        edge_rows.append({"target_date": meta["target_date"], "model": model_name, "source_currency": prepared.currency_names[source_idx], "target_currency": tgt_ccy, "edge_weight": adj_np[b_idx, target_idx, source_idx], "edge_logit": logit_np[b_idx, target_idx, source_idx], "edge_contribution": contrib_np[b_idx, target_idx, source_idx], "rank": edge_rank, "seed": args.seed})
            prev_adj = out["adj"].detach()
    return np.concatenate(pred_norm_all, axis=0), np.concatenate(y_raw_all, axis=0), pd.DataFrame(pred_rows), pd.DataFrame(comp_rows), pd.DataFrame(edge_rows)


def train_baseline_model(model_name: str, prepared: PreparedData, args, output_dir: Path, device: str) -> TrainResult:
    x_all = build_combined_tensor(prepared)
    dataset = SequenceBaselineDataset(x_all, prepared.y_norm, prepared.y_raw, prepared.merged["Date"], args.lookback)
    train_ds, val_ds, test_ds = create_splits(dataset, args.split)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_sequence)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_sequence)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_sequence)
    non_usd_mask_np = np.array([c != "USD" for c in prepared.currency_names], dtype=bool)
    non_usd_mask_t = torch.tensor(non_usd_mask_np, dtype=torch.bool, device=device)
    model = build_baseline_model(model_name, x_all.shape[-1], args.hidden, len(prepared.currency_names), args.lookback, args.top_k, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    best_state = None
    best_score = float("inf")
    no_improve = 0
    for _ in range(args.epochs):
        model.train()
        for x, target, _ in train_loader:
            x = x.to(device)
            y_norm = target["y_norm"].to(device)
            loss = baseline_loss(model(x), y_norm, non_usd_mask_t, args.lambda_dir, args.lambda_rank)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_loss, val_metrics = evaluate_baseline(model, val_loader, device, prepared, non_usd_mask_t, args)
        scheduler.step(val_loss)
        score = build_selection_score(args.selection_metric, val_loss, val_metrics, args.hit_alpha)
        if score < best_score - 1e-8:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    pred_rows, pred_norm_all, y_raw_all = [], [], []
    with torch.no_grad():
        for x, target, metas in test_loader:
            x = x.to(device)
            pred_norm = model(x).detach().cpu().numpy()
            pred_raw = inverse_transform(pred_norm, prepared.y_mean, prepared.y_std)
            y_raw = target["y_raw"].numpy()
            pred_norm_all.append(pred_norm)
            y_raw_all.append(y_raw)
            for t, meta in enumerate(metas):
                for c_idx, ccy in enumerate(prepared.currency_names):
                    pred_rows.append({"target_date": meta["target_date"], "input_end_date": meta["input_end_date"], "model": model_name, "currency": ccy, "pred": pred_raw[t, c_idx], "target": y_raw[t, c_idx], "seed": args.seed})
    pred_df = pd.DataFrame(pred_rows)
    save_dataframe(pred_df, output_dir / "predictions" / f"{model_name}_predictions.parquet")
    pred_raw = inverse_transform(np.concatenate(pred_norm_all, axis=0), prepared.y_mean, prepared.y_std)
    metrics = compute_metrics(pred_raw, np.concatenate(y_raw_all, axis=0), non_usd_mask_np)
    metrics.update({"model": model_name, "universe": args.universe, "lookback": args.lookback, "seed": args.seed})
    return TrainResult(metrics, pred_df, pd.DataFrame(), pd.DataFrame())


def evaluate_baseline(model, loader, device, prepared, non_usd_mask_t, args):
    losses, pred_norm_list, y_raw_list = [], [], []
    model.eval()
    with torch.no_grad():
        for x, target, _ in loader:
            x = x.to(device)
            y_norm = target["y_norm"].to(device)
            pred_norm = model(x)
            losses.append(float(baseline_loss(pred_norm, y_norm, non_usd_mask_t, args.lambda_dir, args.lambda_rank).item()))
            pred_norm_list.append(pred_norm.detach().cpu().numpy())
            y_raw_list.append(target["y_raw"].numpy())
    pred_raw = inverse_transform(np.concatenate(pred_norm_list, axis=0), prepared.y_mean, prepared.y_std)
    y_raw = np.concatenate(y_raw_list, axis=0)
    return float(np.mean(losses)), compute_metrics(pred_raw, y_raw, non_usd_mask_t.cpu().numpy())
