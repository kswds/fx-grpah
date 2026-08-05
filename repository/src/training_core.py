from __future__ import annotations

import copy
from dataclasses import dataclass
from time import perf_counter
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
    from .models_baselines import build_baseline_model, build_corrlstmgat_static_graph
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
    from models_baselines import build_baseline_model, build_corrlstmgat_static_graph
    from models_ours import create_relational_model, small_return_cls_loss, stable_scoregraph_loss

try:
    from xgboost import XGBRegressor  # type: ignore
except ImportError:
    XGBRegressor = None

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


def count_trainable_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def maybe_save_checkpoint(state_dict, output_dir: Path, filename: str, enabled: bool) -> None:
    if not enabled or state_dict is None:
        return
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, ckpt_dir / filename)


def build_selection_score(selection_metric: str, val_loss: float, metrics: Dict[str, float], hit_alpha: float) -> float:
    if selection_metric == "mse":
        return val_loss
    if selection_metric == "hit":
        return -metrics["hit_ratio"]
    if selection_metric == "sharpe":
        return -metrics["long_short_sharpe"]
    return val_loss - hit_alpha * metrics["hit_ratio"]


def resolve_relational_loss_variant(model_name: str, args) -> str:
    explicit_rel = getattr(args, "relational_loss_variant", None)
    if explicit_rel:
        return str(explicit_rel).lower()
    explicit = getattr(args, "loss_variant", None)
    if explicit:
        return str(explicit).lower()
    if model_name == "oursmain":
        return "oursmain"
    if model_name == "oursmain2":
        return "full"
    return "full"


def compute_relational_loss(model_name: str, out, y_norm: torch.Tensor, non_usd_mask_t: torch.Tensor, prepared: PreparedData, args, threshold: float) -> torch.Tensor:
    variant = resolve_relational_loss_variant(model_name, args)
    pred = out["rhat"][:, non_usd_mask_t]
    target = y_norm[:, non_usd_mask_t]
    reg = out["regularization"]
    disable_graph_regularization = model_name.lower() in {"foundation_nograph"}
    loss_component = sum(v.pow(2).mean() for v in out["components"].values())
    if variant == "oursmain":
        active = target.abs() >= threshold
        loss_core = F.softplus(-(pred[active] * torch.sign(target[active]))).mean() if active.any() else F.mse_loss(pred, target)
        graph_penalty = pred.new_tensor(0.0) if disable_graph_regularization else (
            args.lambda_smooth * reg["smoothness"]
            + args.lambda_static * reg["static_deviation"]
            + args.lambda_sparse * reg["sparsity"]
            + args.lambda_spectral * reg["spectral"]
        )
        return loss_core + args.lambda_component * loss_component + graph_penalty
    if variant == "active_core":
        active = target.abs() >= threshold
        return F.softplus(-(pred[active] * torch.sign(target[active]))).mean() if active.any() else F.mse_loss(pred, target)
    if variant == "dir_only":
        return F.softplus(-(pred * torch.sign(target))).mean()
    if variant == "mse":
        return F.mse_loss(pred, target)
    if variant == "mse_dir":
        loss_mse = F.mse_loss(pred, target)
        w = torch.clamp(target.abs() / max(float(prepared.q80_abs_y_train), 1e-6), max=1.0)
        loss_dir = (w * F.softplus(-(pred * target))).mean()
        return loss_mse + args.lambda_dir * loss_dir
    if variant == "mse_rank":
        loss_mse = F.mse_loss(pred, target)
        if pred.size(1) < 2:
            loss_rank = pred.new_tensor(0.0)
        else:
            loss_rank = F.softplus(-((pred[:, :, None] - pred[:, None, :]) * (target[:, :, None] - target[:, None, :]))).mean()
        return loss_mse + args.lambda_rank * loss_rank
    if variant == "baseline":
        return baseline_loss(out["rhat"], y_norm, non_usd_mask_t, args.lambda_dir, args.lambda_rank)
    return stable_scoregraph_loss(out, y_norm, non_usd_mask_t, args.lambda_dir, args.lambda_rank, args.lambda_component, args.lambda_smooth, args.lambda_static, args.lambda_sparse, args.lambda_spectral, prepared.q80_abs_y_train)


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


def resolve_baseline_loss_variant(args) -> str:
    explicit = getattr(args, "baseline_loss_variant", None)
    if explicit:
        return str(explicit).lower()
    explicit_rel = getattr(args, "relational_loss_variant", None)
    if explicit_rel in {"baseline", "oursmain", "mse", "dir_only"}:
        return str(explicit_rel).lower()
    return "baseline"


def compute_baseline_training_loss(
    pred_norm: torch.Tensor,
    y_norm: torch.Tensor,
    non_usd_mask: torch.Tensor,
    args,
    threshold: float,
) -> torch.Tensor:
    variant = resolve_baseline_loss_variant(args)
    pred = pred_norm[:, non_usd_mask]
    target = y_norm[:, non_usd_mask]
    if variant == "oursmain":
        active = target.abs() >= threshold
        return F.softplus(-(pred[active] * torch.sign(target[active]))).mean() if active.any() else F.mse_loss(pred, target)
    if variant == "mse":
        return F.mse_loss(pred, target)
    if variant == "dir_only":
        return F.softplus(-(pred * torch.sign(target))).mean()
    return baseline_loss(pred_norm, y_norm, non_usd_mask, args.lambda_dir, args.lambda_rank)


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
    val_metrics: Dict[str, float]
    best_val_loss: float
    best_selection_score: float
    train_seconds: float


def save_baseline_metadata(output_dir: Path, records: List[Dict[str, object]]) -> None:
    if not records:
        return
    save_dataframe(pd.DataFrame(records), output_dir / "metadata" / "selected_hparams.csv")


def compute_small_return_threshold(prepared: PreparedData, split: Sequence[float], quantile: float) -> float:
    train_end = int(len(prepared.merged) * split[0])
    vals = np.abs(prepared.y_norm[:train_end, 1:].reshape(-1))
    vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, quantile)) if len(vals) else 0.0


def save_corrlstmgat_graph_artifacts(output_dir: Path, graph_info: Dict[str, object], currencies: Sequence[str]) -> None:
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    corr = np.asarray(graph_info["correlation_matrix"], dtype=float)
    adj = np.asarray(graph_info["adjacency_matrix"], dtype=float)
    corr_df = pd.DataFrame(corr, index=list(currencies), columns=list(currencies)).reset_index().rename(columns={"index": "currency"})
    adj_df = pd.DataFrame(adj, index=list(currencies), columns=list(currencies)).reset_index().rename(columns={"index": "currency"})
    edges = []
    for row in graph_info["edge_list"]:
        src_idx = int(row["src_idx"])
        dst_idx = int(row["dst_idx"])
        edges.append(
            {
                "source_currency": currencies[src_idx],
                "target_currency": currencies[dst_idx],
                "corr": float(row["corr"]),
            }
        )
    stats_df = pd.DataFrame(
        [
            {
                "graph_density": float(graph_info["graph_density"]),
                "isolated_count_before_self_loops": int(graph_info["isolated_count_before_self_loops"]),
                "threshold": 0.7,
                "n_nonusd_currencies": len(currencies),
            }
        ]
    )
    save_dataframe(corr_df, meta_dir / "corrlstmgat_correlation_matrix.csv")
    save_dataframe(adj_df, meta_dir / "corrlstmgat_adjacency_matrix.csv")
    save_dataframe(pd.DataFrame(edges), meta_dir / "corrlstmgat_edge_list.csv")
    save_dataframe(stats_df, meta_dir / "corrlstmgat_graph_stats.csv")


def train_relational_model(model_name: str, prepared: PreparedData, args, output_dir: Path, device: str) -> TrainResult:
    train_start = perf_counter()
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
    best_val_loss = float("inf")
    best_val_metrics: Dict[str, float] | None = None
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
            loss = compute_relational_loss(model_name, out, y_norm, non_usd_mask_t, prepared, args, threshold)
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
            best_val_loss = float(val_loss)
            best_val_metrics = dict(val_metrics)
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    maybe_save_checkpoint(best_state, output_dir, f"{model_name}_best.pt", bool(getattr(args, "save_checkpoints", False)))
    pred_norm_np, y_raw_np, pred_df, comp_df, edge_df = collect_relational_outputs(model, test_loader, device, prepared, model_name, args)
    pred_raw_np = inverse_transform(pred_norm_np, prepared.y_mean, prepared.y_std)
    metrics = compute_metrics(pred_raw_np, y_raw_np, non_usd_mask_np)
    metrics.update({"model": model_name, "universe": args.universe, "lookback": args.lookback, "seed": args.seed})
    save_dataframe(pred_df, output_dir / "predictions" / f"{model_name}_predictions.parquet")
    save_dataframe(comp_df, output_dir / "explanations" / f"{model_name}_components.parquet")
    save_dataframe(edge_df, output_dir / "explanations" / f"{model_name}_top_edges.parquet")
    val_metrics_out = dict(best_val_metrics or {})
    val_metrics_out.update({"model": model_name, "universe": args.universe, "lookback": args.lookback, "seed": args.seed})
    return TrainResult(metrics, pred_df, comp_df, edge_df, val_metrics_out, best_val_loss, best_score, perf_counter() - train_start)


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
            loss = compute_relational_loss(model_name, out, y_norm, non_usd_mask_t, prepared, args, threshold)
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
    train_start = perf_counter()
    x_all = build_combined_tensor(prepared)
    dataset = SequenceBaselineDataset(x_all, prepared.y_norm, prepared.y_raw, prepared.merged["Date"], args.lookback)
    train_ds, val_ds, test_ds = create_splits(dataset, args.split)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_sequence)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_sequence)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_sequence)
    non_usd_mask_np = np.array([c != "USD" for c in prepared.currency_names], dtype=bool)
    non_usd_mask_t = torch.tensor(non_usd_mask_np, dtype=torch.bool, device=device)
    if model_name.lower() == "xgboost":
        return train_tree_boosting_baseline(prepared, args, output_dir, non_usd_mask_np, non_usd_mask_t, x_all, dataset, train_ds, val_ds, test_ds, train_start)
    if model_name.lower() == "timexer":
        patch_candidates = [2, 5]
        best_pack = None
        metadata_rows = []
        for patch_len in patch_candidates:
            candidate = build_baseline_model(model_name, x_all.shape[-1], args.hidden, len(prepared.currency_names), args.lookback, args.top_k, args.dropout, patch_len=patch_len).to(device)
            result = _fit_torch_baseline_model(candidate, model_name, prepared, args, output_dir, device, train_loader, val_loader, test_loader, non_usd_mask_np, non_usd_mask_t)
            metadata_rows.append(
                {
                    "model": model_name,
                    "seed": args.seed,
                    "patch_len": patch_len,
                    "selected": 0,
                    "n_trainable_params": count_trainable_params(candidate),
                    "best_val_loss": result.best_val_loss,
                    "best_selection_score": result.best_selection_score,
                    "train_seconds": result.train_seconds,
                }
            )
            if best_pack is None or result.best_selection_score < best_pack[0]:
                best_pack = (result.best_selection_score, result, patch_len, count_trainable_params(candidate))
        assert best_pack is not None
        _, best_result, best_patch_len, best_param_count = best_pack
        for row in metadata_rows:
            row["selected"] = int(row["patch_len"] == best_patch_len)
        save_baseline_metadata(output_dir, metadata_rows)
        save_dataframe(best_result.prediction_df, output_dir / "predictions" / f"{model_name}_predictions.parquet")
        return best_result
    model_kwargs: Dict[str, object] = {}
    if model_name.lower() in {"corr_lstm_gat", "corrlstmgat"}:
        train_end = int(len(prepared.merged) * args.split[0])
        graph_info = build_corrlstmgat_static_graph(prepared.y_raw[:train_end, 1:], threshold=0.7)
        model_kwargs["adjacency"] = graph_info["adjacency_matrix"]
        model_kwargs["architecture_order"] = getattr(args, "corr_architecture_order", "lstm_then_gat")
        model_kwargs["usd_idx"] = 0
        save_corrlstmgat_graph_artifacts(output_dir, graph_info, prepared.currency_names[1:])
    elif model_name.lower() in {"fxrp", "fxir_edge_gnn"}:
        model_kwargs["usd_idx"] = 0
        model_kwargs["num_layers"] = int(getattr(args, "fxrp_num_layers", 3))
    model = build_baseline_model(
        model_name,
        x_all.shape[-1],
        args.hidden,
        len(prepared.currency_names),
        args.lookback,
        args.top_k,
        args.dropout,
        **model_kwargs,
    ).to(device)
    result = _fit_torch_baseline_model(model, model_name, prepared, args, output_dir, device, train_loader, val_loader, test_loader, non_usd_mask_np, non_usd_mask_t)
    save_dataframe(result.prediction_df, output_dir / "predictions" / f"{model_name}_predictions.parquet")
    save_baseline_metadata(
        output_dir,
        [
            {
                "model": model_name,
                "seed": args.seed,
                "patch_len": pd.NA,
                "selected": 1,
                "n_trainable_params": count_trainable_params(model),
                "best_val_loss": result.best_val_loss,
                "best_selection_score": result.best_selection_score,
                "train_seconds": result.train_seconds,
            }
        ],
    )
    return result


def _fit_torch_baseline_model(model, model_name: str, prepared: PreparedData, args, output_dir: Path, device: str, train_loader, val_loader, test_loader, non_usd_mask_np: np.ndarray, non_usd_mask_t: torch.Tensor) -> TrainResult:
    train_start = perf_counter()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay) if trainable_params else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3) if optimizer is not None else None
    threshold = compute_small_return_threshold(prepared, args.split, args.small_return_quantile)
    best_state = None
    best_score = float("inf")
    best_val_loss = float("inf")
    best_val_metrics: Dict[str, float] | None = None
    no_improve = 0
    if optimizer is None:
        val_loss, val_metrics = evaluate_baseline(model, val_loader, device, prepared, non_usd_mask_t, args)
        best_score = build_selection_score(args.selection_metric, val_loss, val_metrics, args.hit_alpha)
        best_val_loss = float(val_loss)
        best_val_metrics = dict(val_metrics)
    else:
        for _ in range(args.epochs):
            model.train()
            for x, target, _ in train_loader:
                x = x.to(device)
                y_norm = target["y_norm"].to(device)
                loss = compute_baseline_training_loss(model(x), y_norm, non_usd_mask_t, args, threshold)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            val_loss, val_metrics = evaluate_baseline(model, val_loader, device, prepared, non_usd_mask_t, args, threshold)
            scheduler.step(val_loss)
            score = build_selection_score(args.selection_metric, val_loss, val_metrics, args.hit_alpha)
            if score < best_score - 1e-8:
                best_score = score
                best_val_loss = float(val_loss)
                best_val_metrics = dict(val_metrics)
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    maybe_save_checkpoint(best_state, output_dir, f"{model_name}_best.pt", bool(getattr(args, "save_checkpoints", False)))
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
    pred_raw = inverse_transform(np.concatenate(pred_norm_all, axis=0), prepared.y_mean, prepared.y_std)
    metrics = compute_metrics(pred_raw, np.concatenate(y_raw_all, axis=0), non_usd_mask_np)
    metrics.update({"model": model_name, "universe": args.universe, "lookback": args.lookback, "seed": args.seed})
    val_metrics_out = dict(best_val_metrics or {})
    val_metrics_out.update({"model": model_name, "universe": args.universe, "lookback": args.lookback, "seed": args.seed})
    return TrainResult(metrics, pred_df, pd.DataFrame(), pd.DataFrame(), val_metrics_out, best_val_loss, best_score, perf_counter() - train_start)


def _subset_xy(dataset: SequenceBaselineDataset, subset: Subset) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rows_x, rows_y_norm, rows_y_raw, metas = [], [], [], []
    for idx in subset.indices:
        x, target, meta = dataset[idx]
        rows_x.append(x.numpy())
        rows_y_norm.append(target["y_norm"].numpy())
        rows_y_raw.append(target["y_raw"].numpy())
        metas.append(meta)
    return np.stack(rows_x, axis=0), np.stack(rows_y_norm, axis=0), np.stack(rows_y_raw, axis=0), metas


def _flatten_sequence_features(x_seq: np.ndarray) -> np.ndarray:
    b, l, n, f = x_seq.shape
    return x_seq.transpose(0, 2, 1, 3).reshape(b, n * l * f)


def _build_xgboost_estimator(args):
    if XGBRegressor is not None:
        base = XGBRegressor(
            n_estimators=max(50, min(300, args.epochs * 5)),
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            reg_lambda=1.0,
            random_state=args.seed,
            n_jobs=1,
        )
        return MultiOutputRegressor(base), "xgboost"
    fallback = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=4,
        max_iter=max(100, min(400, args.epochs * 8)),
        random_state=args.seed,
    )
    return MultiOutputRegressor(fallback), "hist_gradient_boosting_fallback"


def train_tree_boosting_baseline(prepared: PreparedData, args, output_dir: Path, non_usd_mask_np: np.ndarray, non_usd_mask_t: torch.Tensor, x_all: np.ndarray, dataset: SequenceBaselineDataset, train_ds: Subset, val_ds: Subset, test_ds: Subset, train_start: float) -> TrainResult:
    x_train_seq, y_train_norm, _, _ = _subset_xy(dataset, train_ds)
    x_val_seq, y_val_norm, y_val_raw, _ = _subset_xy(dataset, val_ds)
    x_test_seq, _, y_test_raw, test_metas = _subset_xy(dataset, test_ds)
    x_train = _flatten_sequence_features(x_train_seq)
    x_val = _flatten_sequence_features(x_val_seq)
    x_test = _flatten_sequence_features(x_test_seq)
    estimator, backend = _build_xgboost_estimator(args)
    estimator.fit(x_train, y_train_norm)
    pred_val_norm = estimator.predict(x_val).astype(np.float32)
    pred_test_norm = estimator.predict(x_test).astype(np.float32)
    threshold = compute_small_return_threshold(prepared, args.split, args.small_return_quantile)
    val_loss = float(
        compute_baseline_training_loss(
            torch.tensor(pred_val_norm, dtype=torch.float32),
            torch.tensor(y_val_norm, dtype=torch.float32),
            non_usd_mask_t.cpu(),
            args,
            threshold,
        ).item()
    )
    pred_val_raw = inverse_transform(pred_val_norm, prepared.y_mean, prepared.y_std)
    val_metrics = compute_metrics(pred_val_raw, y_val_raw, non_usd_mask_np)
    best_score = build_selection_score(args.selection_metric, val_loss, val_metrics, args.hit_alpha)
    pred_test_raw = inverse_transform(pred_test_norm, prepared.y_mean, prepared.y_std)
    pred_rows = []
    for t, meta in enumerate(test_metas):
        for c_idx, ccy in enumerate(prepared.currency_names):
            pred_rows.append(
                {
                    "target_date": meta["target_date"],
                    "input_end_date": meta["input_end_date"],
                    "model": "xgboost",
                    "currency": ccy,
                    "pred": pred_test_raw[t, c_idx],
                    "target": y_test_raw[t, c_idx],
                    "seed": args.seed,
                    "backend": backend,
                }
            )
    pred_df = pd.DataFrame(pred_rows)
    save_dataframe(pred_df, output_dir / "predictions" / "xgboost_predictions.parquet")
    metrics = compute_metrics(pred_test_raw, y_test_raw, non_usd_mask_np)
    metrics.update({"model": "xgboost", "universe": args.universe, "lookback": args.lookback, "seed": args.seed})
    val_metrics_out = dict(val_metrics)
    val_metrics_out.update({"model": "xgboost", "universe": args.universe, "lookback": args.lookback, "seed": args.seed})
    return TrainResult(metrics, pred_df, pd.DataFrame(), pd.DataFrame(), val_metrics_out, val_loss, best_score, perf_counter() - train_start)


def evaluate_baseline(model, loader, device, prepared, non_usd_mask_t, args, threshold: float | None = None):
    losses, pred_norm_list, y_raw_list = [], [], []
    model.eval()
    if threshold is None:
        threshold = compute_small_return_threshold(prepared, args.split, args.small_return_quantile)
    with torch.no_grad():
        for x, target, _ in loader:
            x = x.to(device)
            y_norm = target["y_norm"].to(device)
            pred_norm = model(x)
            losses.append(float(compute_baseline_training_loss(pred_norm, y_norm, non_usd_mask_t, args, threshold).item()))
            pred_norm_list.append(pred_norm.detach().cpu().numpy())
            y_raw_list.append(target["y_raw"].numpy())
    pred_raw = inverse_transform(np.concatenate(pred_norm_list, axis=0), prepared.y_mean, prepared.y_std)
    y_raw = np.concatenate(y_raw_list, axis=0)
    return float(np.mean(losses)), compute_metrics(pred_raw, y_raw, non_usd_mask_t.cpu().numpy())
