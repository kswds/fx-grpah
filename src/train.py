from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config import OURSMAIN_DEFAULTS
from data_pipeline import prepare_data, set_seed
from training import train_baseline_model, train_relational_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ARC_FX and baseline models for the main predictive experiment.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "main_experiment.yaml"),
        help="YAML configuration for the main experiment.",
    )
    parser.add_argument("--fx-data-path", default=None, help="Optional override for the processed FX CSV path.")
    parser.add_argument("--nonfx-data-path", default=None, help="Optional override for the processed non-FX CSV path.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-arcfx", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return cfg


def resolve_data_path(root: Path, configured: str, override: str | None) -> str:
    if override:
        return str(Path(override).resolve())
    return str((root / configured).resolve())


def as_namespace(cfg: Dict[str, Any], seed: int, model_name: str, device: str, fx_data_path: str | None, nonfx_data_path: str | None) -> SimpleNamespace:
    data_cfg = cfg["data"]
    exp_cfg = cfg["experiment"]
    train_cfg = cfg["training"]
    graph_cfg = cfg["graph"]
    baseline_cfg = cfg["baselines"]
    output_cfg = cfg["outputs"]
    ns = SimpleNamespace()
    ns.fx_data_path = resolve_data_path(ROOT, data_cfg["fx_data_path"], fx_data_path)
    ns.nonfx_data_path = resolve_data_path(ROOT, data_cfg["nonfx_data_path"], nonfx_data_path)
    ns.lookback = int(exp_cfg["lookback"])
    ns.split = list(exp_cfg["split"])
    ns.hidden = int(train_cfg.get("hidden", OURSMAIN_DEFAULTS["hidden"]))
    ns.epochs = int(train_cfg["epochs"])
    ns.batch_size = int(train_cfg["batch_size"])
    ns.lr = float(train_cfg["lr"])
    ns.weight_decay = float(train_cfg["weight_decay"])
    ns.patience = int(train_cfg["patience"])
    ns.dropout = float(train_cfg.get("dropout", OURSMAIN_DEFAULTS["dropout"]))
    ns.edge_dropout = float(graph_cfg.get("edge_dropout", OURSMAIN_DEFAULTS["edge_dropout"]))
    ns.top_k = int(graph_cfg["top_k"])
    ns.graph_rank = int(graph_cfg["graph_rank"])
    ns.spectral_bound = float(graph_cfg.get("spectral_bound", OURSMAIN_DEFAULTS["spectral_bound"]))
    ns.lambda_dir = float(train_cfg["lambda_dir"])
    ns.lambda_rank = float(train_cfg["lambda_rank"])
    ns.lambda_component = float(train_cfg.get("lambda_component", OURSMAIN_DEFAULTS["lambda_component"]))
    ns.lambda_smooth = float(train_cfg.get("lambda_smooth", OURSMAIN_DEFAULTS["lambda_smooth"]))
    ns.lambda_static = float(train_cfg.get("lambda_static", OURSMAIN_DEFAULTS["lambda_static"]))
    ns.lambda_sparse = float(train_cfg.get("lambda_sparse", OURSMAIN_DEFAULTS["lambda_sparse"]))
    ns.lambda_spectral = float(train_cfg.get("lambda_spectral", OURSMAIN_DEFAULTS["lambda_spectral"]))
    ns.small_return_quantile = float(train_cfg["small_return_quantile"])
    ns.component_gate_type = str(graph_cfg.get("component_gate_type", OURSMAIN_DEFAULTS["component_gate_type"]))
    ns.selection_metric = str(train_cfg["selection_metric"])
    ns.hit_alpha = float(train_cfg["hit_alpha"])
    ns.save_checkpoints = True
    ns.seed = int(seed)
    ns.device = device
    ns.universe = str(exp_cfg["universe_name"])
    ns.corr_architecture_order = str(baseline_cfg.get("corrlstmgat_architecture_order", "lstm_then_gat"))
    ns.fxrp_num_layers = int(baseline_cfg.get("fxrp_num_layers", 3))
    ns.relational_loss_variant = "oursmain" if model_name == "oursmain" else "baseline"
    ns.baseline_loss_variant = "oursmain"
    ns.output_root = str((ROOT / output_cfg["prediction_root"]).resolve())
    return ns


def train_arcfx(cfg: Dict[str, Any], device: str, force_retrain: bool, fx_data_path: str | None, nonfx_data_path: str | None) -> None:
    currencies = list(cfg["experiment"]["currencies"])
    prepared = prepare_data(
        resolve_data_path(ROOT, cfg["data"]["fx_data_path"], fx_data_path),
        resolve_data_path(ROOT, cfg["data"]["nonfx_data_path"], nonfx_data_path),
        currencies,
        int(cfg["experiment"]["lookback"]),
        list(cfg["experiment"]["split"]),
        include_regime_onehot=True,
    )
    arcfx_root = (ROOT / cfg["outputs"]["prediction_root"] / "arc_fx").resolve()
    for seed in cfg["experiment"]["seeds"]:
        run_dir = arcfx_root / f"oursmain_seed{seed}"
        pred_path = run_dir / "predictions" / "oursmain_predictions.parquet"
        if pred_path.exists() and not force_retrain:
            print(f"[skip ARC_FX] seed={seed}")
            continue
        args = as_namespace(cfg, int(seed), "oursmain", device, fx_data_path, nonfx_data_path)
        set_seed(int(seed))
        result = train_relational_model("oursmain", prepared, args, run_dir, device)
        print(
            f"[trained ARC_FX] seed={seed} "
            f"hit={result.raw_metrics['hit_ratio']:.4f} rmse={result.raw_metrics['rmse']:.6f}"
        )


def train_baselines(cfg: Dict[str, Any], device: str, force_retrain: bool, fx_data_path: str | None, nonfx_data_path: str | None) -> None:
    currencies = list(cfg["experiment"]["currencies"])
    prepared = prepare_data(
        resolve_data_path(ROOT, cfg["data"]["fx_data_path"], fx_data_path),
        resolve_data_path(ROOT, cfg["data"]["nonfx_data_path"], nonfx_data_path),
        currencies,
        int(cfg["experiment"]["lookback"]),
        list(cfg["experiment"]["split"]),
        include_regime_onehot=True,
    )
    baseline_root = (ROOT / cfg["outputs"]["prediction_root"] / "baselines").resolve()
    for seed in cfg["experiment"]["seeds"]:
        args = as_namespace(cfg, int(seed), "baseline", device, fx_data_path, nonfx_data_path)
        set_seed(int(seed))
        for model_name in cfg["experiment"]["baseline_models"]:
            run_dir = baseline_root / f"{model_name}_seed{seed}"
            pred_path = run_dir / "predictions" / f"{model_name}_predictions.parquet"
            if pred_path.exists() and not force_retrain:
                print(f"[skip baseline] model={model_name} seed={seed}")
                continue
            result = train_baseline_model(model_name, prepared, args, run_dir, device)
            print(
                f"[trained baseline] model={model_name} seed={seed} "
                f"hit={result.raw_metrics['hit_ratio']:.4f} rmse={result.raw_metrics['rmse']:.6f}"
            )


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    if not args.skip_arcfx:
        train_arcfx(cfg, args.device, args.force_retrain, args.fx_data_path, args.nonfx_data_path)
    if not args.skip_baselines:
        train_baselines(cfg, args.device, args.force_retrain, args.fx_data_path, args.nonfx_data_path)


if __name__ == "__main__":
    main()
