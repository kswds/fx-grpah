from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"

DEFAULT_FX_DATA = DATA_DIR / "factor_daily_alligned_krw.csv"
DEFAULT_NONFX_DATA = DATA_DIR / "score_vA_nonfx_features.csv"

UNIVERSE_PRESETS = {
    "major3": ["USD", "EUR", "JPY", "GBP"],
    "core6": ["USD", "EUR", "JPY", "GBP", "CAD", "AUD", "CHF"],
    "krw7": ["USD", "EUR", "JPY", "GBP", "CAD", "AUD", "KRW", "CHF"],
}

OURSMAIN_DEFAULTS = {
    "hidden": 48,
    "top_k": 2,
    "graph_rank": 8,
    "dropout": 0.25,
    "edge_dropout": 0.05,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "lambda_dir": 0.10,
    "lambda_rank": 0.08,
    "lambda_component": 1e-4,
    "lambda_smooth": 1e-3,
    "lambda_static": 1e-3,
    "lambda_sparse": 1e-4,
    "lambda_spectral": 1e-3,
    "small_return_quantile": 0.4,
    "component_gate_type": "sigmoid",
    "spectral_bound": 1.0,
    "lookback": 10,
}

