"""
Configuration for FX Strength Research Pipeline
Single source of truth for all features, splits, and hyperparameters.

Train: 60% | Val: 20% | Test: 20%
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # ------------------------------------------------------------------ Paths
    file_path: str = str(BASE_DIR / "data" / "factor_daily_legacy.csv")

    # ------------------------------------------------------------ Random seed
    seed: int = 42

    # ------------------------------------------------------------------ Data
    lookback: int = 20
    rv_window: int = 20      # Realized Volatility window

    # ------------------------------------- Train / Val / Test split (60/20/20)
    train_ratio: float = 0.60
    val_ratio:   float = 0.20
    # test_ratio is implicitly 1 - train_ratio - val_ratio = 0.20

    # --------------------------------------------------------------- Training
    batch_size: int = 128
    epochs: int = 80
    lr: float = 3e-4
    early_stopping_patience: int = 10   # Stop after N epochs of no val improvement

    # ----------------------------------------------------------------- Model
    gnn_type: str = "gat"        # "gcn", "sage", "gat"
    hidden: int = 64
    heads: int = 4

    # Hybrid model config
    hybrid_hidden: int = 32   # Tuned: Group1 best (was 64)
    dropout: float = 0.1
    top_k: int = 8               # Tuned: Group1 best (was 6)

    # ----------------------------------------------------------- Loss weights
    # VICReg-style variance constraint: loss_var = relu(γ - std(ds))
    lambda_var: float = 0.01
    vicreg_gamma: float = 1.0   # target std for VICReg variance term
    # BCE direction loss weight
    lambda_dir: float = 0.3
    # Pairwise ranking loss weight
    lambda_rank: float = 0.1
    # L1 sparsity on A matrix
    lambda_a_l1: float = 1e-4

    # ------------------------------------------------------------- Currencies
    ccys: List[str] = field(default_factory=lambda: [
        "USD", "EUR", "JPY", "GBP", "CAD", "AUD", "CHF", "NZD", "SEK", "NOK", "CNY"
    ])

    # --------------------------------------------------- Global macro features
    # SINGLE source of truth — ALL pipelines MUST use config.global_features
    global_features: List[str] = field(default_factory=lambda: [
        "Global_Gold",
        "Global_VIX",
        "Global_Oil",
        "Global_US10Y",
        "Global_Copper",
        "Global_SP500",
        "Global_US2Y",
        "Global_Shanghai",
        "Global_IronOre",
    ])

    # ------------------------------------------------------- Identifiability
    # Choose ONE: "usd_pin" (recommended) or "zero_mean"
    identifiability: str = "usd_pin"

    # --------------------------------------------------- Enhanced mode flags
    use_skip_connection: bool = False
    use_layer_norm: bool = False
    use_magnitude_head: bool = False

    # -------------------------------------------------------- Evaluation
    top_k_portfolio: int = 3    # Long top-k, short bottom-k for Sharpe

    # ---------------------------------------------------------------- Derived
    @property
    def n_ccy(self) -> int:
        return len(self.ccys)

    @property
    def usd_idx(self) -> int:
        return self.ccys.index("USD")

    @property
    def local_dim(self) -> int:
        return 4  # FXRet, dY10, StockRet, RV (RV added in FXDataset)

    @property
    def macro_dim(self) -> int:
        return len(self.global_features)

    # Val split index (applied to dataset sample count, not raw T)
    def get_split_indices(self, n_samples: int):
        """Return (train_end, val_end) indices for n_samples."""
        train_end = int(n_samples * self.train_ratio)
        val_end   = int(n_samples * (self.train_ratio + self.val_ratio))
        return train_end, val_end
