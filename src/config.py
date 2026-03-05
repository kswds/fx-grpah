"""
Configuration for FX Strength GNN
Matches original fx_train.py hyperparameters exactly for reproducibility
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Paths
    file_path: str = "data/factor_final_daily_legacy.csv"  # was: factor_final_daily.csv

    # Random seed
    seed: int = 42

    # Data
    lookback: int = 20
    rv_window: int = 20  # Realized Volatility window

    # Training
    batch_size: int = 128
    epochs: int = 30
    lr: float = 3e-4

    # Model
    gnn_type: str = "gat"  # "gcn", "sage", "gat"
    hidden: int = 64
    heads: int = 4

    # Loss weights
    lambda_var: float = 0.005  # Variance regularization
    lambda_a_l1: float = 1e-4  # L1 sparsity on A matrix

    # Currencies
    ccys: List[str] = field(default_factory=lambda: [
        "USD", "EUR", "JPY", "GBP", "CAD", "AUD", "CHF", "NZD", "SEK", "NOK", "CNY"
    ])

    # Global features
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

    # Enhanced mode options (our additions)
    use_skip_connection: bool = False
    use_layer_norm: bool = False
    use_magnitude_head: bool = False

    @property
    def n_ccy(self) -> int:
        return len(self.ccys)

    @property
    def usd_idx(self) -> int:
        return self.ccys.index("USD")

    @property
    def local_dim(self) -> int:
        return 4  # FXRet, dY10, StockRet, RV

    @property
    def macro_dim(self) -> int:
        return len(self.global_features)
