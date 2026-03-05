from dataclasses import dataclass
import torch

@dataclass
class Config:
    """
    Raw-Y FX forecasting research config.

    Target:
      - RAW FX log returns (USD-base convention)

    Inputs:
      - Local per-currency features (scaled using TRAIN-only stats):
          [FXRet, dY10, StockRet] + RV (computed within lookback)
      - Macro features (scaled using TRAIN-only stats)

    Graph:
      - FC baseline: static row-stochastic adjacency with zero diagonal.
      - Granger-proxy graph: directed adjacency estimated from TRAIN only using lag-1 correlation.
      - Shock-propagation model: propagate -> macro injection -> propagate (optional).
    """
    # data
    file_path: str = "factor_final_daily.csv"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    lookback: int = 10
    rv_window: int = 10

    # training
    batch_size: int = 128
    epochs: int = 30
    lr: float = 3e-4

    # model
    hidden: int = 64
    dropout: float = 0.1

    # regularization (applied to strength-based models)
    lambda_var: float = 0.005
    lambda_a_l1: float = 1e-4

    # Granger graph (proxy) settings
    use_granger_graph: bool = True
    granger_topk: int = 3              # keep top-k incoming per node (row-wise); 0 = keep all
    granger_min_weight: float = 0.0    # threshold applied before top-k
    granger_use_abs_corr: bool = True

    # shock propagation structure
    shockprop_steps: int = 2           # 1: propagate once, 2: propagate->macro->propagate

    # evaluation
    extreme_percentile: float = 0.90   # top 10% |y| in TRAIN as "extreme"

    # currencies / macros
    ccys = ["USD", "EUR", "JPY", "GBP", "CAD", "AUD", "CHF", "NZD", "SEK", "NOK"]
    global_features = [
        "Global_Gold", "Global_VIX", "Global_Oil", "Global_US10Y",
        "Global_Copper", "Global_SP500", "Global_US2Y"
    ]

    @property
    def n_ccy(self): return len(self.ccys)

    @property
    def usd_idx(self): return self.ccys.index("USD")

    @property
    def macro_dim(self): return len(self.global_features)

    @property
    def base_local_dim(self): return 3

    @property
    def local_dim(self): return self.base_local_dim + 1  # + RV
