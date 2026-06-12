"""
Dataset for FX Strength Research Pipeline

Key design decisions:
- Train / Val / Test split: 60 / 20 / 20
- Normalization computed on TRAIN set only, applied to all splits
- Identifiability: USD-pinning (recommended) — single mechanism, no conflict
- All macro features strictly follow config.global_features (single source of truth)
- Full data transparency: logs shape, dates, missing values
"""
import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------------------------------------------------------ FX conventions
USD_PER_UNIT     = {"EUR", "GBP", "AUD", "NZD"}      # USD per 1 unit (e.g. EUR/USD)
FOREIGN_PER_USD  = {"JPY", "CAD", "CHF", "SEK", "NOK", "CNY"}  # units per USD


def fx_to_log(series: pd.Series, ccy: str) -> np.ndarray:
    """Convert FX rate to log-price from USD-strength perspective."""
    x = series.astype(float)
    if ccy in USD_PER_UNIT:
        return np.log(x)            #  +log => stronger foreign ccy
    elif ccy in FOREIGN_PER_USD:
        return -np.log(x)           # -log => stronger foreign = lower price
    elif ccy == "USD":
        return np.zeros(len(x))     # USD is numeraire
    else:
        raise ValueError(f"Unknown currency: {ccy}")


# ------------------------------------------------------------------ Data loading

def load_data(config: Config) -> pd.DataFrame:
    """
    Load and validate raw data.

    Logs:
        - Start / end date
        - Total rows, missing values per column
        - Holiday alignment note
    """
    if os.path.exists(config.file_path):
        df = pd.read_csv(config.file_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        logger.info("=== Data Transparency Report ===")
        logger.info(f"  File        : {config.file_path}")
        logger.info(f"  Start date  : {df['Date'].min().date()}")
        logger.info(f"  End date    : {df['Date'].max().date()}")
        logger.info(f"  Total rows  : {len(df)}")

        n_missing = df.isnull().sum()
        if n_missing.any():
            logger.warning("  Missing values detected:")
            for col, cnt in n_missing[n_missing > 0].items():
                logger.warning(f"    {col}: {cnt} missing")
        else:
            logger.info("  Missing values: None")

        logger.info("  Holiday alignment: forward-filled by sort_values+reset_index")
        logger.info("================================")

    else:
        logger.warning(f"'{config.file_path}' not found — generating synthetic dummy data.")
        dates = pd.date_range(start="2010-01-01", periods=1000, freq="B")
        df = pd.DataFrame({"Date": dates})
        for c in config.ccys:
            if c == "USD":
                df[f"{c}_FX"]      = 1.0
            elif c == "JPY":
                df[f"{c}_FX"]      = np.exp(np.cumsum(np.random.randn(1000) * 0.002) + np.log(110))
            else:
                df[f"{c}_FX"]      = np.exp(np.cumsum(np.random.randn(1000) * 0.003) + np.log(1.1))
            df[f"{c}_Yield10Y"]    = np.abs(np.cumsum(np.random.randn(1000) * 0.002) + 2.0)
            df[f"{c}_Stock"]       = np.exp(np.cumsum(np.random.randn(1000) * 0.005) + np.log(100))
        for g in config.global_features:
            df[g] = np.exp(np.cumsum(np.random.randn(1000) * 0.003) + np.log(50))

    return df


# ------------------------------------------------------------------ Feature engineering

def build_features(df: pd.DataFrame, config: Config):
    """
    Build time-series feature tensors from raw DataFrame.

    Returns
    -------
    X_local_base : np.ndarray  [T, N, 3]  — FXRet, dY10, StockRet (RV added in FXDataset)
    X_macro      : np.ndarray  [T, M]      — 9 global macro features (config.global_features)
    Y            : np.ndarray  [T, N]      — next-day FX log-return

    NOTE: X[t] and Y[t] are same-row (no shift here).
          FXDataset applies: window [idx:idx+L] → target Y[idx+L]
    """
    # --- FX log prices & returns
    p_fx = pd.DataFrame({c: fx_to_log(df[f"{c}_FX"], c) for c in config.ccys})
    r_fx = p_fx.diff()

    # --- Per-currency local features
    feat = {}
    for c in config.ccys:
        feat[f"{c}_FXRet"]   = r_fx[c]
        feat[f"{c}_dY10"]    = df[f"{c}_Yield10Y"].diff()
        feat[f"{c}_StockRet"]= np.log(df[f"{c}_Stock"]).diff()

    # --- Global macro features (STRICTLY from config.global_features — single source of truth)
    for g in config.global_features:
        if g == "Global_VIX" or "Yield" in g or "US10Y" in g or "US2Y" in g:
            feat[g] = df[g].diff()          # level-based: first difference
        else:
            feat[g] = np.log(df[g]).diff()  # price-based: log return

    data = pd.DataFrame(feat).dropna().reset_index(drop=True)

    T = len(data)
    BASE_LOCAL_DIM = 3           # FXRet, dY10, StockRet  (RV appended in FXDataset)
    X_local_base = np.zeros((T, config.n_ccy, BASE_LOCAL_DIM), dtype=np.float32)
    X_macro      = data[config.global_features].values.astype(np.float32)  # shape: [T, 9]
    Y            = np.zeros((T, config.n_ccy), dtype=np.float32)

    for i, c in enumerate(config.ccys):
        X_local_base[:, i, 0] = data[f"{c}_FXRet"].values
        X_local_base[:, i, 1] = data[f"{c}_dY10"].values
        X_local_base[:, i, 2] = data[f"{c}_StockRet"].values
        Y[:, i]               = data[f"{c}_FXRet"].values

    logger.info(f"build_features: T={T}, N={config.n_ccy}, M={config.macro_dim}")
    return X_local_base, X_macro, Y


# ------------------------------------------------------------------ Normalization

def normalize_data(X_local: np.ndarray,
                   X_macro: np.ndarray,
                   Y: np.ndarray,
                   train_idx: int = None):
    """
    Normalize features using TRAIN set statistics only.
    Y (targets) is intentionally NOT normalized — hit rate is sign-relative to 0.

    Parameters
    ----------
    train_idx : if None, uses default 60% split from raw array length.

    Returns
    -------
    X_local_scaled, X_macro_scaled, Y (raw), stats dict
    """
    n_total = len(X_local)
    if train_idx is None:
        train_idx = int(n_total * 0.60)

    train_local = X_local[:train_idx]
    train_macro = X_macro[:train_idx]

    local_mean = train_local.mean(axis=(0, 1), keepdims=True)  # [1,1,D]
    local_std  = train_local.std(axis=(0, 1),  keepdims=True) + 1e-6
    macro_mean = train_macro.mean(axis=0, keepdims=True)        # [1,M]
    macro_std  = train_macro.std(axis=0,  keepdims=True) + 1e-6

    X_local_scaled = (X_local - local_mean) / local_std
    X_macro_scaled = (X_macro - macro_mean) / macro_std

    stats = {
        "local_mean": local_mean, "local_std": local_std,
        "macro_mean": macro_mean, "macro_std": macro_std,
        "train_idx": train_idx,
    }
    return X_local_scaled, X_macro_scaled, Y, stats


# ------------------------------------------------------------------ Realized Volatility

def realized_vol_within_window(fxret_seq: torch.Tensor, window: int = 20) -> torch.Tensor:
    """Compute rolling realized volatility within lookback window."""
    L, N = fxret_seq.shape
    rv = torch.zeros((L, N), dtype=fxret_seq.dtype)
    for t in range(L):
        s = max(0, t - window + 1)
        seg = fxret_seq[s:t + 1]
        if seg.size(0) >= 2:
            rv[t] = seg.std(dim=0, unbiased=False)
    return rv


# ------------------------------------------------------------------ Dataset class

class FXDataset(Dataset):
    """
    FX Dataset with 4 local features (FXRet, dY10, StockRet, RV).

    Time-offset convention:
        __getitem__(idx) → (X[idx:idx+L], Y[idx+L])
        - Feature window : days [idx, idx+L-1]
        - Target         : day idx+L  (no leakage)
    """

    def __init__(
        self,
        X_local_base: np.ndarray,
        X_macro: np.ndarray,
        Y: np.ndarray,
        config: Config,
        macro_mode: str = "real",
    ):
        self.X_local_base = X_local_base
        self.X_macro      = X_macro
        self.Y            = Y
        self.L            = config.lookback
        self.rv_window    = min(config.rv_window, config.lookback)
        self.macro_mode   = macro_mode

    def __len__(self) -> int:
        return len(self.X_local_base) - self.L

    def __getitem__(self, idx: int):
        xl_base = torch.tensor(self.X_local_base[idx: idx + self.L], dtype=torch.float32)  # [L, N, 3]
        xm      = torch.tensor(self.X_macro[idx: idx + self.L],      dtype=torch.float32)  # [L, M]

        if self.macro_mode == "zero":
            xm = torch.zeros_like(xm)

        y = torch.tensor(self.Y[idx + self.L], dtype=torch.float32)   # [N]

        # Append realized volatility as 4th local feature
        fxret_seq = xl_base[:, :, 0]   # [L, N]
        rv_seq    = realized_vol_within_window(fxret_seq, window=self.rv_window)  # [L, N]
        xl        = torch.cat([xl_base, rv_seq.unsqueeze(-1)], dim=2)  # [L, N, 4]

        return xl, xm, y


# ------------------------------------------------------------------ DataLoader factory

def create_dataloaders(config: Config, macro_mode: str = "real"):
    """
    Create train / val / test DataLoaders with 60/20/20 time-series split.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    df = load_data(config)
    X_local_base, X_macro, Y = build_features(df, config)

    n_total   = len(X_local_base)
    train_raw = int(n_total * config.train_ratio)

    # Normalize using train-only statistics
    X_local_scaled, X_macro_scaled, Y, stats = normalize_data(
        X_local_base, X_macro, Y, train_idx=train_raw
    )

    # Dataset (length = n_total - L)
    dataset = FXDataset(X_local_scaled, X_macro_scaled, Y, config, macro_mode)
    n       = len(dataset)

    # Map raw-row split indices to dataset indices
    # dataset[idx] uses rows [idx, idx+L], so:
    #   raw train end → dataset train end = train_raw - L
    train_end, val_end = config.get_split_indices(n)

    train_ds = torch.utils.data.Subset(dataset, list(range(0,          train_end)))
    val_ds   = torch.utils.data.Subset(dataset, list(range(train_end,  val_end)))
    test_ds  = torch.utils.data.Subset(dataset, list(range(val_end,    n)))

    logger.info(
        f"Dataset split | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}"
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ------------------------------------------------------------------ Graph utility

def fully_connected_edge_index(n: int) -> torch.Tensor:
    """Create fully connected graph edge index (excluding self-loops)."""
    edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    return torch.tensor(edges, dtype=torch.long).T
