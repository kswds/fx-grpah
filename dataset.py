"""
Dataset for FX Strength GNN
Matches original fx_train.py data processing exactly
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config


# FX quote conventions
USD_PER_UNIT = {"EUR", "GBP", "AUD", "NZD"}  # USD per 1 unit of currency
FOREIGN_PER_USD = {"JPY", "CAD", "CHF", "SEK", "NOK"}  # currency per 1 USD


def fx_to_log(series: pd.Series, ccy: str) -> np.ndarray:
    """Convert FX rate to log price (USD strength perspective)"""
    x = series.astype(float)
    if ccy in USD_PER_UNIT:
        return np.log(x)
    elif ccy in FOREIGN_PER_USD:
        return -np.log(x)
    elif ccy == "USD":
        return np.zeros(len(x))
    else:
        raise ValueError(f"Unknown currency: {ccy}")


def load_data(config: Config) -> pd.DataFrame:
    """Load and validate data"""
    if os.path.exists(config.file_path):
        df = pd.read_csv(config.file_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    else:
        print(f"Warning: '{config.file_path}' not found. Using dummy data.")
        dates = pd.date_range(start="2020-01-01", periods=500)
        df = pd.DataFrame({"Date": dates})
        for c in config.ccys:
            df[f"{c}_FX"] = np.random.uniform(100, 150, 500) if c == "JPY" else np.random.uniform(0.5, 1.5, 500)
            df[f"{c}_Yield10Y"] = np.random.uniform(1, 5, 500)
            df[f"{c}_Stock"] = np.cumsum(np.random.randn(500)) + 100
        for g in config.global_features:
            df[g] = np.cumsum(np.random.randn(500)) + 50
    return df


def build_features(df: pd.DataFrame, config: Config):
    """
    Build feature tensors from raw data
    Returns: X_local_base [T, N, 3], X_macro [T, M], Y [T, N]
    """
    # FX log prices and returns
    p_fx = {c: fx_to_log(df[f"{c}_FX"], c) for c in config.ccys}
    p_fx = pd.DataFrame(p_fx)
    r_fx = p_fx.diff()

    # Build feature dict
    feat = {}
    for c in config.ccys:
        feat[f"{c}_StockRet"] = np.log(df[f"{c}_Stock"]).diff()
        feat[f"{c}_dY10"] = df[f"{c}_Yield10Y"].diff()
        feat[f"{c}_FXRet"] = r_fx[c]

    for g in config.global_features:
        if g == "Global_VIX" or "Yield" in g or "US10Y" in g or "US2Y" in g:
            feat[g] = df[g].diff()
        else:
            feat[g] = np.log(df[g]).diff()

    data = pd.DataFrame(feat).dropna().reset_index(drop=True)

    # Build tensors
    T = len(data)
    BASE_LOCAL_DIM = 3
    X_local_base = np.zeros((T, config.n_ccy, BASE_LOCAL_DIM), dtype=np.float32)
    X_macro = data[config.global_features].values.astype(np.float32)
    Y = np.zeros((T, config.n_ccy), dtype=np.float32)

    for i, c in enumerate(config.ccys):
        X_local_base[:, i, 0] = data[f"{c}_FXRet"].values
        X_local_base[:, i, 1] = data[f"{c}_dY10"].values
        X_local_base[:, i, 2] = data[f"{c}_StockRet"].values
        Y[:, i] = data[f"{c}_FXRet"].values

    # NOTE: Normalization moved to create_dataloaders() for proper train-only normalization
    # Do NOT normalize here to avoid double normalization
    return X_local_base, X_macro, Y


def realized_vol_within_window(fxret_seq: torch.Tensor, window: int = 20) -> torch.Tensor:
    """Compute realized volatility within lookback window"""
    L, N = fxret_seq.shape
    rv = torch.zeros((L, N), dtype=fxret_seq.dtype)
    for t in range(L):
        s = max(0, t - window + 1)
        seg = fxret_seq[s:t + 1]
        if seg.size(0) >= 2:
            rv[t] = seg.std(dim=0, unbiased=False)
        else:
            rv[t] = 0.0
    return rv


class FXDataset(Dataset):
    """FX Dataset with realized volatility feature"""

    def __init__(
        self,
        X_local_base: np.ndarray,
        X_macro: np.ndarray,
        Y: np.ndarray,
        config: Config,
        macro_mode: str = "real"
    ):
        self.X_local_base = X_local_base
        self.X_macro = X_macro
        self.Y = Y
        self.L = config.lookback
        self.rv_window = min(config.rv_window, config.lookback)
        self.macro_mode = macro_mode

    def __len__(self) -> int:
        return len(self.X_local_base) - self.L

    def __getitem__(self, idx: int):
        xl_base = torch.tensor(self.X_local_base[idx:idx + self.L], dtype=torch.float32)
        xm = torch.tensor(self.X_macro[idx:idx + self.L], dtype=torch.float32)

        if self.macro_mode == "zero":
            xm = torch.zeros_like(xm)

        y = torch.tensor(self.Y[idx + self.L], dtype=torch.float32)

        # Add realized volatility as 4th feature
        fxret_seq = xl_base[:, :, 0]
        rv_seq = realized_vol_within_window(fxret_seq, window=self.rv_window)
        xl = torch.cat([xl_base, rv_seq.unsqueeze(-1)], dim=2)

        return xl, xm, y


def create_dataloaders(config: Config, macro_mode: str = "real"):
    """Create train and test dataloaders"""
    df = load_data(config)
    X_local_base, X_macro, Y = build_features(df, config)

    # Train set normalization (train-only statistics for proper out-of-sample eval)
    n_total = len(X_local_base)
    split_idx = int(n_total * 0.8)

    train_local = X_local_base[:split_idx]
    train_macro = X_macro[:split_idx]
    train_Y = Y[:split_idx]

    local_mean = train_local.mean(axis=(0, 1), keepdims=True)
    local_std = train_local.std(axis=(0, 1), keepdims=True) + 1e-6
    macro_mean = train_macro.mean(axis=0, keepdims=True)
    macro_std = train_macro.std(axis=0, keepdims=True) + 1e-6
    y_mean = train_Y.mean(axis=0, keepdims=True)
    y_std = train_Y.std(axis=0, keepdims=True) + 1e-6

    X_local_scaled = (X_local_base - local_mean) / local_std
    X_macro_scaled = (X_macro - macro_mean) / macro_std
    Y_scaled = (Y - y_mean) / y_std

    # Create dataset
    dataset = FXDataset(X_local_scaled, X_macro_scaled, Y_scaled, config, macro_mode)

    # Split
    n = len(dataset)
    split = int(n * 0.8)
    train_ds = torch.utils.data.Subset(dataset, list(range(0, split)))
    test_ds = torch.utils.data.Subset(dataset, list(range(split, n)))

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader


def normalize_data(X_local: np.ndarray, X_macro: np.ndarray, Y: np.ndarray,
                   train_idx: int = None):
    """
    Normalize data using train set statistics.

    Args:
        X_local: [T, N, D] local features
        X_macro: [T, M] macro features
        Y: [T, N] targets
        train_idx: index to split train/test. If None, uses 80% split.

    Returns:
        X_local_scaled, X_macro_scaled, Y_scaled, stats dict
    """
    n_total = len(X_local)
    if train_idx is None:
        train_idx = int(n_total * 0.8)

    # Compute train statistics
    train_local = X_local[:train_idx]
    train_macro = X_macro[:train_idx]
    train_Y = Y[:train_idx]

    local_mean = train_local.mean(axis=(0, 1), keepdims=True)
    local_std = train_local.std(axis=(0, 1), keepdims=True) + 1e-6
    macro_mean = train_macro.mean(axis=0, keepdims=True)
    macro_std = train_macro.std(axis=0, keepdims=True) + 1e-6
    y_mean = train_Y.mean(axis=0, keepdims=True)
    y_std = train_Y.std(axis=0, keepdims=True) + 1e-6

    # Apply normalization
    X_local_scaled = (X_local - local_mean) / local_std
    X_macro_scaled = (X_macro - macro_mean) / macro_std
    Y_scaled = (Y - y_mean) / y_std

    stats = {
        'local_mean': local_mean, 'local_std': local_std,
        'macro_mean': macro_mean, 'macro_std': macro_std,
        'y_mean': y_mean, 'y_std': y_std
    }

    return X_local_scaled, X_macro_scaled, Y_scaled, stats


def fully_connected_edge_index(n: int) -> torch.Tensor:
    """Create fully connected graph edge index"""
    edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    return torch.tensor(edges, dtype=torch.long).T
