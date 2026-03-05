import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import Config

USD_PER_UNIT = {"EUR", "GBP", "AUD", "NZD"}             # USD per 1 unit of ccy
FOREIGN_PER_USD = {"JPY", "CAD", "CHF", "SEK", "NOK"}   # ccy per 1 USD

def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Safe log to avoid "invalid value encountered in log".
    Values <= 0 become NaN and are dropped later by dropna().
    """
    x = x.astype(float)
    x = np.where(x > 0, x, np.nan)
    return np.log(x + eps)

def fx_to_log(series: pd.Series, ccy: str) -> np.ndarray:
    """
    Convert spot FX level to log-price in a consistent USD-base convention.

    If series is "USD per 1 unit of ccy" (EUR, GBP, ...), log-price is log(S).
    If series is "ccy per 1 USD" (JPY, CAD, ...), use -log(S) so that an increase
    still corresponds to USD appreciation (consistent direction).
    USD node is identically zero.
    """
    x = series.astype(float).values
    if ccy in USD_PER_UNIT:
        return safe_log(x)
    if ccy in FOREIGN_PER_USD:
        return -safe_log(x)
    if ccy == "USD":
        return np.zeros(len(x), dtype=float)
    raise ValueError(ccy)

def process_data(config: Config) -> pd.DataFrame:
    """
    Load factor_final_daily.csv and produce a clean feature DataFrame with diffs/returns.

    Columns:
      - {CCY}_FXRet      : RAW FX log return (USD-base)
      - {CCY}_dY10       : 10Y yield change (diff)
      - {CCY}_StockRet   : equity proxy log return
      - Global_*         : diff (for yields/VIX) or log-diff (for prices)
    """
    if not os.path.exists(config.file_path):
        raise FileNotFoundError(f"File not found: {config.file_path}")

    df = pd.read_csv(config.file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # FX log returns (USD-base)
    p_fx = {c: fx_to_log(df[f"{c}_FX"], c) for c in config.ccys}
    p_fx = pd.DataFrame(p_fx)
    r_fx = p_fx.diff()

    feat = {}
    for c in config.ccys:
        feat[f"{c}_FXRet"] = r_fx[c].values
        feat[f"{c}_dY10"] = df[f"{c}_Yield10Y"].astype(float).diff().values
        stock_log = safe_log(df[f"{c}_Stock"].astype(float).values)
        feat[f"{c}_StockRet"] = pd.Series(stock_log).diff().values

    for g in config.global_features:
        x = df[g].astype(float).values
        if (g == "Global_VIX") or ("Yield" in g) or ("US10Y" in g) or ("US2Y" in g):
            feat[g] = pd.Series(x).diff().values
        else:
            feat[g] = pd.Series(safe_log(x)).diff().values

    data = pd.DataFrame(feat).dropna().reset_index(drop=True)
    return data

def build_tensors_raw(d: pd.DataFrame, config: Config):
    """
    Build RAW tensors (NO cross-sectional norm, NO Y norm).

    Returns:
      X_local_raw: [T, N, 3]  (FXRet, dY10, StockRet)
      X_macro_raw: [T, M]
      Y_raw      : [T, N]     (RAW FXRet target)
    """
    T = len(d)
    X_local_raw = np.zeros((T, config.n_ccy, config.base_local_dim), dtype=np.float32)
    X_macro_raw = d[config.global_features].values.astype(np.float32)
    Y_raw = np.zeros((T, config.n_ccy), dtype=np.float32)

    for t in range(T):
        fx_ret = np.array([d.iloc[t][f"{c}_FXRet"] for c in config.ccys], dtype=np.float32)
        dy10   = np.array([d.iloc[t][f"{c}_dY10"] for c in config.ccys], dtype=np.float32)
        stk    = np.array([d.iloc[t][f"{c}_StockRet"] for c in config.ccys], dtype=np.float32)

        X_local_raw[t, :, 0] = fx_ret
        X_local_raw[t, :, 1] = dy10
        X_local_raw[t, :, 2] = stk
        Y_raw[t, :] = fx_ret

    return X_local_raw, X_macro_raw, Y_raw

def realized_vol_within_window(fxret_seq: torch.Tensor, window: int) -> torch.Tensor:
    """
    fxret_seq: [L, N] (scaled feature)
    returns:   [L, N] realized vol computed using only within-lookback samples
    """
    L, N = fxret_seq.shape
    rv = torch.zeros((L, N), dtype=fxret_seq.dtype)
    for t in range(L):
        s = max(0, t - window + 1)
        seg = fxret_seq[s:t + 1]
        rv[t] = seg.std(dim=0, unbiased=False) if seg.size(0) >= 2 else 0.0
    return rv

class FXDataset(Dataset):
    """
    (xl, xm, y, t_idx)
      - xl: [L, N, 4] = [FXRet, dY10, StockRet, RV]
      - xm: [L, M]
      - y : [N] RAW FXRet at time t+L (NO normalization)
      - t_idx: label time index (used for train-threshold extremes)
    """
    def __init__(self, X_local_scaled, X_macro_scaled, Y_raw, config: Config, macro_mode: str = "real"):
        self.X_local = X_local_scaled
        self.X_macro = X_macro_scaled
        self.Y = Y_raw
        self.L = config.lookback
        self.rv_window = min(config.rv_window, config.lookback)
        assert macro_mode in ("real", "zero")
        self.macro_mode = macro_mode

    def __len__(self):
        return len(self.X_local) - self.L

    def __getitem__(self, idx):
        xl_base = torch.tensor(self.X_local[idx:idx + self.L], dtype=torch.float32)  # [L,N,3]
        xm = torch.tensor(self.X_macro[idx:idx + self.L], dtype=torch.float32)       # [L,M]
        if self.macro_mode == "zero":
            xm = torch.zeros_like(xm)

        y = torch.tensor(self.Y[idx + self.L], dtype=torch.float32)                  # [N]
        t_idx = torch.tensor(idx + self.L, dtype=torch.long)

        fxret_seq = xl_base[:, :, 0]  # scaled FXRet feature
        rv_seq = realized_vol_within_window(fxret_seq, window=self.rv_window)
        xl = torch.cat([xl_base, rv_seq.unsqueeze(-1)], dim=2)                       # [L,N,4]
        return xl, xm, y, t_idx
