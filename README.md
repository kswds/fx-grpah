# FX Strength Model: Ridge Regression on Currency Strength

Predicting FX returns via latent currency strengths. Ridge regression on per-currency local features (FX return, yield change, stock return) and global macro factors, then pairwise trading signals from strength differences.

## Project Structure

```
fx-graph-other/
├── src/                        # Core source code
│   ├── config.py              # Configuration (11 currencies, 9 macro features, hyperparams)
│   ├── dataset.py             # Data loading, feature engineering, normalization
│   ├── models.py              # GNN model (FXStrengthGNN, for reference)
│   ├── train.py               # Training utilities
│   └── utils.py               # Helper functions
├── scripts/
│   └── exp_full_comparison.py # Ridge vs MLP vs GNN comparison (Table 1)
├── evaluation/
│   ├── eval_trading.py        # Trading backtest with signal-strength filtering
│   ├── eval_network.py        # Macro-currency network visualization
│   └── eval_rolling.py        # Rolling window backtest (robustness check)
├── data/
│   └── factor_final_daily_legacy.csv  # Daily FX, yield, stock, macro data (2005-2025)
└── results/                    # Output JSON + PNG files
```

## Method

1. **Features**: Per-currency (FX return, 10Y yield change, stock return) + global macro (VIX, Gold, Oil, Copper, US2Y, IronOre)
2. **Model**: Ridge regression predicts 11 currency strengths from last-day features (lookback=20, uses last day only)
3. **Trading signal**: `pred_ij = strength_i - strength_j` for all 55 (or 45 excluding USD) currency pairs
4. **Signal-strength filter**: Select top X% pairs by `|pred_ij|`, trade `sign(pred_ij)` direction
5. **Backtest**: Daily compounding, equal-weight allocation across selected pairs

## Usage

```bash
# Ridge vs MLP vs GNN comparison table
python3 scripts/exp_full_comparison.py

# Trading backtest (generates PNGs + JSON)
python3 evaluation/eval_trading.py

# Macro-currency network visualization
python3 evaluation/eval_network.py

# Rolling window robustness check (18 folds, train 4yr / test 1yr)
python3 evaluation/eval_rolling.py
```

## Results

### Model Comparison (Hit Rate on 55 pairs, 3 seeds)

| Model | Lookback=last | Lookback=2 | Lookback=5 | Lookback=full |
|-------|:---:|:---:|:---:|:---:|
| Ridge | 54.2% | 53.8% | 53.5% | 52.7% |
| MLP | 52.1% | 51.9% | 51.7% | 51.2% |
| GNN | 52.3% | — | — | — |

### Trading Backtest (80/20 split, 0 bps, 55 pairs)

| Strategy | Trades | Hit Rate | Total | Annual | Sharpe |
|----------|--------|----------|-------|--------|--------|
| All (55) | 55 | 61.2% | 568% | 54.2% | 11.58 |
| Top 50% | 27 | 67.7% | 1914% | 98.5% | 11.15 |
| Top 30% | 16 | 70.3% | 3603% | 128.1% | 10.23 |
| Top 20% | 11 | 71.4% | 5130% | 146.8% | 9.15 |
| Top 10% | 5 | 73.1% | 8784% | 178.5% | 9.01 |

### Rolling Window (18 folds, 0 bps, 55 pairs)

| Strategy | Hit Rate | Annual (mean) | Sharpe (mean) |
|----------|----------|---------------|---------------|
| All (55) | 60.6% | 54.3% ± 23.1% | 10.75 ± 4.85 |
| Top 30% | 68.4% | 127.0% ± 59.4% | 9.97 ± 4.88 |
| Top 10% | 71.1% | 183.4% ± 93.4% | 9.17 ± 4.48 |

## Data

- 11 currencies: USD, EUR, JPY, GBP, CAD, AUD, CHF, NZD, SEK, NOK, CNY
- 9 macro features: VIX, Gold, Oil, Copper, US2Y, IronOre, US10Y, SP500, DXY
- Period: 2005-01 ~ 2025-12 (~5500 trading days)

## Requirements

- Python 3.8+
- NumPy, Pandas, Scikit-learn, Matplotlib
- PyTorch (for MLP/GNN comparison only)
