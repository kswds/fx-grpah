# Data Description

This repository does not include the raw downloaded data. Instead, it expects two processed CSV files that already contain the aligned daily forecasting panel used by the released experiments:

- `data/processed/factor_daily_alligned_krw.csv`
- `data/processed/score_vA_nonfx_features.csv`

The model is trained on a multi-currency USD-relative forecasting universe:

- `USD, EUR, JPY, GBP, CAD, AUD, KRW, CHF, NZD, SEK, NOK`

## Data Groups

The processed panel combines three types of information.

### 1. FX Data

The FX panel provides spot exchange-rate series and next-day target returns for the 10 non-USD currencies relative to USD.

Examples of FX-related columns used by the pipeline:

- `{ccy}_FX`
- `Target_{ccy}_FX`
- `TARGET_{ccy}_FX_RET_FWD1`

From these raw FX series, the pipeline constructs local-currency FX features such as:

- 1-day log return
- 5-day and 20-day momentum
- 5-day and 20-day rolling volatility
- 20-day rolling z-score

These are implemented in `src/data_pipeline.py` through:

- `{ccy}_FX_ret_1d`
- `{ccy}_FX_mom_5`
- `{ccy}_FX_mom_20`
- `{ccy}_FX_vol_5`
- `{ccy}_FX_vol_20`
- `{ccy}_FX_zscore_20`

### 2. Country-Specific Financial and Macro Data

For each currency, the processed panel includes country-level financial and macroeconomic variables. The code expects or constructs the following blocks:

Rate block:

- `{ccy}_Yield10Y`
- `{ccy}_Yield10Y_change`
- `{ccy}_Yield10Y_minus_US10Y`
- `{ccy}_RealRate10Y`
- `{ccy}_Yield10Y_Available`

Equity block:

- `{ccy}_Stock_ret`
- `{ccy}_Stock_mom_5`
- `{ccy}_Stock_mom_20`
- `{ccy}_Stock_vol_20`
- `{ccy}_Stock_ret_minus_SP500_ret`

Country macro block:

- `{ccy}_GDP_growth`
- `{ccy}_CPIInfl`
- `{ccy}_PPIInfl`
- `{ccy}_GDP_growth_minus_USD`
- `{ccy}_CPIInfl_minus_USD`
- `{ccy}_PPIInfl_minus_USD`

### 3. Global Market and Macro Data

The processed panel also includes global market variables that are broadcast to all currencies:

- `Global_VIX`
- `Global_VIX_change`
- `Global_BroadDollar_ret`
- `Global_DXY_ret`
- `Global_SP500_ret`
- `Global_Oil_ret`
- `Global_Gold_ret`
- `Global_Copper_ret`
- `Global_US1Y`
- `Global_US2Y`
- `Global_US5Y`
- `Global_US10Y`
- `Global_US2Y_change`
- `Global_US10Y_change`
- `Global_US10Y_minus_US2Y`

## Upstream Sources

The processed files were built from standard public market and macroeconomic data sources used in the original project pipeline.

Primary source families:

- **Yahoo Finance**: FX spot series, equity-index series, commodity prices, DXY / broad-dollar style market series, VIX, and other market-price-based global factors
- **FRED**: U.S. Treasury yields and related macro-financial benchmark series
- **Country-level macro releases / compiled macro panels**: GDP growth, CPI inflation, PPI inflation, local 10Y yields, and related country-specific macro variables included in the processed panel

This anonymous release intentionally omits:

- raw vendor downloads
- source-specific scraping or collection scripts
- credentials and proprietary access logic

## Preprocessing and Feature Construction

The public code in `src/data_pipeline.py` applies the following preprocessing steps.

### Date alignment

- All dates are normalized to daily timestamps
- The FX and non-FX processed files are merged on `Date`
- When overlapping columns exist, the pipeline prefers the right-hand non-FX panel during merge

### FX return convention

The code converts exchange rates into a unified local-currency appreciation convention:

- for `EUR`, `GBP`, `AUD`, and `NZD`, it uses `log(FX)`
- for `JPY`, `CAD`, `CHF`, `SEK`, `NOK`, and `KRW`, it uses `-log(FX)`

This ensures that positive transformed returns correspond to local-currency strength in a common direction across currencies.

### Derived market features

If not already present in the processed files, the pipeline derives:

- log returns for equity and commodity price series
- VIX daily change
- U.S. yield changes and slope variables
- local 10Y yield changes and local-minus-U.S. spreads
- relative equity-return features versus the S&P 500
- macro differentials versus the U.S.

### Missing-feature fallback behavior

For some macro variables, the pipeline fills unavailable series conservatively:

- missing `{ccy}_GDP_growth` defaults to `0.0`
- missing `{ccy}_PPIInfl` defaults to `0.0`
- missing macro-differential columns are created when possible from country and U.S. values, otherwise default to `0.0`

### Target construction

The next-day forecasting target for each currency is stored as:

- `TargetRet_{ccy}`

If a precomputed target column such as `TARGET_{ccy}_FX_RET_FWD1` is unavailable, the code reconstructs the one-step-ahead target from `Target_{ccy}_FX` and the current `_{ccy}_FX` series.

## Files Expected By The Public Scripts

Training:

```bash
python src/train.py --config configs/main_experiment.yaml --fx-data-path data/processed/factor_daily_alligned_krw.csv --nonfx-data-path data/processed/score_vA_nonfx_features.csv
```

Evaluation:

```bash
python src/evaluate.py --config configs/main_experiment.yaml --fx-data-path data/processed/factor_daily_alligned_krw.csv --nonfx-data-path data/processed/score_vA_nonfx_features.csv
```

In short, this repository releases the full modeling and evaluation code, while the user must supply the two processed CSV panels that contain the aligned FX, country-specific, and global macro-financial data described above.
