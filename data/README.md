# Data Description

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

## Preprocessing and Feature Construction

The public code in `src/data_pipeline.py` applies the following preprocessing steps.

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


### Target construction

The next-day forecasting target for each currency is stored as:

- `TargetRet_{ccy}`

If a precomputed target column such as `TARGET_{ccy}_FX_RET_FWD1` is unavailable, the code reconstructs the one-step-ahead target from `Target_{ccy}_FX` and the current `_{ccy}_FX` series.
