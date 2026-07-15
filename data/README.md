# Data README

## Data files actually used by `oursmain`

`oursmain` is trained using the following two processed CSV files together:

1. `data/processed/factor_daily_alligned_krw.csv`
2. `data/processed/score_vA_nonfx_features.csv`

In the shared pipeline:

- `factor_daily_alligned_krw.csv` provides the FX and target side
- `score_vA_nonfx_features.csv` provides the non-FX feature side
- the two files are inner-joined on `Date`

In practice, the first file serves as the aligned FX/target base dataset, while the second file provides most of the explanatory non-FX inputs.

## 1. `factor_daily_alligned_krw.csv`

### Role

This file contains:

- FX levels for each currency
- some legacy-aligned stock and 10Y yield series
- forward target variables
- the base time series aligned to the legacy research pipeline

### Main columns

- Currency-level FX / market columns
  - `USD_FX`, `EUR_FX`, `JPY_FX`, `GBP_FX`, `CAD_FX`, `AUD_FX`, `KRW_FX`, `CHF_FX`, `NZD_FX`, `SEK_FX`, `NOK_FX`
  - `USD_Stock`, `EUR_Stock`, ..., `NOK_Stock`
  - `USD_Yield10Y`, `EUR_Yield10Y`, ..., `NOK_Yield10Y`
- Global columns
  - `Global_Gold`, `Global_VIX`, `Global_Oil`, `Global_US10Y`, `Global_Copper`, `Global_SP500`, `Global_US2Y`
- Target columns
  - `Target_USD_FX`, `Target_EUR_FX`, ..., `Target_NOK_FX`
  - `TARGET_EUR_FX_RET_FWD1`, ..., `TARGET_NOK_FX_RET_FWD1`

### Source / construction

This is not a freshly assembled raw-source file. It is a legacy aligned dataset retained from the original research pipeline.

According to `data/processed/feature_source_map.csv`:

- most FX / stock / 10Y yield / global factor series are marked as `legacy`
- `Target_*` and `TARGET_*_FX_RET_FWD1` are marked as `derived`

So, in the current shared pipeline, this file should be treated as the final aligned FX/target base dataset.

### Notes

- some currencies include `shift_back_1d` alignment adjustments
- target variables were recomputed after this alignment step

## 2. `score_vA_nonfx_features.csv`

### Role

This file is the main non-FX feature source for `oursmain`.

The model builds the following input groups from this file:

- `local`
  - FX return / momentum / volatility / z-score
- `rate`
  - 10Y yield, yield change, spread vs US 10Y, availability flag
- `equity`
  - local stock return / momentum / volatility / relative return vs S&P 500
- `countrymacro`
  - GDP growth, CPI inflation, PPI inflation, and their differences vs the US
- `global`
  - VIX, broad dollar, DXY, S&P 500, oil, gold, copper, and US curve/slope changes
- `regime one-hot`
  - `Regime_LowVol`, `Regime_MidVol`, `Regime_HighVol`, `Regime_RiskOff`, `Regime_DollarShock`, `Regime_CommodityShock`
  - these regime indicators are generated in the experiment code from existing columns

### Main data included in the file

According to `score_vA_collection_report.txt`, the global features include:

- `Global_VIX`
- `Global_BroadDollar`
- `Global_US1Y`
- `Global_US2Y`
- `Global_US5Y`
- `Global_US10Y`
- `Global_SP500`
- `Global_Oil`
- `Global_Gold`
- `Global_Copper`
- `Global_DXY`
- `Global_US10Y_minus_US2Y`
- `Global_US10Y_change`
- `Global_US2Y_change`
- `Global_VIX_change`
- `Global_SP500_ret`
- `Global_Oil_ret`
- `Global_Gold_ret`
- `Global_Copper_ret`
- `Global_BroadDollar_ret`
- `Global_DXY_ret`

For each currency, the file typically contains around the following 17 columns:

- `{CCY}_Stock`
- `{CCY}_Stock_ret`
- `{CCY}_Stock_mom_5`
- `{CCY}_Stock_mom_20`
- `{CCY}_Stock_vol_20`
- `{CCY}_Stock_ret_minus_SP500_ret`
- `{CCY}_Yield10Y`
- `{CCY}_Yield10Y_change`
- `{CCY}_Yield10Y_minus_US10Y`
- `{CCY}_RealRate10Y`
- `{CCY}_Yield10Y_Available`
- `{CCY}_GDP`
- `{CCY}_GDP_growth`
- `{CCY}_CPI`
- `{CCY}_CPIInfl`
- `{CCY}_PPI`
- `{CCY}_PPIInfl`

The final model inputs are derived again from these base columns.

## Feature blocks actually fed into `oursmain`

Under the experiment pipeline, `oursmain` uses the following feature templates.

### Local features

- `{ccy}_FX_ret_1d`
- `{ccy}_FX_mom_5`
- `{ccy}_FX_mom_20`
- `{ccy}_FX_vol_5`
- `{ccy}_FX_vol_20`
- `{ccy}_FX_zscore_20`

### Rate features

- `{ccy}_Yield10Y`
- `{ccy}_Yield10Y_change`
- `{ccy}_Yield10Y_minus_US10Y`
- `{ccy}_RealRate10Y`
- `{ccy}_Yield10Y_Available`
- Broadcast global inputs:
  - `Global_US1Y`
  - `Global_US2Y`
  - `Global_US5Y`
  - `Global_US10Y`
  - `Global_US10Y_minus_US2Y`
  - `Global_US2Y_change`
  - `Global_US10Y_change`

### Equity features

- `{ccy}_Stock_ret`
- `{ccy}_Stock_mom_5`
- `{ccy}_Stock_mom_20`
- `{ccy}_Stock_vol_20`
- `{ccy}_Stock_ret_minus_SP500_ret`

### Country macro features

- `{ccy}_GDP_growth`
- `{ccy}_CPIInfl`
- `{ccy}_PPIInfl`
- `{ccy}_GDP_growth_minus_USD`
- `{ccy}_CPIInfl_minus_USD`
- `{ccy}_PPIInfl_minus_USD`

### Global features

- `Global_VIX`
- `Global_VIX_change`
- `Global_BroadDollar_ret`
- `Global_DXY_ret`
- `Global_SP500_ret`
- `Global_Oil_ret`
- `Global_Gold_ret`
- `Global_Copper_ret`
- `Global_US2Y_change`
- `Global_US10Y_change`
- `Global_US10Y_minus_US2Y`

## Data sources

### Data collected from FRED

Examples:

- `VIXCLS` -> `Global_VIX`
- `DTWEXBGS` -> `Global_BroadDollar`
- `DGS1`, `DGS2`, `DGS5`, `DGS10` -> US Treasury yields
- country 10Y yield proxies such as:
  - `IRLTLT01DEM156N`
  - `IRLTLT01JPM156N`
  - `IRLTLT01GBM156N`
  - `IRLTLT01CAM156N`
  - `IRLTLT01AUM156N`
  - `IRLTLT01KRM156N`
  - `IRLTLT01CHM156N`
  - `IRLTLT01NZM156N`
  - `IRLTLT01SEM156N`
  - `IRLTLT01NOM156N`
- low-frequency macro series for GDP / CPI / PPI

### Data collected from yfinance

Examples:

- `^GSPC` -> S&P 500 / US equity proxy
- `^GDAXI`, `^N225`, `^FTSE`, `^GSPTSE`, `^AXJO`, `^KS11`, `^SSMI`, `^NZ50`, `EWD`, `OBX.OL`
- `CL=F`, `GC=F`, `HG=F`
- `DX-Y.NYB` -> DXY

### Legacy aligned source

`factor_daily_alligned_krw.csv` is treated as the aligned FX/target base dataset inherited from the original research repository.

It is not a newly fetched raw-source table in the current shared pipeline.

## Treatment of low-frequency data

Based on the processing report:

- GDP / CPI / PPI / some 10Y yield proxy series are forward-filled to business days after applying publication lags
- values are not backfilled before the first available release
- therefore, missing values in the early part of the sample are intentional

## Current limitations of the data

- `RealRate10Y` is heavily missing for many countries
- some `PPI / PPIInfl` series are unavailable
- some country macro series such as `CHF_GDP` are unavailable
- direct foreign flow or daily current-account style variables are not included in the current `oursmain` inputs

## One-line summary

`oursmain` takes FX levels and targets from `factor_daily_alligned_krw.csv`, combines them with rate / equity / macro / global features from `score_vA_nonfx_features.csv`, merges the two files on `Date`, and uses the result as input to the relational score-graph model.
