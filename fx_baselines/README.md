# FX Graph Research

- **Baseline (MLP)**: strong non-graph model (shared GRU per currency + heterogeneous macro injection)
- **Static FC Graph**: fully-connected graph aggregation + macro injection
- **Granger-proxy + Shock Propagation**: directed Granger-proxy graph from TRAIN data + propagate → macro shock → propagate

## Key updates

### 1) Target is RAW FX log return

- We use a consistent USD-base convention (see `dataset.py: fx_to_log()`).
- Target **Y is not normalized**: `Y_raw[t, i] = FXRet[t, i]` where FXRet is the log-diff of the USD-base log price.
- All reported metrics are computed **in RAW return space**.

### 2) No leakage scaling

Only **inputs** are standardized using **TRAIN-only** statistics:

- `X_local_scaled = (X_local_raw - mean_train) / std_train`
- `X_macro_scaled = (X_macro_raw - mean_train) / std_train`

Targets are untouched.

### 3) Extreme-move metrics

Extreme threshold is defined by TRAIN distribution:

- `thr = quantile(|Y_train|, q=extreme_percentile)`
- We report:
  - overall Hit / RMSE / MAE
  - Normal vs Extreme Hit/RMSE
  - Weighted Hit (sign accuracy weighted by |y|)

### 4) Shock propagation model added
- Instead of using fully-connected graph, correlation based edge selection method is introduced.
- By defining edges through lead-lag correlations, the model explicitly accounts for the time-lagged spillover effects of macro shocks across the global currency network.

## Files

- `config.py` : experiment configuration
- `dataset.py` : data processing + dataset (includes safe log to avoid invalid log warnings)
- `models.py` : MLP baseline + FC graph model + Granger shock-prop model
- `train.py` : train/eval loop & metrics
- `main.py` : multi-seed runner producing `results/results_all_compare_rawY.json`
- `utils.py` : seed, CI summary, JSON writer


Output:

- `results/results_all_compare_rawY.json`

## Notes 

- Replace Granger-proxy lag-1 correlation with proper VAR/Granger tests (sparse VAR, Lasso VAR).
- Add time-varying graph (macro → adjacency) and/or triangular-arbitrage constraints.
- Add causality validation (lead-lag, event studies, shock decomposition).
