# FX Relational Forecasting Pipeline

This repository is a compact research share of our FX forecasting pipeline centered on `oursmain`, a relational score-graph model for predicting next-period cross-currency FX returns.

The goal of this README is to help a new reader understand:

- what problem the project is trying to solve
- how the prediction task is defined
- what data is used
- how the model is designed
- how to reproduce the main experiments

## What Problem Are We Solving?

We study cross-sectional FX forecasting.

At each date, the model observes recent FX, rates, equity, macro, and global market information and predicts the next-period relative return of multiple currencies against USD.

This is not framed as a pure single-series time-series problem. Instead, it is a multi-currency relational problem:

- currencies move together
- macro shocks propagate across countries
- the importance of pairwise currency relationships changes over time
- the same global event can affect different currencies in different ways

Because of that, we model currencies jointly rather than fitting a separate forecasting model to each FX series in isolation.

## Problem Setup

The prediction target is the next-step forward FX return for each non-USD currency.

In code terms, the pipeline predicts a score for each currency and then pins everything relative to USD:

- the model first produces a latent currency score vector
- the USD score is subtracted from all currency scores
- the resulting output is interpreted as the predicted next-period USD-relative FX return

This design makes the task naturally cross-sectional and keeps the output anchored to a common reference currency.

### Supported currency universes

The repository includes three preset universes in [src/config.py]:

- `major3`: `USD, EUR, JPY, GBP`
- `core6`: `USD, EUR, JPY, GBP, CAD, AUD, CHF`
- `krw7`: `USD, EUR, JPY, GBP, CAD, AUD, KRW, CHF`

The experiment scripts also support `--custom-currencies`, so larger custom universes such as a full 10-currency set can be run without changing the model code.

## Data

The training pipeline uses two processed CSV files:

- `data/processed/factor_daily_alligned_krw.csv`
- `data/processed/score_vA_nonfx_features.csv`

These two files are inner-joined on `Date`.

### 1. `factor_daily_alligned_krw.csv`

This is the aligned FX-and-target base dataset.

It provides:

- FX levels for each currency
- aligned target FX levels
- precomputed forward FX return targets
- some legacy market variables used in the original research pipeline

In practice, this file defines the core FX timeline and target construction.

### 2. `score_vA_nonfx_features.csv`

This is the main non-FX feature source.

It provides:

- local equity features
- country yield features
- inflation and GDP features
- global market and macro variables

This file contains most of the explanatory information used by the model beyond raw FX levels.

### Feature groups actually used by the model

The pipeline organizes inputs into five main groups:

1. `local`
- FX return
- FX momentum
- FX volatility
- FX z-score

2. `rate`
- local 10Y yield
- yield change
- yield spread versus US 10Y
- availability flag
- broadcast US curve information

3. `equity`
- local stock return
- stock momentum
- stock volatility
- relative return versus S&P 500

4. `countrymacro`
- GDP growth
- CPI inflation
- PPI inflation
- country-minus-US macro differences

5. `global`
- VIX
- broad dollar / DXY signals
- S&P 500 return
- oil, gold, copper returns
- US curve slope and yield changes

For `oursmain`, the pipeline can also generate regime one-hot features such as:

- low / mid / high volatility
- risk-off
- dollar shock
- commodity shock

More detailed data notes are documented in [data/README.md].

## Model Family

The main model implementation is in [src/models_ours.py].

The repository contains two groups of models:

- relational models
- baseline models

### Relational models

- `oursmain`
- `foundation_relational`
- `foundation_static`
- `foundation_nograph`

### Baseline models

- `mlp`
- `lstm`
- `gru`
- `gnn`

## Model Design

### High-level idea

`oursmain` is a multi-block relational forecasting model.

Each currency is represented using several feature blocks, and the model learns:

- what each feature block says about that currency on its own
- how currencies influence one another through a graph
- how the importance of different components changes over time

### Block encoders

Each input block is encoded separately.

For local, rate, equity, and country macro inputs, the model uses:

- an input projection
- a GRU sequence encoder
- attention pooling over the lookback window

This produces a compact per-currency representation for each block.

The global block is encoded with a similar GRU-plus-attention structure, but at the global level rather than per currency.

### Currency-level representation

For each currency node, the model combines:

- local FX representation
- rate representation
- equity representation
- country macro representation
- global representation
- a learned currency embedding

This gives a shared hidden representation per currency before relational message passing.

### Relational graph

The distinctive part of the model is the graph layer.

The graph can contain:

- a static graph component
- a dynamic graph component

The static component is a learned persistent structure that captures stable cross-currency relationships.

The dynamic component changes over time using:

- global state-dependent low-rank graph updates
- node-to-node attention-based edge scores

The graph is sparsified with a top-`k` mechanism, so each currency only keeps its strongest connections.

### Message passing

Once the graph is constructed, the model performs message passing across currencies.

This produces a relational representation for each currency that captures:

- spillovers
- relative positioning
- cross-currency dependence

### Component scoring and gates

The model then computes component-level scores:

- local
- rate
- equity
- macro
- relational

A learned gate determines how much weight to place on each component for each currency at each date.

This allows the model to adapt across regimes. For example:

- in one regime, rates may matter more
- in another, cross-currency contagion may dominate

### Final prediction

The model combines the component scores into a currency score vector and then subtracts the USD score.

That final USD-pinned vector is the predicted next-period FX return vector.

## What Makes `oursmain` Different?

Structurally, `oursmain` and `foundation_relational` share the same full relational backbone:

- both use static and dynamic graph components
- both use the same multi-block encoder design
- both use the same component gate mechanism

The main difference is the training objective.

### `foundation_relational`

This model uses the base relational loss:

- regression error
- directional loss
- ranking loss
- graph regularization

It is the clean full-graph backbone reference model.

### `oursmain`

This model uses a modified objective that emphasizes meaningful directional moves and includes a small-return classification-style loss.

Intuitively, it is designed to care less about tiny noisy moves and focus more on whether the model gets the direction right when the move matters.

So:

- `foundation_relational` is the full relational backbone baseline
- `oursmain` is the practical tuned version of that backbone

## Ablation Structure

The graph ablation experiment uses the following sequence:

- `foundation_nograph`
  - no graph
- `foundation_static`
  - static graph only
- `foundation_relational`
  - static + dynamic graph
- `oursmain`
  - same full graph backbone, but with the `oursmain` loss

This lets us separately study:

- the value of adding any graph at all
- the value of adding dynamic graph structure beyond static edges
- the value of the `oursmain` training objective on top of the full graph architecture

## Evaluation

The repository includes both prediction metrics and investment-style metrics.

### Prediction metrics

Defined in [src/metrics.py]:

- RMSE
- MAE
- hit ratio
- non-tiny hit ratio
- extreme hit ratio
- pairwise hit
- information coefficient (IC)

### Long-short metrics

The pipeline also evaluates the quality of the forecast ranking using long-short portfolio style summaries:

- long-short Sharpe
- long-short Sortino
- cumulative return
- max drawdown

There is also a dedicated long-short backtest comparison with transaction costs.

## Repository Structure

- `data/processed`
  - processed datasets used by the shared pipeline
- `src`
  - model, data, training, and metric code
- `experiments`
  - experiment entrypoints
- `scripts`
  - convenience scripts for common runs
- `results`
  - experiment outputs

## Main Experiment Scripts

- `experiments/run_model_comparison.py`
  - main benchmark comparison across models
- `experiments/run_extreme_regime_comparison.py`
  - compares model behavior during volatility spikes and macro shock regimes
- `experiments/run_long_short_backtest_comparison.py`
  - compares long-short portfolio performance under `0 / 2 / 5 bp` costs
- `experiments/run_graph_ablation_study.py`
  - compares no-graph, static-graph, full-graph, and `oursmain`
- `experiments/run_all10_hparam_search.py`
  - hyperparameter search for larger 10-currency runs

## Quick Start

### 1. Copy data

```powershell
powershell -ExecutionPolicy Bypass -File scripts\copy_local_data.ps1
```

### 2. Run the default model comparison

```powershell
C:\Python\python.exe .\experiments\run_model_comparison.py `
  --models oursmain foundation_relational mlp lstm gru gnn `
  --universe core6 `
  --lookback 10 `
  --epochs 80 `
  --seeds 42 123 456 `
  --output-dir .\results\core6_compare
```

Or:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_core6.ps1
```

### 3. Run extreme-regime comparison

```powershell
C:\Python\python.exe .\experiments\run_extreme_regime_comparison.py `
  --models oursmain foundation_relational mlp lstm gru gnn `
  --universe core6 `
  --lookback 10 `
  --epochs 80 `
  --seeds 42 123 456 `
  --output-dir .\results\core6_extreme_compare
```

### 4. Run long-short backtest comparison

```powershell
C:\Python\python.exe .\experiments\run_long_short_backtest_comparison.py `
  --models oursmain foundation_relational mlp lstm gru gnn `
  --universe core6 `
  --lookback 10 `
  --epochs 80 `
  --seeds 42 123 456 `
  --transaction-cost-bps 0 2 5 `
  --output-dir .\results\core6_portfolio_compare
```

### 5. Run graph ablation

```powershell
C:\Python\python.exe .\experiments\run_graph_ablation_study.py `
  --universe core6 `
  --lookback 10 `
  --epochs 80 `
  --seeds 42 123 456 `
  --output-dir .\results\core6_graph_ablation
```

### 6. Run all-10 hyperparameter search

```powershell
C:\Python\python.exe .\experiments\run_all10_hparam_search.py `
  --models oursmain foundation_relational `
  --epochs 20 `
  --seeds 42 `
  --output-dir .\results\all10_hparam_search
```

By default, this script now runs a fast all-10 search that varies `top_k` from `2` to `7` while keeping the other relational hyperparameters fixed at the current tuned defaults.

## Main Output Files

- `results/.../tables/metrics_summary.csv`
  - raw seed-level prediction metrics
- `results/.../tables/seed_metrics_aggregate.csv`
  - aggregated model comparison metrics
- `results/.../tables/metrics_summary.md`
  - short comparison summary
- `results/.../tables/extreme_metrics_summary.md`
  - regime-conditional performance summary
- `results/.../tables/scenario_flags.csv`
  - date-level regime labels used in extreme-regime analysis
- `results/.../tables/portfolio_summary.md`
  - long-short performance summary under transaction costs
- `results/.../tables/seed_portfolio_metrics_aggregate.csv`
  - aggregated long-short portfolio results
- `results/.../tables/portfolio_daily_returns.csv`
  - daily gross/net returns, turnover, and cost series
- `results/.../tables/prediction_comparison.csv`
  - aligned stage-by-stage predictions for graph ablation
- `results/.../tables/transition_aggregate.csv`
  - summary of prediction changes between ablation stages
- `results/.../tables/trial_metrics_aggregate.csv`
  - aggregated trial-level results for all-10 tuning
- `results/.../tables/best_trials.csv`
  - best hyperparameter configuration per model
- `results/.../tables/hparam_search_summary.md`
  - Markdown summary of top all-10 tuning trials
- `results/.../<model>_seed*/predictions/*.parquet`
  - stored prediction outputs
- `results/.../<model>_seed*/explanations/*.parquet`
  - component and edge-level explanations for relational models

## Notes

- This repository is a minimal share, not the full original research repository.
- `oursmain` explanations can be inspected through `components.parquet` and `top_edges.parquet`.
- If you run large-universe experiments, it is usually better to retune graph-related hyperparameters such as `top_k`, `graph_rank`, and regularization strengths rather than directly reusing the `core6` defaults.
