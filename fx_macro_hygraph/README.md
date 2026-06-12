# FX Strength Research Package

This repository is a compact shareable package for the FX strength prediction project. It includes the selected final model `Ours`, its main ablations, the core experiment scripts, and curated result artifacts.

## Overview

The project predicts next-period FX return structure by combining:
- currency-level local time-series features
- global macro features
- graph-based cross-currency interaction modeling

The main proposed model is `Ours`, implemented as `MACROHyGraph` in `src/models.py`. Key ablation variants use explicit names such as `PureGraphFX`, `FiLMHyGraph`, `NoGraph`, `NoMacro`, and `StaticGraph`.

## Repository Layout

- `data/`
  - Input dataset used by the included experiments
  - Main file: `factor_daily_legacy.csv`

- `src/`
  - Core modeling pipeline
  - `config.py`: central configuration
  - `dataset.py`: data loading, feature building, normalization, dataloaders
  - `models.py`: `Ours`, baselines, and ablation models
  - `train.py`: losses, trainer, evaluation metrics
  - `utils.py`: common utility helpers

- `scripts/`
  - Shared helper modules used across experiments
  - `exp_utils.py`: shared train/predict/evaluate utilities
  - `ablation_utils.py`: component ablation helpers
  - `stress_utils.py`: stress regime helper logic
  - `exp_graph_adaptation_analysis.py`: graph adaptation analysis
  - `plot_graph_change_shock_focus.py`: shock-focused graph change plots

- `experiments/`
  - Main executable experiment entrypoints
  - `exp_prediction_metrics_comparison.py`: prediction accuracy comparison
  - `exp_portfolio_investment_comparison2.py`: portfolio backtest comparison
  - `exp_topk_sensitivity_ablation.py`: top-k graph sparsity sensitivity
  - `exp_full_ablation_macro_hygraph.py`: full ablation benchmark
  - `exp_component_ablation.py`: targeted component ablation
  - `exp_stress_regime_performance.py`: stress-regime performance analysis
  - `exp_stress_regime_macro_hygraph.py`: saved-prediction stress comparison

- `results/`
  - Curated experiment outputs already generated in this package
  - `model_prediction_comparison/`
  - `model_portfolio_comparison/`
  - `topk_sensitivity_ablation/`
  - `model_ablation/`
  - `stress_regime_analysis/`

## Naming Conventions

- `Ours`
  - Final selected model used throughout the package

- `FiLMHyGraph`
  - Hybrid graph model with FiLM-style macro conditioning

- `PureGraphFX`
  - Graph-only ablation without the direct branch

- `NoGraph`, `NoMacro`, `StaticGraph`
  - Explicit ablation names used consistently in scripts and outputs

Older short version labels like `V3`, `V4`, and `V5` were removed from the active workflow in favor of descriptive names.

## Quick Start

Run commands from the repository root.

```powershell
python experiments/exp_prediction_metrics_comparison.py
python experiments/exp_portfolio_investment_comparison2.py --cost-bps-grid 0 1 2 5 10
python experiments/exp_topk_sensitivity_ablation.py --models Ours --topk-values 2 4 6 8 10
python experiments/exp_full_ablation_macro_hygraph.py
python experiments/exp_component_ablation.py
python experiments/exp_stress_regime_performance.py
python scripts/exp_graph_adaptation_analysis.py
```

## Typical Workflow

1. Run `experiments/exp_prediction_metrics_comparison.py` to compare predictive performance across models.
2. Run `experiments/exp_portfolio_investment_comparison2.py` to translate predictions into portfolio results.
3. Run `experiments/exp_full_ablation_macro_hygraph.py` or `experiments/exp_component_ablation.py` to analyze which model components matter most.
4. Run `experiments/exp_stress_regime_performance.py` and `scripts/exp_graph_adaptation_analysis.py` to inspect stress robustness and graph adaptation behavior.

## Notes

- Default paths in the code assume this repository layout: `data/`, `src/`, `scripts/`, `experiments/`, and `results/`.
- Included `results/` artifacts are snapshots for sharing and analysis; rerunning experiments may overwrite parts of them.
- The package is intentionally trimmed to the main model, ablations, and primary experiment outputs rather than all exploratory work.
