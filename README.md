# ARC_FX Anonymous Reproduction Package

This folder is a compact anonymous GitHub package for reproducing the main predictive comparison of `ARC_FX`.

`ARC_FX` is the paper name for the internal implementation `oursmain`.

## Included

- ARC_FX model implementation
- baseline implementations
- processed-data pipeline code
- training and evaluation entrypoints
- configuration file for the main experiment
- documentation for model settings and hyperparameters

## Not Included

- raw source downloads
- data-collection scripts
- vendor credentials
- raw input datasets

The code starts from the processed panel described in [data/README.md](data/README.md).

## Repository Layout

```text
repository/
├── README.md
├── requirements.txt
├── configs/
│   └── main_experiment.yaml
├── docs/
│   ├── model_and_hyperparameters.md
│   └── models_details.md
├── data/
│   └── README.md
└── src/
    ├── models/
    │   └── arc_fx.py
    ├── baselines/
    │   └── models.py
    ├── train.py
    ├── evaluate.py
    ├── training_core.py
    ├── data_pipeline.py
    ├── metrics.py
    └── config.py
```

## Main Predictive Experiment

Headline setting:

- Universe: `USD, EUR, JPY, GBP, CAD, AUD, KRW, CHF, NZD, SEK, NOK`
- Lookback: `10`
- ARC_FX graph sparsity: `top-k = 3`
- Seeds: `21, 42, 456`

Baselines in the released comparison:

- `MLP`
- `Transformer`
- `GNN`
- `Corr-LSTM-GAT`
- `FXRP`

`Corr-LSTM-GAT` and `FXRP` are reference baselines and are documented as such in [docs/model_and_hyperparameters.md](docs/model_and_hyperparameters.md).

## Installation

```bash
pip install -r requirements.txt
```

## Expected Processed Data

Place the following files under `data/processed/`:

- `factor_daily_alligned_krw.csv`
- `score_vA_nonfx_features.csv`

The expected column conventions and usage notes are summarized in [data/README.md](data/README.md).

## Train From Scratch

```bash
python src/train.py --config configs/main_experiment.yaml --device cpu
```

This trains:

- `ARC_FX`
- `MLP`
- `Transformer`
- `GNN`
- `Corr-LSTM-GAT`
- `FXRP`

Prediction files are written under `results/repro_runs/`.

## Build The Main Predictive Report

```bash
python src/evaluate.py --config configs/main_experiment.yaml
```

This creates:

- `results/predictive_results/final_overall_prediction.md`
- `results/predictive_results/model_comparison_detail.csv`
- `results/predictive_results/model_comparison_aggregate.csv`
- `results/predictive_results/nontrivial_directional_detail.csv`
- `results/predictive_results/nontrivial_directional_aggregate.csv`

## Notes On The Released Comparison

The headline markdown report intentionally omits `IC` from the public table.

The released baseline comparison uses the same active directional core as ARC_FX on non-USD targets above the small-return threshold, while ARC_FX keeps its own graph/component regularization terms.

The CSV outputs use public-facing model names such as `ARC_FX`, `MLP`, `Transformer`, `GNN`, `Corr-LSTM-GAT`, and `FXRP`.
