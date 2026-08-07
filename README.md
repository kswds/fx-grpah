# ARC_FX: Anonymous Repository for Model Implementations and Results
This repository provides an anonymous implementation package for `ARC_FX`, including the model implementation, baseline implementations, training configurations, and the main predictive comparison results reported in the paper.

`ARC_FX` is the name used in the paper for the model implemented internally as `oursmain`.


## Included
- ARC_FX model implementation
- baseline implementations
- processed-data pipeline code
- training and evaluation entrypoints
- configuration file for the main experiment
- documentation for model settings and hyperparameters

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
