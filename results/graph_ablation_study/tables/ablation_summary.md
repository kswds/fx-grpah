# Graph Ablation Study

Stages:
- `no_graph`: relational branch removed
- `static_graph`: static graph only
- `static_plus_dynamic_graph`: static + dynamic graph
- `oursmain`: same full graph backbone as `static_plus_dynamic_graph`, but trained with the oursmain loss

## Predictive Metrics

| Stage | Mean Hit | Mean RMSE | Mean MAE | Mean Pairwise Hit | Mean IC | Mean LS Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oursmain | 0.5057 | 0.005474 | 0.003953 | 0.5054 | 0.0308 | 0.3489 |
| static_graph | 0.4987 | 0.005436 | 0.003923 | 0.5021 | 0.0209 | 0.7810 |
| static_plus_dynamic_graph | 0.4984 | 0.005436 | 0.003923 | 0.4989 | 0.0103 | 0.6765 |
| no_graph | 0.4966 | 0.005435 | 0.003923 | 0.4938 | 0.0003 | 0.4690 |

## Prediction Transitions

| Transition | Mean Abs Prediction Change | Prediction Corr | MAE Delta | Improved Obs Ratio | Worsened Obs Ratio | Same Sign Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamic_minus_nograph | 0.000176 | 0.6355 | -0.000000 | 0.4287 | 0.4285 | 0.7672 |
| dynamic_minus_static | 0.000173 | 0.6686 | 0.000000 | 0.4238 | 0.4333 | 0.7893 |
| oursmain_minus_dynamic | 0.000506 | 0.2236 | 0.000025 | 0.4150 | 0.4421 | 0.6393 |
| oursmain_minus_nograph | 0.000533 | 0.1516 | 0.000025 | 0.4149 | 0.4422 | 0.5866 |
| oursmain_minus_static | 0.000535 | 0.1972 | 0.000026 | 0.4125 | 0.4446 | 0.5946 |
| static_minus_nograph | 0.000178 | 0.6793 | -0.000001 | 0.4318 | 0.4253 | 0.7991 |