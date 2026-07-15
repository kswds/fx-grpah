# Graph Ablation Study

Stages:
- `no_graph`: relational branch removed
- `static_graph`: static graph only
- `static_plus_dynamic_graph`: static + dynamic graph

## Predictive Metrics

| Stage | Mean Hit | Mean RMSE | Mean MAE | Mean Pairwise Hit | Mean IC | Mean LS Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static_graph | 0.5002 | 0.005435 | 0.003920 | 0.4990 | 0.0083 | 0.4754 |
| no_graph | 0.4997 | 0.005438 | 0.003925 | 0.4949 | 0.0017 | 0.2635 |
| static_plus_dynamic_graph | 0.4976 | 0.005437 | 0.003923 | 0.5023 | 0.0206 | 0.7406 |

## Prediction Transitions

| Transition | Mean Abs Prediction Change | Prediction Corr | MAE Delta | Improved Obs Ratio | Worsened Obs Ratio | Same Sign Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamic_minus_nograph | 0.000144 | 0.7469 | -0.000002 | 0.4328 | 0.4243 | 0.8126 |
| dynamic_minus_static | 0.000130 | 0.7510 | 0.000002 | 0.4272 | 0.4299 | 0.8097 |
| static_minus_nograph | 0.000149 | 0.7580 | -0.000004 | 0.4350 | 0.4222 | 0.8067 |