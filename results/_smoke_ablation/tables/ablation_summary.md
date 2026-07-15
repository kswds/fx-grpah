# Graph Ablation Study

Stages:
- `no_graph`: relational branch removed
- `static_graph`: static graph only
- `static_plus_dynamic_graph`: static + dynamic graph

## Predictive Metrics

| Stage | Mean Hit | Mean RMSE | Mean MAE | Mean Pairwise Hit | Mean IC | Mean LS Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static_plus_dynamic_graph | 0.4863 | 0.005460 | 0.003950 | 0.4949 | -0.0031 | -0.2012 |
| no_graph | 0.4854 | 0.005488 | 0.003992 | 0.4867 | -0.0228 | -0.3350 |
| static_graph | 0.4829 | 0.005464 | 0.003964 | 0.4939 | -0.0036 | 0.2496 |

## Prediction Transitions

| Transition | Mean Abs Prediction Change | Prediction Corr | MAE Delta | Improved Obs Ratio | Worsened Obs Ratio | Same Sign Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamic_minus_nograph | 0.000528 | 0.1274 | -0.000036 | 0.4473 | 0.4099 | 0.6391 |
| dynamic_minus_static | 0.000496 | 0.1709 | -0.000012 | 0.4410 | 0.4162 | 0.6168 |
| static_minus_nograph | 0.000544 | 0.3874 | -0.000024 | 0.4432 | 0.4139 | 0.6598 |