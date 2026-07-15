# Predictive Comparison

| Model | Universe | Lookback | Mean Hit | Mean RMSE | Mean MAE | Mean Pairwise Hit | Mean IC | Mean LS Sharpe |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oursmain | core6 | 10 | 0.5057 | 0.005474 | 0.003953 | 0.5054 | 0.0308 | 0.3489 |
| static_graph | core6 | 10 | 0.4987 | 0.005436 | 0.003923 | 0.5021 | 0.0209 | 0.7810 |
| static_plus_dynamic_graph | core6 | 10 | 0.4984 | 0.005436 | 0.003923 | 0.4989 | 0.0103 | 0.6765 |
| no_graph | core6 | 10 | 0.4966 | 0.005435 | 0.003923 | 0.4938 | 0.0003 | 0.4690 |