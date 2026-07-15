# Predictive Comparison

| Model | Universe | Lookback | Mean Hit | Mean RMSE | Mean MAE | Mean Pairwise Hit | Mean IC | Mean LS Sharpe |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oursmain | core6 | 10 | 0.5041 | 0.005491 | 0.003979 | 0.5030 | 0.0229 | 0.9099 |
| static_graph | core6 | 10 | 0.5032 | 0.005436 | 0.003931 | 0.4936 | 0.0046 | 0.7953 |
| no_graph | core6 | 10 | 0.4889 | 0.005455 | 0.003949 | 0.4876 | -0.0164 | -0.0763 |
| static_plus_dynamic_graph | core6 | 10 | 0.4735 | 0.005482 | 0.003972 | 0.4915 | -0.0075 | -0.3989 |