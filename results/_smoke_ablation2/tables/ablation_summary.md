# Graph Ablation Study

Stages:
- `no_graph`: relational branch removed
- `static_graph`: static graph only
- `static_plus_dynamic_graph`: static + dynamic graph
- `oursmain`: same full graph backbone as `static_plus_dynamic_graph`, but trained with the oursmain loss

## Predictive Metrics

| Stage | Mean Hit | Mean RMSE | Mean MAE | Mean Pairwise Hit | Mean IC | Mean LS Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oursmain | 0.5041 | 0.005491 | 0.003979 | 0.5030 | 0.0229 | 0.9099 |
| static_graph | 0.5032 | 0.005436 | 0.003931 | 0.4936 | 0.0046 | 0.7953 |
| no_graph | 0.4889 | 0.005455 | 0.003949 | 0.4876 | -0.0164 | -0.0763 |
| static_plus_dynamic_graph | 0.4735 | 0.005482 | 0.003972 | 0.4915 | -0.0075 | -0.3989 |

## Prediction Transitions

| Transition | Mean Abs Prediction Change | Prediction Corr | MAE Delta | Improved Obs Ratio | Worsened Obs Ratio | Same Sign Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamic_minus_nograph | 0.000592 | 0.0129 | 0.000019 | 0.4180 | 0.4392 | 0.4197 |
| dynamic_minus_static | 0.000540 | -0.1630 | 0.000035 | 0.4126 | 0.4445 | 0.4937 |
| oursmain_minus_dynamic | 0.000756 | -0.0139 | 0.000006 | 0.4231 | 0.4341 | 0.4746 |
| oursmain_minus_nograph | 0.000547 | 0.2661 | 0.000025 | 0.4165 | 0.4406 | 0.6923 |
| oursmain_minus_static | 0.000572 | 0.2865 | 0.000041 | 0.4070 | 0.4501 | 0.6685 |
| static_minus_nograph | 0.000375 | 0.3192 | -0.000015 | 0.4414 | 0.4158 | 0.6756 |