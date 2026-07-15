# Refined All-10 Hyperparameter Search

Fixed top_k: `7`

Custom currencies:
- `EUR, JPY, GBP, CAD, AUD, KRW, CHF, NZD, SEK, NOK`

## oursmain

Rank metric: `hit_ratio`

| Trial | Hit | RMSE | Pairwise Hit | IC | LS Sharpe | hidden | graph_rank | dropout | edge_dropout | spectral_bound | lr | lambda_dir | lambda_rank | small_return_q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oursmain_trial0001 | 0.5016 | 0.006300 | 0.4935 | -0.0060 | -0.7685 | 48 | 8 | 0.20 | 0.05 | 1.00 | 0.00030 | 0.100 | 0.080 | 0.35 |
| oursmain_trial0003 | 0.5015 | 0.006300 | 0.4946 | -0.0019 | -0.5523 | 48 | 8 | 0.25 | 0.05 | 1.00 | 0.00030 | 0.100 | 0.080 | 0.35 |
| oursmain_trial0004 | 0.5009 | 0.006300 | 0.4959 | 0.0009 | -0.5282 | 48 | 8 | 0.25 | 0.05 | 1.00 | 0.00030 | 0.100 | 0.080 | 0.40 |
| oursmain_trial0002 | 0.5005 | 0.006299 | 0.4945 | -0.0028 | -0.7152 | 48 | 8 | 0.20 | 0.05 | 1.00 | 0.00030 | 0.100 | 0.080 | 0.40 |
