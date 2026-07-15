# 10-Currency Forecast Performance Report

## Scope

This report summarizes the results in:

- `results/all10_compare/tables/seed_metrics_detail.csv`
- `results/all10_compare/tables/seed_metrics_aggregate.csv`

The run was executed on the full 10-currency set:

- `EUR`
- `JPY`
- `GBP`
- `CAD`
- `AUD`
- `KRW`
- `CHF`
- `NZD`
- `SEK`
- `NOK`

Internal prediction files confirm that the model output contains `USD + 10 non-USD currencies` (11 currencies total including USD as the anchor).

Note:
The summary CSV still shows `universe=core6`, but that is only a metadata label carried over from the command arguments. The actual prediction outputs are for the full 10-currency run.

## Aggregate Ranking

### Sorted by Hit Ratio

| Rank | Model | Hit Ratio | RMSE | MAE | Extreme Hit | Pairwise Hit | IC | LS Sharpe |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `mlp` | 0.5017 | 0.006182 | 0.004421 | 0.5232 | 0.4957 | 0.0026 | 0.0785 |
| 2 | `foundation_relational` | 0.5016 | 0.006181 | 0.004420 | 0.5215 | 0.4939 | -0.0034 | 0.2071 |
| 3 | `gnn` | 0.4997 | 0.006178 | 0.004418 | 0.5217 | 0.4958 | 0.0043 | 0.0271 |
| 4 | `oursmain` | 0.4965 | 0.006235 | 0.004464 | 0.4945 | 0.4939 | -0.0038 | -0.3214 |
| 5 | `gru` | 0.4948 | 0.006188 | 0.004426 | 0.5107 | 0.4953 | 0.0005 | 0.1596 |
| 6 | `lstm` | 0.4924 | 0.006185 | 0.004426 | 0.5125 | 0.4935 | -0.0023 | -0.0124 |

### Sorted by RMSE

| Rank | Model | RMSE | Hit Ratio | MAE | Extreme Hit | Pairwise Hit | IC | LS Sharpe |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `gnn` | 0.006178 | 0.4997 | 0.004418 | 0.5217 | 0.4958 | 0.0043 | 0.0271 |
| 2 | `foundation_relational` | 0.006181 | 0.5016 | 0.004420 | 0.5215 | 0.4939 | -0.0034 | 0.2071 |
| 3 | `mlp` | 0.006182 | 0.5017 | 0.004421 | 0.5232 | 0.4957 | 0.0026 | 0.0785 |
| 4 | `lstm` | 0.006185 | 0.4924 | 0.004426 | 0.5125 | 0.4935 | -0.0023 | -0.0124 |
| 5 | `gru` | 0.006188 | 0.4948 | 0.004426 | 0.5107 | 0.4953 | 0.0005 | 0.1596 |
| 6 | `oursmain` | 0.006235 | 0.4965 | 0.004464 | 0.4945 | 0.4939 | -0.0038 | -0.3214 |

## Full Metric Table

| Model | RMSE | MAE | Hit Ratio | Non-Tiny Hit | Extreme Hit | Pairwise Hit | IC | LS Sharpe | LS Sortino | Max Drawdown | Cumulative Return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `foundation_relational` | 0.006181 | 0.004420 | 0.5016 | 0.5234 | 0.5215 | 0.4939 | -0.0034 | 0.2071 | 0.3287 | -0.1234 | 0.0570 |
| `gnn` | 0.006178 | 0.004418 | 0.4997 | 0.5180 | 0.5217 | 0.4958 | 0.0043 | 0.0271 | 0.0586 | -0.1185 | 0.0078 |
| `gru` | 0.006188 | 0.004426 | 0.4948 | 0.5141 | 0.5107 | 0.4953 | 0.0005 | 0.1596 | 0.2489 | -0.1084 | 0.0427 |
| `lstm` | 0.006185 | 0.004426 | 0.4924 | 0.5111 | 0.5125 | 0.4935 | -0.0023 | -0.0124 | -0.0235 | -0.1583 | -0.0009 |
| `mlp` | 0.006182 | 0.004421 | 0.5017 | 0.5175 | 0.5232 | 0.4957 | 0.0026 | 0.0785 | 0.1161 | -0.1653 | 0.0209 |
| `oursmain` | 0.006235 | 0.004464 | 0.4965 | 0.5036 | 0.4945 | 0.4939 | -0.0038 | -0.3214 | -0.4256 | -0.1888 | -0.0900 |

## Main Takeaways

- `mlp` achieved the highest average directional hit ratio (`0.5017`), although the margin over `foundation_relational` (`0.5016`) is extremely small.
- `gnn` achieved the best RMSE (`0.006178`) and MAE (`0.004418`) among the models.
- `foundation_relational` was the strongest relational model in the 10-currency setting and also had the best long-short Sharpe among the six models (`0.2071`).
- `oursmain` underperformed relative to the other models in this 10-currency run:
  it had the weakest RMSE, MAE, extreme-hit ratio, Sharpe, Sortino, and cumulative return.
- The spread between the top models is narrow on point-forecast metrics, but much wider on long-short portfolio-style metrics.

## Suggested Reading of the Result

- If the focus is directional classification accuracy, `mlp` and `foundation_relational` are effectively tied at the top.
- If the focus is regression fit, `gnn` is the strongest baseline in this run.
- If the focus is relational modeling under the full 10-currency universe, `foundation_relational` is the best-performing graph-based model here.
- `oursmain` does not appear to scale as well to the broader 10-currency setting under this particular configuration and seed set.
