# Main Predictive Results

This report reproduces the headline predictive comparison used in the anonymous GitHub release.

Evaluation setup:
- All models use the same shared training objective for this comparison: the active directional core on non-USD targets only.
- Active set definition: `A = {(b, i) : |y_{b,i}| >= tau}`.
- Threshold definition: `tau = Q0.40(|y_train_norm|)`.
- Core loss on the active set: `mean softplus(- rhat_{b,i} * sign(y_{b,i}))`.
- Seed set: `21, 42, 456`. Reported numbers are mean ± standard deviation across these seeds.
- `Extreme Hit` is the sign hit ratio computed on the pooled train-based extreme subset, where `|y|` exceeds the training `Q0.80` threshold over non-USD targets.
- `Stress Mean Hit Avg.` is the simple average of sign hit ratios across the three stress scenarios `{VIX stress, FX stress, Yield stress}`.
- `Stress Extreme Hit Avg. @ Q80` is computed inside each stress scenario using that scenario's realized `Q0.80` absolute-return subset, and then averaged across the three scenarios.
- Stress scenarios are defined from training-sample percentile rules: `VIX stress = train-rank(|VIX change|) >= 0.85`, `FX stress = train-rank(cross-currency mean absolute FX return) >= 0.85`, and `Yield stress = train-rank(US10Y change) >= 0.85`.
- `Corr-LSTM-GAT` and `FXRP` are included as reference baselines.

## Headline Predictive Metrics

| Model | Mean Hit ↑ | Extreme Hit ↑ | Stress Mean Hit Avg. ↑ | Stress Extreme Hit Avg. @ Q80 ↑ | RMSE ×10³ ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| **ARC_FX** | **0.5159 ± 0.0028** | **0.5262 ± 0.0148** | **0.5273 ± 0.0033** | **0.5440 ± 0.0192** | **6.213 ± 0.013** |
| MLP | 0.5061 ± 0.0096 | 0.5155 ± 0.0120 | 0.5069 ± 0.0124 | 0.5186 ± 0.0095 | 6.182 ± 0.003 |
| Transformer | 0.5036 ± 0.0075 | 0.4923 ± 0.0254 | 0.5057 ± 0.0080 | 0.4951 ± 0.0170 | 6.220 ± 0.036 |
| GNN | 0.4958 ± 0.0021 | 0.4885 ± 0.0211 | 0.4935 ± 0.0109 | 0.4894 ± 0.0151 | 6.194 ± 0.003 |
| Corr-LSTM-GAT | 0.4980 ± 0.0052 | 0.4959 ± 0.0123 | 0.5043 ± 0.0018 | 0.5004 ± 0.0176 | 6.183 ± 0.001 |
| FXRP | 0.4742 ± 0.0019 | 0.4869 ± 0.0025 | 0.4762 ± 0.0021 | 0.4991 ± 0.0046 | 6.195 ± 0.008 |

## Non-Trivial Directional Hit And F1

The neutral-band thresholds below are basis-point bands (`±1bp`, `±3bp`, `±5bp`, `±10bp`).

- `Non-trivial hit`: excludes only the observations where both the prediction and the realized return fall inside the same neutral band, then measures sign hit on the remaining subset.
- `Macro-F1`: computed on the same subset using ternary labels `{down, flat, up}` induced by the threshold.

### Non-Trivial Directional Hit

| Model | ±1bp | ±3bp | ±5bp | ±10bp |
| --- | ---: | ---: | ---: | ---: |
| ARC_FX | 0.5171 ± 0.0026 | 0.5203 ± 0.0032 | 0.5233 ± 0.0030 | 0.5275 ± 0.0027 |
| MLP | 0.5086 ± 0.0089 | 0.5131 ± 0.0093 | 0.5175 ± 0.0102 | 0.5203 ± 0.0094 |
| Transformer | 0.5053 ± 0.0069 | 0.5083 ± 0.0067 | 0.5102 ± 0.0073 | 0.5127 ± 0.0093 |
| GNN | 0.4980 ± 0.0020 | 0.5020 ± 0.0022 | 0.5040 ± 0.0027 | 0.5050 ± 0.0054 |
| Corr-LSTM-GAT | 0.5017 ± 0.0042 | 0.5068 ± 0.0041 | 0.5098 ± 0.0048 | 0.5087 ± 0.0068 |
| FXRP | 0.4759 ± 0.0031 | 0.4786 ± 0.0050 | 0.4809 ± 0.0034 | 0.4825 ± 0.0024 |

### Macro-F1

| Model | ±1bp | ±3bp | ±5bp | ±10bp |
| --- | ---: | ---: | ---: | ---: |
| ARC_FX | 0.3160 ± 0.0058 | 0.2646 ± 0.0088 | 0.2112 ± 0.0159 | 0.1068 ± 0.0423 |
| MLP | 0.2755 ± 0.0124 | 0.1908 ± 0.0211 | 0.1216 ± 0.0330 | 0.0335 ± 0.0310 |
| Transformer | 0.2790 ± 0.0075 | 0.2095 ± 0.0310 | 0.1530 ± 0.0509 | 0.0745 ± 0.0631 |
| GNN | 0.2862 ± 0.0037 | 0.1920 ± 0.0205 | 0.1213 ± 0.0296 | 0.0329 ± 0.0191 |
| Corr-LSTM-GAT | 0.2307 ± 0.0165 | 0.1118 ± 0.0604 | 0.0349 ± 0.0395 | 0.0000 ± 0.0001 |
| FXRP | 0.1994 ± 0.0098 | 0.1233 ± 0.0559 | 0.0449 ± 0.0481 | 0.0000 ± 0.0000 |
