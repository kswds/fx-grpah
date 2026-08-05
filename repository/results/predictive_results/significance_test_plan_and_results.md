# Statistical Testing Plan and Results for Table 1

This note summarizes the hypothesis-testing setup aligned with the paper's main claims and reports the resulting bootstrap-based inference.

## Bootstrap Design

Both tests use the same paired moving-block bootstrap:

- Bootstrap unit: test dates
- Test period dates: `1093`
- Cross-section kept within each sampled date: all `10` non-USD currencies
- Seeds kept within each sampled date block: `21, 42, 456`
- Block length: `10` trading days
- Repetitions: `10,000`
- In each bootstrap draw, the same sampled date blocks are applied jointly to both competing models
- Metrics are computed separately by seed and then averaged across seeds before taking the model difference

---

## 1. Mean Hit: One-Sided Superiority Test

Comparison:

- `ARC-FX` vs `MLP`

Test statistic:

- `ΔHit = Hit_ARC-FX - Hit_MLP`

Hypotheses:

- `H0: ΔHit <= 0`
- `H1: ΔHit > 0`

This is the appropriate test when the ex-ante research claim is directional superiority of `ARC-FX` over the strongest direct baseline in Mean Hit.

### Result

- Observed difference: `ΔHit = 0.009820`
- Two-sided 95% CI: `[-0.000366, 0.020129]`
- Two-sided p-value: `0.0615`
- One-sided 95% CI: `[0.001128, +inf)`
- One-sided p-value: `0.0306`

### Judgment

Under the one-sided superiority formulation, the Mean Hit improvement of `ARC-FX` over `MLP` is statistically significant at the 5% level.

---

## 2. One-Sided RMSE Superiority Test: MLP vs Corr-LSTM-GAT

For a direct top-two RMSE comparison, define the difference as:

- `ΔRMSE = RMSE_MLP - RMSE_Corr-LSTM-GAT`

Since lower RMSE is better, negative values favor `MLP`.

Although the rounded table values suggest only a tiny gap, the statistical test is computed from the unrounded raw prediction panels and the re-evaluated bootstrap RMSE values.

Hypotheses:

- `H0: ΔRMSE >= 0`
- `H1: ΔRMSE < 0`

This is a one-sided superiority test asking whether `MLP` has significantly lower RMSE than `Corr-LSTM-GAT`.

### Result

- Observed difference: `ΔRMSE = -0.00000142`
- Two-sided 95% CI: `[-0.00001536, 0.00001249]`
- One-sided p-value: `0.3998`

### Judgment

The observed difference has the favorable sign for `MLP`, but the one-sided p-value is far above 0.05. Therefore, the bootstrap test does not support the claim that `MLP` has significantly lower RMSE than `Corr-LSTM-GAT`.

## Reproduction

After generating prediction parquet files under `results/repro_runs/`, run:

```bash
python src/significance_test.py
```
