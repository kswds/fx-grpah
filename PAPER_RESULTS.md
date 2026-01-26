# FXStrengthGNN: Heterogeneous Macro-to-Currency Transmission for FX Prediction

## Paper Results Summary

---

## 1. Model Overview

### 1.1 Core Idea
- **Heterogeneous A Matrix**: 각 통화가 매크로 팩터에 다르게 반응
- `A ∈ R^{N_ccy × M_macro}`: 통화별 매크로 민감도 학습

### 1.2 Architecture
```
Input → GRU (temporal) → GNN (currency spillover) → Hetero A (macro transmission) → Prediction
```

### 1.3 Key Innovation
- 기존: 모든 통화에 동일한 매크로 영향 가정
- 제안: 통화별로 다른 매크로 민감도 학습 (USD는 US10Y에 민감, AUD는 Copper에 민감 등)

---

## 2. Data

| 항목 | 값 |
|------|-----|
| 기간 | 2005-01-01 ~ 2025-12-23 (21년) |
| Trading Days | 5,544일 |
| 통화 | 10개 (USD, EUR, JPY, GBP, CAD, AUD, CHF, NZD, SEK, NOK) |
| 매크로 팩터 | 7개 (Gold, VIX, Oil, US10Y, Copper, SP500, US2Y) |
| Train/Test Split | 80/20 (temporal) |
| Test 기간 | ~4.4년 (1,104일) |

**파일**: `factor_final_daily.csv`

---

## 3. Baseline Comparison

### Results

| Model | RMSE | MAE | Hit Rate |
|-------|------|-----|----------|
| LSTM | 0.8012 | 0.5842 | 61.23% |
| Transformer | 0.7891 | 0.5756 | 62.45% |
| Informer | 0.7834 | 0.5698 | 63.12% |
| TFT | 0.7756 | 0.5623 | 63.89% |
| **Ours (Full)** | **0.7469** | **0.5448** | **65.45%** |

### 실행 방법
```bash
python3 main.py  # baseline 포함 비교
```

---

## 4. Ablation Study

### Results

| Model | RMSE | MAE | Hit Rate | 의미 |
|-------|------|-----|----------|------|
| Full | 0.7469 | 0.5448 | 65.45% | 전체 모델 |
| No GNN | 0.7443 | 0.5424 | 65.28% | GNN 제거 (영향 미미) |
| Homo A | 0.8022 | 0.5819 | 60.94% | 동질적 A (성능 급락) |
| No Macro | 0.7996 | 0.5807 | 61.38% | 매크로 제거 (성능 급락) |

### Key Finding
- **Heterogeneous A가 핵심 컴포넌트** (제거시 Hit Rate -4.5%p)
- GNN (currency spillover)은 영향 미미

### 실행 방법
```bash
python3 train.py  # WITH_MACRO / WITHOUT_MACRO 비교
# 결과: ablation_results.txt
```

---

## 5. Experiment Results

### Exp5: Confidence Analysis

**질문**: 모델이 확신할 때 더 정확한가?

| Quartile | Coverage | Hit Rate |
|----------|----------|----------|
| All | 100% | 65.4% |
| Q3 | 50% | 73.6% |
| Q4 (Top 25%) | 25% | **79.1%** |

- **통계적 유의성**: z = 14.56, p < 0.0001
- **5개 seed 모두 robust**
- 75% accuracy @ 40% coverage

### 실행 방법
```bash
python3 exp5_confidence_analysis/run.py
# 결과: exp5_confidence_analysis/results.json
# 시각화: exp5_confidence_analysis/confidence_analysis.png
```

---

### Exp6: Trading Backtest

**질문**: 실제 트레이딩에서 수익이 나는가?

#### Without Transaction Costs

| Strategy | Sharpe | Win Rate | Total Return |
|----------|--------|----------|--------------|
| All trades | 9.0 | 65.4% | 19.9x |
| p50 selective | 10.4 | 73.6% | 38.5x |
| p75 selective | **12.6** | **79.1%** | 26.9x |

#### With Transaction Costs (10bp Institutional)

| Strategy | Net Return | Sharpe | Max Drawdown |
|----------|------------|--------|--------------|
| All | 1959.8% | 8.97 | -2.0% |
| p50 | 3834.5% | 10.38 | -2.9% |
| p75 | 2669.8% | **12.61** | -2.6% |

- **Break-even cost**: 200 bps (매우 robust)
- Retail (20bp)에서도 Sharpe > 12

### 실행 방법
```bash
python3 exp6_trading_backtest/run.py           # 기본 백테스트
python3 exp6_trading_backtest/run_with_costs.py  # 거래비용 포함
# 결과: exp6_trading_backtest/results.json
# 시각화: exp6_trading_backtest/trading_backtest.png
```

---

### Exp7: Big Move Analysis

**질문**: 큰 움직임을 잘 맞추는가? (중요한 거래 기회)

#### Accuracy by Actual Move Size

| Quintile | Hit Rate | Return 기여도 |
|----------|----------|---------------|
| Q1 (Smallest) | 53.3% | 0.4% |
| Q2 | 59.5% | 3.6% |
| Q3 | 65.1% | 10.4% |
| Q4 | 71.5% | 24.4% |
| Q5 (Largest) | **77.8%** | **61.3%** |

#### Key Findings

| 분석 | 결과 |
|------|------|
| Extreme moves (Top 10%) | 79.9% accuracy |
| Normal moves (Bottom 90%) | 63.8% accuracy |
| Statistical significance | z = 10.10, p < 0.0001 |
| High conviction + Big move | **90.0% accuracy** |

### 실행 방법
```bash
python3 exp7_big_move_analysis/run.py
# 결과: exp7_big_move_analysis/results.json
# 시각화: exp7_big_move_analysis/big_move_analysis.png
```

---

## 6. Key Figures

### Figure 1: Cumulative Returns
![Cumulative Returns](cumulative_returns.png)

- Test period: ~4.4 years
- All trades: 20.8x return (Sharpe 9.0)
- Selective p75: 28.2x return (Sharpe 12.6)

### Figure 2: Confidence Analysis
![Confidence Analysis](exp5_confidence_analysis/confidence_analysis.png)

### Figure 3: Trading with Costs
![Trading with Costs](exp6_trading_backtest/trading_with_costs.png)

### Figure 4: Big Move Analysis
![Big Move Analysis](exp7_big_move_analysis/big_move_analysis.png)

---

## 7. Paper Contributions

### Contribution 1: Heterogeneous Macro Transmission
- 통화별로 다른 매크로 민감도 학습
- Ablation에서 가장 중요한 컴포넌트 입증

### Contribution 2: Confidence-Accuracy Correlation
- |prediction| 크기와 정확도 상관관계 발견
- Top 25% 확신 예측 → 79% 정확도

### Contribution 3: Big Move Prediction
- 큰 움직임일수록 더 정확 (Q5: 77.8% vs Q1: 53.3%)
- High conviction + Big move = 90% accuracy

### Contribution 4: Practical Trading Value
- Sharpe 12.6 with selective trading
- 거래비용 (10-50bp) 포함해도 profitable
- Break-even: 200 bps

---

## 8. Comparison with Yoonsik's Model (arxiv 2508.14784)

### Model Differences

| Aspect | Ours (FXStrengthGNN) | Yoonsik (EdgeGNN) |
|--------|---------------------|-------------------|
| Prediction Level | Node (currency strength) | Edge (FX rate) |
| Macro Features | 7 factors | IR only |
| Key Component | Heterogeneous A matrix | Edge message passing |

### Results

| Model | Hit Rate |
|-------|----------|
| Yoonsik-style (IR only) | 49.3% |
| Ours (no macro) | 60.9% |
| **Ours (full)** | **65.5%** |

**Key Insight**:
- Node-level prediction > Edge-level: +11.6%p
- Multiple macro factors > IR only: +4.6%p
- Total improvement: **+32.9%** over Yoonsik-style

### 실행 방법
```bash
python3 exp8_yoonsik_comparison/run_fast.py
# 결과: exp8_yoonsik_comparison/results.json
```

---

### Exp9: Interpretability Analysis

**질문**: 모델의 학습된 관계가 해석 가능한가?

#### 1. Currency-Macro Sensitivity (A Matrix)

| Currency | Top Macro Factor | Sensitivity |
|----------|-----------------|-------------|
| EUR | Oil | +0.236 |
| NOK | US10Y | -0.234 |
| CHF | VIX | -0.233 |
| GBP | US2Y | +0.222 |
| JPY | SP500 | -0.186 |

**해석**:
- EUR: 유가 상승시 강세 (에너지 수입 영향)
- NOK: 금리 상승시 약세 (노르웨이 원유 의존)
- CHF: VIX 상승시 약세 (안전자산 역설적 패턴)
- GBP: US 단기금리에 동조
- JPY: 주식시장 역상관 (캐리 트레이드)

#### 2. Macro Factor Importance (Ablation)

| Macro | Ablated Hit Rate | Drop | Importance |
|-------|-----------------|------|------------|
| SP500 | 61.90% | 3.52%p | 5.4% |
| US2Y | 62.29% | 3.13%p | 4.8% |
| Gold | 63.57% | 1.85%p | 2.8% |
| US10Y | 64.47% | 0.95%p | 1.4% |
| Oil | 64.85% | 0.57%p | 0.9% |

**Key Finding**: SP500과 US2Y가 가장 중요한 매크로 팩터

#### 3. Temporal Robustness

| Year | Hit Rate | Days |
|------|----------|------|
| 2021 | 67.5% | 54 |
| 2022 | 68.0% | 263 |
| 2023 | 66.5% | 264 |
| 2024 | 62.7% | 265 |
| 2025 | 64.0% | 258 |

- **모든 연도에서 50% 이상** (random baseline 상회)
- Best: 2022 (68.0%), Worst: 2024 (62.7%)

### 실행 방법
```bash
python3 exp9_interpretability/run.py
# 결과: exp9_interpretability/results.json
# 시각화: exp9_interpretability/A_matrix_heatmap.png
```

---

### Exp11: Dynamic A Matrix (Novelty Contribution)

**핵심 아이디어**: A 행렬이 시장 regime에 따라 동적으로 변화

**수식**:
```
A(t) = A_base + σ(VIX_t) × A_delta
```

#### 성능 비교

| Model | Hit Rate | RMSE |
|-------|----------|------|
| Static A (Baseline) | **65.3%** | 0.744 |
| Dynamic A (Proposed) | 65.0% | 0.740 |

**Note**: 성능 개선은 없으나, interpretability 기여

#### 학습된 Regime-Dependent 민감도

| Currency | Macro | Risk-ON | Risk-OFF | Change | 해석 |
|----------|-------|---------|----------|--------|------|
| CAD | US2Y | -0.10 | -0.26 | ↓0.16 | 금리 민감도 증가 |
| AUD | Gold | -0.15 | -0.29 | ↓0.14 | 금 역상관 강화 |
| CAD | Oil | -0.06 | +0.07 | ↑0.13 | 유가 관계 반전 |
| USD | Gold | +0.03 | +0.15 | ↑0.13 | Safe haven 강화 |

**경제적 해석**:
- Risk-OFF 시 CAD의 금리(US2Y) 민감도 증가 → 금리 차별화 강화
- Risk-OFF 시 CAD-Oil 관계 반전 → 원자재 통화 특성 변화
- Risk-OFF 시 USD-Gold 관계 강화 → 안전자산 동조 현상

**Novelty Claim**:
1. **Architecture**: 시간에 따라 변화하는 A matrix
2. **Interpretability**: 경제적으로 의미있는 regime shift 패턴
3. **Flexibility**: 기존 heterogeneous A의 자연스러운 확장

### 실행 방법
```bash
python3 exp11_dynamic_A/run_proper.py
# 결과: exp11_dynamic_A/results.json
# 시각화: exp11_dynamic_A/A_matrix_regimes.png
```

---

### Exp14: Statistical Significance of A Matrix

**질문**: A matrix의 학습된 관계가 통계적으로 유의한가?

**방법**: Bootstrap (10 seeds로 반복 학습)
- 각 run에서 A matrix 수집
- 평균, 표준편차, t-statistic, p-value 계산

#### 모델 안정성

| Metric | Value |
|--------|-------|
| Mean Hit Rate | 65.4% |
| Std Hit Rate | ± 0.2% |
| Bootstrap Runs | 10 |

#### 통계적으로 유의한 관계

| Currency | Macro | 계수 (mean±std) | p-value | 유의수준 |
|----------|-------|----------------|---------|----------|
| **USD** | **Oil** | +0.102 ± 0.087 | **0.005** | *** |
| **AUD** | **US10Y** | +0.070 ± 0.066 | **0.009** | *** |
| EUR | Oil | +0.066 ± 0.073 | 0.019 | ** |
| NOK | Gold | -0.045 ± 0.056 | 0.034 | ** |
| CHF | SP500 | +0.056 ± 0.080 | 0.054 | * |

#### Summary

| 유의수준 | 개수 | 비율 |
|----------|------|------|
| p < 0.01 (***) | 2 | 2.9% |
| p < 0.05 (**) | 4 | 5.7% |
| p < 0.10 (*) | 5 | 7.1% |

**핵심 발견**:
1. **70개 관계 중 5개만 유의** (7.1%) - 대부분은 noise
2. **USD ← Oil 가장 강함** (p=0.005): Petrodollar 효과
3. **AUD ← US10Y** (p=0.009): Carry trade 동조

### 실행 방법
```bash
python3 exp14_significance/run.py
# 결과: exp14_significance/results.json
# 시각화: exp14_significance/significance_heatmap.png
```

---

## 9. All Experiment Files

| 실험 | 파일 | 결과 |
|------|------|------|
| Main training | `main.py` | 기본 결과 |
| Enhanced training | `train.py` | `enhanced_result.txt` |
| Ablation | `train.py` | `ablation_results.txt` |
| Confidence | `exp5_confidence_analysis/run.py` | `exp5_confidence_analysis/` |
| Trading | `exp6_trading_backtest/run.py` | `exp6_trading_backtest/` |
| Trading + Costs | `exp6_trading_backtest/run_with_costs.py` | `exp6_trading_backtest/` |
| Big Move | `exp7_big_move_analysis/run.py` | `exp7_big_move_analysis/` |
| A Matrix Analysis | `exp8_a_matrix_analysis/run.py` | `exp8_a_matrix_analysis/` |
| Yoonsik Comparison | `exp8_yoonsik_comparison/run_fast.py` | `exp8_yoonsik_comparison/` |
| Spillover Role | `exp9_spillover_role/run.py` | `exp9_spillover_role/` |
| Interpretability | `exp9_interpretability/run.py` | `exp9_interpretability/` |
| Robustness | `exp10_robustness/run.py` | `exp10_robustness/` |
| Dynamic A | `exp11_dynamic_A/run_proper.py` | `exp11_dynamic_A/` |
| Attention over A | `exp12_attention_over_a/run.py` | `exp12_attention_over_a/` |
| Error Pattern | `exp13_error_pattern/run.py` | `exp13_error_pattern/` |
| Significance | `exp14_significance/run.py` | `exp14_significance/` |

---

## 10. Quick Run All Experiments

```bash
# 1. Main training with baselines
python3 main.py

# 2. Enhanced training (ablation)
python3 train.py

# 3. Confidence analysis
python3 exp5_confidence_analysis/run.py

# 4. Trading backtest
python3 exp6_trading_backtest/run.py
python3 exp6_trading_backtest/run_with_costs.py

# 5. Big move analysis
python3 exp7_big_move_analysis/run.py

# 6. Yoonsik comparison
python3 exp8_yoonsik_comparison/run_fast.py

# 7. Interpretability analysis
python3 exp9_interpretability/run.py

# 8. Dynamic A (novelty)
python3 exp11_dynamic_A/run_proper.py

# 9. Statistical significance
python3 exp14_significance/run.py
```

---

## 11. Summary Table for Paper

| Metric | Value |
|--------|-------|
| Overall Hit Rate | 65.4% |
| Top 25% Confidence Hit Rate | 79.1% |
| Big Move (Q5) Hit Rate | 77.8% |
| High Conviction + Big Move | 90.0% |
| Trading Sharpe (all) | 9.0 |
| Trading Sharpe (selective) | 12.6 |
| Break-even Cost | 200 bps |
| Max Drawdown | -2.6% |
| Test Period Return | 20-28x |
| Most Important Macro | SP500 (5.4%) |
| Yearly Hit Rate Range | 62.7% - 68.0% |
| Significant A Coefficients (p<0.05) | 4/70 (5.7%) |
| Strongest: USD←Oil | +0.102 (p=0.005) |

---

## 12. Recommended Venues

| Venue | 적합도 | 이유 |
|-------|--------|------|
| **ICAIF** | ★★★★☆ | Finance + AI 특화, 실용적 결과 중시 |
| **KDD Applied** | ★★★☆☆ | 실용적 가치 있으나 novelty 보완 필요 |
| Finance Journal | ★★★★☆ | 실증 분석 강점 |
| KDD Main | ★★☆☆☆ | 수학적 novelty 부족 |

---

## 13. Additional Experiments (추가 예정)

### Exp8: A Matrix Deep Analysis

**질문**: A matrix의 구조적 특성과 경제적 해석

#### Column Sparsity (통화별 매크로 의존도)

| Currency | Active Factors | Interpretation |
|----------|---------------|----------------|
| CAD | Oil, US2Y | 원유 수출국 |
| NOK | Oil, US10Y | 원유 수출국 |
| AUD | Copper, Gold | 광물 수출국 |
| CHF | VIX, Gold | 안전자산 |
| JPY | SP500, VIX | 캐리 트레이드 |

#### Currency Clustering

모델이 학습한 통화 그룹:
- **Commodity**: CAD, NOK, AUD
- **Safe Haven**: CHF, JPY
- **Europe**: EUR, GBP, SEK

#### VIX Regime Sensitivity Shift

| Currency | Low VIX | High VIX | Shift |
|----------|---------|----------|-------|
| AUD | +0.15 | -0.29 | ↓ Risk-off 약세 |
| USD | +0.03 | +0.15 | ↑ Safe haven |
| CAD | -0.06 | +0.07 | 반전 |

### 실행 방법
```bash
python3 exp8_a_matrix_analysis/run.py
```

---

### Exp9: Spillover (GNN) Role Justification

**질문**: GNN이 예측 정확도가 아닌 다른 가치가 있는가?

#### GNN의 역할 재정의

| Metric | With GNN | No GNN | Interpretation |
|--------|----------|--------|----------------|
| Hit Rate | 65.3% | 65.2% | 성능 차이 미미 |
| Triangle Error | 3.2e-9 | 4.1e-9 | **GNN이 22% 감소** |
| Shock Coherence | 0.82 | 0.71 | **GNN이 더 일관적** |

**Key Insight**: GNN은 "prediction accuracy"가 아닌 "macro transmission stability" 제공
- 통화 간 일관성 유지 (s_i - s_j + s_j - s_k = s_i - s_k)
- 매크로 충격의 coherent propagation

### 실행 방법
```bash
python3 exp9_spillover_role/run.py
```

---

### Exp10: Robustness Analysis

**질문**: 다양한 시장 환경에서 robust한가?

#### Rolling Window Validation (13 folds)

| Metric | Mean | Std |
|--------|------|-----|
| Hit Rate | 61.7% | ±3.0% |
| RMSE | 0.82 | ±0.18 |

#### Random Split (5 seeds)

| Metric | Mean | Std |
|--------|------|-----|
| Hit Rate | 65.5% | ±0.5% |
| RMSE | 0.73 | ±0.01 |

#### Walk-Forward Validation

| Metric | Value |
|--------|-------|
| Hit Rate | 61.7% |
| Retrain Interval | 500 days |
| N Predictions | 3,537 |

**Key Finding**: Random split이 rolling보다 높음 → 시장 regime 변화에 민감

### 실행 방법
```bash
python3 exp10_robustness/run.py
```

---

### Exp13: Error Pattern Analysis

**질문**: 예측 오류에 체계적 패턴이 있는가?

#### 통화별 정확도

| Currency | Hit Rate | 비고 |
|----------|----------|------|
| JPY | 69.6% | Best |
| NOK | 67.5% | |
| GBP | 67.3% | |
| SEK | 66.8% | |
| CHF | 66.6% | |
| NZD | 65.9% | |
| CAD | 65.2% | |
| EUR | 63.9% | |
| **AUD** | **53.7%** | ⚠️ Worst |

#### AUD 문제 분석

| 분석 | 결과 |
|------|------|
| AUD Hit Rate | 53.7% (거의 random) |
| AUD 제외시 전체 Hit Rate | 66.6% (+1.4%p) |
| Triangle 오류 원인 비율 | 21.6% (expected 11.1%) |

**Root Cause**: 현재 macro feature set의 한계
- AUD 핵심 드라이버: **Iron Ore** (호주 수출 30%)
- 현재 모델: Copper, Gold만 포함 → AUD 설명력 부족
- 중국 경제 지표 없음 (호주 수출 35%가 중국행)

#### 통화별 주요 드라이버와 모델 커버리지

| 통화 | 유형 | 주요 드라이버 | 현재 모델 |
|-----|------|--------------|----------|
| **AUD** | Commodity | Iron Ore (80%), Gold | ⚠️ Gold만 |
| **CAD** | Commodity | Crude Oil (WTI) | ✅ Oil 포함 |
| **NOK** | Commodity | Brent Oil, Gas | ✅ Oil 포함 |
| **NZD** | Commodity | Dairy (GDT Index) | ⚠️ 없음 |
| **EUR** | Non-Commodity | ECB 정책, 산업생산 | ✅ 금리차 |
| **GBP** | Non-Commodity | BoE 정책, Fiscal | ✅ 금리차 |
| **JPY** | Safe Haven | BoJ 정책, Risk-off | ✅ VIX |
| **CHF** | Safe Haven | Risk-off, Gold | ✅ VIX, Gold |
| **SEK** | Non-Commodity | EUR 연동, 금리차 | ✅ 금리차 |

#### Limitation & Future Work

```
Limitation:
- AUD shows lower accuracy (54%) compared to other currencies (65-70%)
- This is expected as AUD is primarily driven by Iron Ore prices,
  which are not included in our macro feature set

Future Work:
- Add Iron Ore and China-related factors (Shanghai Composite, PMI)
- Consider currency-specific feature selection
```

### 실행 방법
```bash
python3 exp13_error_pattern/run.py
python3 exp13_error_pattern/triangle_analysis.py
python3 exp13_error_pattern/verify_aud_lag.py
```

---

*Generated: 2026-01-26*
