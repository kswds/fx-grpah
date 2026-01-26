# Exp1: Dynamic Cross-Attention for Macro-Currency Transmission

## Motivation

기존 모델의 ablation study 결과:
- **Heterogeneous A**가 가장 중요한 컴포넌트 (ΔRMSE=+0.055, ΔHit=-4.5%p)
- GNN은 거의 영향 없음

하지만 현재 A 행렬의 한계:
1. **Static**: A가 학습 후 고정됨 - 시장 상황 변화 반영 불가
2. **Linear**: 단순 행렬곱으로 macro-currency 관계 모델링

## Hypothesis

> 시장 상황(market state)에 따라 통화별 매크로 민감도가 동적으로 변해야 한다.
> 예: 위기 시 safe-haven 통화(JPY, CHF)는 risk sentiment에 더 민감해짐

## Proposed Method: Dynamic Cross-Attention A

기존:
```
m_msg = A @ macro_embed  (A는 고정 파라미터)
```

제안:
```
A_t = CrossAttention(Q=currency_embed, K=macro_embed, V=macro_embed)
m_msg = A_t @ macro_embed  (A_t는 시점/상황에 따라 동적)
```

### Architecture

1. **Currency Query**: GRU output을 query로 사용
2. **Macro Key/Value**: Macro embedding을 key/value로 사용
3. **Dynamic Attention**: 각 통화가 어떤 매크로 팩터에 주목할지 attention으로 결정
4. **Interpretability**: Attention weight가 곧 A_t가 되어 해석 가능

## Expected Outcomes

1. 성능 향상 (RMSE, Hit rate)
2. 시간에 따른 A_t 변화 시각화 가능
3. "Dynamic Heterogeneous Transmission" contribution 확립

## Files

- `models.py`: FXStrengthDynamicA 모델 구현
- `run.py`: 실험 실행 스크립트
- `results.json`: 실험 결과
