# Exp2: Graph Structure Analysis

## Motivation

Ablation study에서 GNN이 성능에 거의 기여하지 않음:
- No GNN: RMSE 0.7443 (오히려 더 좋음)
- With GNN: RMSE 0.7469

**가설**: Fully-connected graph가 문제일 수 있음
- 모든 통화가 연결 → 사실상 global averaging
- 의미 없는 edge가 noise 추가

## Experiment: Different Graph Structures

통화 목록: USD, EUR, JPY, GBP, CAD, AUD, CHF, NZD, SEK, NOK

### Graph Types

1. **no_graph**: 엣지 없음 (baseline, GNN 비활성화)

2. **full**: 완전 연결 그래프 (현재 기본값)

3. **regional**: 지역 기반 클러스터
   - Americas: USD ↔ CAD
   - Europe: EUR ↔ GBP ↔ CHF ↔ SEK ↔ NOK
   - Asia-Pacific: JPY ↔ AUD ↔ NZD

4. **commodity**: 원자재 통화 연결
   - AUD ↔ CAD ↔ NOK ↔ NZD (원자재 수출국)

5. **safe_haven**: 안전자산 통화 연결
   - USD ↔ JPY ↔ CHF

6. **major_pairs**: 주요 통화쌍 기반
   - USD와 G7 통화 연결

## Expected Insights

- Sparse meaningful graph > Full graph?
- 어떤 경제적 관계가 FX 예측에 유용한가?
- GNN이 정말 필요 없는가, 아니면 graph 설계가 문제인가?
