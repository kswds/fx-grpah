"""
AUD Lag 원인 검증
- 데이터 처리 문제인지 vs 실제 시장 타이밍 문제인지
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from config import Config
from dataset import load_data


def verify_data_alignment():
    """데이터 정렬 확인"""
    config = Config()
    df = load_data(config)

    print("=" * 60)
    print("1. 데이터 정렬 확인")
    print("=" * 60)

    # 날짜 확인
    print(f"\n데이터 기간: {df['Date'].min()} ~ {df['Date'].max()}")
    print(f"총 일수: {len(df)}")

    # 결측치 확인
    print("\n결측치:")
    for col in ['AUD_FX', 'Global_Copper', 'Global_Gold']:
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"  {col}: {missing} ({missing/len(df)*100:.2f}%)")

    return df


def analyze_aud_copper_relationship(df):
    """AUD와 Copper 관계 분석"""
    print("\n" + "=" * 60)
    print("2. AUD-Copper 상관관계 (다양한 lag)")
    print("=" * 60)

    # 수익률 계산
    aud_ret = np.log(df['AUD_FX']).diff()
    copper_ret = np.log(df['Global_Copper']).diff()

    correlations = {}
    for lag in range(-5, 6):
        if lag == 0:
            corr = aud_ret.corr(copper_ret)
        elif lag > 0:
            # Copper가 AUD보다 lag일 앞서는 경우 (Copper leads)
            corr = aud_ret.iloc[lag:].corr(copper_ret.iloc[:-lag])
        else:
            # AUD가 Copper보다 앞서는 경우
            corr = aud_ret.iloc[:lag].corr(copper_ret.iloc[-lag:])
        correlations[lag] = corr

    print("\nlag | Corr(AUD_ret, Copper_ret)")
    print("-" * 35)
    for lag, corr in sorted(correlations.items()):
        marker = " *** BEST" if corr == max(correlations.values()) else ""
        print(f" {lag:+2d} | {corr:+.4f}{marker}")

    best_lag = max(correlations, key=correlations.get)
    print(f"\n→ Best lag: {best_lag}")
    if best_lag == 0:
        print("  데이터 정렬이 정상으로 보임")
    elif best_lag > 0:
        print(f"  Copper가 AUD보다 {best_lag}일 선행 (Copper leads AUD)")
    else:
        print(f"  AUD가 Copper보다 {-best_lag}일 선행 (AUD leads Copper)")

    return correlations, aud_ret, copper_ret


def check_specific_dates(df, aud_ret, copper_ret):
    """특정 날짜 케이스 분석"""
    print("\n" + "=" * 60)
    print("3. 특정 날짜 케이스 분석")
    print("=" * 60)

    # 큰 Copper 변동이 있었던 날
    copper_big_moves = copper_ret.abs() > copper_ret.std() * 2
    big_move_dates = df.loc[copper_big_moves, 'Date'].values

    print(f"\nCopper 큰 변동일 (|ret| > 2σ): {copper_big_moves.sum()}일")

    # 몇 개 샘플 분석
    print("\n샘플 분석 (Copper 큰 변동일):")
    print("-" * 70)
    print(f"{'Date':12s} | {'Copper_t':>10s} | {'AUD_t':>10s} | {'AUD_t+1':>10s} | {'Same sign?':>10s}")
    print("-" * 70)

    count_same_day = 0
    count_next_day = 0

    for i, date in enumerate(big_move_dates[:20]):  # 처음 20개
        idx = df[df['Date'] == date].index[0]
        if idx + 1 >= len(df):
            continue

        copper_t = copper_ret.iloc[idx]
        aud_t = aud_ret.iloc[idx]
        aud_t1 = aud_ret.iloc[idx + 1] if idx + 1 < len(aud_ret) else np.nan

        same_day = "✓" if np.sign(copper_t) == np.sign(aud_t) else "✗"
        next_day = "✓" if np.sign(copper_t) == np.sign(aud_t1) else "✗"

        print(f"{str(date)[:10]:12s} | {copper_t:+10.4f} | {aud_t:+10.4f} | {aud_t1:+10.4f} | {same_day} (t), {next_day} (t+1)")

        if np.sign(copper_t) == np.sign(aud_t):
            count_same_day += 1
        if np.sign(copper_t) == np.sign(aud_t1):
            count_next_day += 1

    total = min(20, len(big_move_dates))
    print(f"\nCopper과 같은 방향:")
    print(f"  당일 (t):   {count_same_day}/{total} = {count_same_day/total:.1%}")
    print(f"  다음날 (t+1): {count_next_day}/{total} = {count_next_day/total:.1%}")


def analyze_prediction_timing(df, aud_ret, copper_ret):
    """예측 타이밍 분석 - 모델이 실제로 어떻게 학습했을지"""
    print("\n" + "=" * 60)
    print("4. 모델 관점에서의 타이밍 분석")
    print("=" * 60)

    # 모델은 t-L:t 데이터로 t+1을 예측
    # 만약 Copper_t로 AUD_t+1을 예측한다면, 상관관계가 높아야 함

    # Copper_t vs AUD_t+1
    corr_predict = copper_ret.iloc[:-1].corr(aud_ret.iloc[1:])
    print(f"\nCorr(Copper_t, AUD_t+1) = {corr_predict:.4f}")
    print("  → 이게 높으면: Copper 당일 데이터로 AUD 다음날 예측 가능")

    # Copper_t vs AUD_t
    corr_same = copper_ret.corr(aud_ret)
    print(f"Corr(Copper_t, AUD_t) = {corr_same:.4f}")
    print("  → 이게 높으면: 동시 반응")

    # Copper_t-1 vs AUD_t
    corr_lag = copper_ret.iloc[:-1].corr(aud_ret.iloc[1:])
    print(f"Corr(Copper_t-1, AUD_t) = {corr_lag:.4f}")
    print("  → 이게 높으면: Copper가 하루 뒤 AUD에 영향")

    print("\n해석:")
    if abs(corr_same) > abs(corr_predict):
        print("  → AUD는 Copper에 당일 반응 (데이터 정렬 정상)")
        print("  → 모델의 lag-1 이슈는 다른 원인")
    else:
        print("  → Copper_t가 AUD_t+1에 더 영향 (실제 시장 딜레이)")


def plot_lag_analysis(correlations, save_path):
    """Lag 상관관계 시각화"""
    fig, ax = plt.subplots(figsize=(10, 6))

    lags = sorted(correlations.keys())
    corrs = [correlations[l] for l in lags]

    bars = ax.bar(lags, corrs, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 최대값 강조
    max_lag = max(correlations, key=correlations.get)
    max_corr = correlations[max_lag]
    ax.bar(max_lag, max_corr, color='red', alpha=0.7)

    ax.set_xlabel('Lag (days)')
    ax.set_ylabel('Correlation')
    ax.set_title('AUD Return vs Copper Return at Different Lags', fontweight='bold')
    ax.set_xticks(lags)

    # 주석
    ax.annotate(f'Best: lag={max_lag}\ncorr={max_corr:.3f}',
                xy=(max_lag, max_corr),
                xytext=(max_lag + 1, max_corr + 0.05),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved: {save_path}")


def main():
    print("=" * 60)
    print("AUD Lag 원인 검증")
    print("=" * 60)

    # 1. 데이터 로드 및 정렬 확인
    df = verify_data_alignment()

    # 2. AUD-Copper 상관관계 분석
    correlations, aud_ret, copper_ret = analyze_aud_copper_relationship(df)

    # 3. 특정 날짜 케이스 분석
    check_specific_dates(df, aud_ret, copper_ret)

    # 4. 모델 관점 분석
    analyze_prediction_timing(df, aud_ret, copper_ret)

    # 5. 시각화
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    plot_lag_analysis(correlations, os.path.join(exp_dir, "aud_copper_lag.png"))

    # 결론
    print("\n" + "=" * 60)
    print("결론")
    print("=" * 60)

    best_lag = max(correlations, key=correlations.get)
    if best_lag == 0:
        print("""
✓ 데이터 정렬 문제 없음
  - AUD와 Copper는 당일 상관관계가 가장 높음
  - 모델의 lag-1 이슈는 데이터 버그가 아님

→ 가능한 원인:
  1. 모델이 다른 feature 조합에서 타이밍 오류
  2. lookback window 내 패턴 학습 문제
  3. 다른 macro factor와의 간섭
        """)
    else:
        print(f"""
⚠ 실제 시장 타이밍 이슈 발견
  - AUD-Copper 상관관계는 lag={best_lag}에서 최대
  - 이는 실제 시장 반응 딜레이를 반영

→ 해결책:
  1. AUD에 대해 Copper를 lag={-best_lag}로 사용
  2. 또는 모델이 이 패턴을 학습하도록 lookback 조정
        """)


if __name__ == "__main__":
    main()
