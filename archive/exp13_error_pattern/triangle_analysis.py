"""
Triangle Consistency 분석
- 어떤 통화가 삼각 관계를 깨뜨리는지
- 특정 통화 방향이 틀려서 연쇄적으로 다른 것들이 틀리는지
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from itertools import combinations

from config import Config
from models import FXStrengthGNN
from dataset import create_dataloaders, fully_connected_edge_index
from train import Trainer


def get_predictions(model, test_loader, edge_index, device):
    """Get predictions"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xl, xm, yb in test_loader:
            xl, xm = xl.to(device), xm.to(device)
            rhat, ds, z_ccy, m_msg = model(xl, xm, edge_index)
            all_preds.append(rhat.cpu().numpy())
            all_targets.append(yb.numpy())

    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)


def analyze_triangle_errors(preds, targets, config):
    """
    삼각 관계 분석:
    - r(A/USD) - r(B/USD) = r(A/B)
    - 만약 A, B 둘 다 맞았는데 cross가 틀리면? 또는 그 반대면?
    """
    ccys = config.ccys
    n_ccy = len(ccys)
    usd_idx = config.usd_idx

    # 방향 예측
    pred_sign = np.sign(preds)
    target_sign = np.sign(targets)

    # 각 통화별 정확도
    correct = (pred_sign == target_sign)

    triangle_results = []

    # 모든 삼각형 조합 (i, j, k) 분석
    non_usd = [i for i in range(n_ccy) if i != usd_idx]

    for i, j, k in combinations(non_usd, 3):
        ccy_i, ccy_j, ccy_k = ccys[i], ccys[j], ccys[k]

        # 각 시점에서 이 삼각형의 오류 패턴
        # correct_i[t] = 통화 i의 방향이 맞았는지
        correct_i = correct[:, i]
        correct_j = correct[:, j]
        correct_k = correct[:, k]

        # 삼각형 내 오류 패턴 분류
        # 0: 모두 맞음, 1: 하나만 틀림, 2: 둘 틀림, 3: 모두 틀림
        n_errors = (~correct_i).astype(int) + (~correct_j).astype(int) + (~correct_k).astype(int)

        # 하나만 틀렸을 때 어떤 통화가 틀렸는지
        single_error_cases = (n_errors == 1)
        if single_error_cases.sum() > 0:
            i_caused = (single_error_cases & ~correct_i).sum()
            j_caused = (single_error_cases & ~correct_j).sum()
            k_caused = (single_error_cases & ~correct_k).sum()
        else:
            i_caused = j_caused = k_caused = 0

        triangle_results.append({
            'triangle': (ccy_i, ccy_j, ccy_k),
            'all_correct': (n_errors == 0).sum(),
            'one_wrong': (n_errors == 1).sum(),
            'two_wrong': (n_errors == 2).sum(),
            'all_wrong': (n_errors == 3).sum(),
            'single_error_caused_by': {
                ccy_i: int(i_caused),
                ccy_j: int(j_caused),
                ccy_k: int(k_caused),
            },
            'total': len(n_errors),
        })

    return triangle_results


def find_problematic_currency(triangle_results, config):
    """
    어떤 통화가 삼각형 오류의 주범인지 찾기
    """
    ccys = [c for c in config.ccys if c != "USD"]

    # 각 통화가 single-error의 원인이 된 횟수
    single_error_counts = {ccy: 0 for ccy in ccys}
    total_single_errors = 0

    for tri in triangle_results:
        for ccy, count in tri['single_error_caused_by'].items():
            single_error_counts[ccy] += count
            total_single_errors += count

    # 비율 계산
    if total_single_errors > 0:
        single_error_rates = {
            ccy: count / total_single_errors
            for ccy, count in single_error_counts.items()
        }
    else:
        single_error_rates = {ccy: 0 for ccy in ccys}

    # 기대값 (균등 분포일 경우)
    expected_rate = 1 / len(ccys)

    # 과대/과소 대표 통화
    problematic = {}
    for ccy in ccys:
        rate = single_error_rates[ccy]
        deviation = rate - expected_rate
        problematic[ccy] = {
            'single_error_count': single_error_counts[ccy],
            'single_error_rate': rate,
            'expected_rate': expected_rate,
            'deviation': deviation,
            'is_problematic': deviation > 0.05,  # 5%p 이상 초과
        }

    return problematic


def analyze_error_propagation(preds, targets, config):
    """
    오류 전파 분석:
    - 특정 통화가 틀리면 다른 통화도 틀리는 경향이 있는지
    """
    ccys = [c for c in config.ccys if c != "USD"]
    n = len(ccys)

    pred_sign = np.sign(preds)
    target_sign = np.sign(targets)
    correct = (pred_sign == target_sign)

    # 통화 인덱스 매핑
    idx_map = {c: i for i, c in enumerate(config.ccys)}
    ccy_indices = [idx_map[c] for c in ccys]

    # 조건부 확률: P(j 틀림 | i 틀림)
    conditional_error = np.zeros((n, n))

    for i_idx, i in enumerate(ccy_indices):
        i_wrong = ~correct[:, i]
        n_i_wrong = i_wrong.sum()

        if n_i_wrong > 0:
            for j_idx, j in enumerate(ccy_indices):
                if i != j:
                    # i가 틀렸을 때 j도 틀린 비율
                    j_wrong_given_i_wrong = (~correct[:, j] & i_wrong).sum()
                    conditional_error[i_idx, j_idx] = j_wrong_given_i_wrong / n_i_wrong

    return {
        'conditional_error_matrix': conditional_error,
        'currencies': ccys,
    }


def create_triangle_analysis_plot(triangle_results, problematic, propagation, config, save_path):
    """삼각형 분석 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ccys = [c for c in config.ccys if c != "USD"]

    # 1. 통화별 single-error 원인 비율
    ax1 = axes[0, 0]
    rates = [problematic[c]['single_error_rate'] for c in ccys]
    expected = problematic[ccys[0]]['expected_rate']

    colors = ['red' if problematic[c]['is_problematic'] else 'steelblue' for c in ccys]
    bars = ax1.bar(ccys, rates, color=colors, alpha=0.7)
    ax1.axhline(y=expected, color='green', linestyle='--', label=f'Expected ({expected:.1%})')
    ax1.set_ylabel('Single-Error Cause Rate')
    ax1.set_title('Which Currency Breaks Triangles?', fontweight='bold')
    ax1.set_xticklabels(ccys, rotation=45, ha='right')
    ax1.legend()

    # 값 표시
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{rate:.1%}', ha='center', fontsize=9)

    # 2. 조건부 오류 확률 히트맵
    ax2 = axes[0, 1]
    cond_err = propagation['conditional_error_matrix']
    im = ax2.imshow(cond_err, cmap='Reds', vmin=0, vmax=0.6)
    ax2.set_xticks(range(len(ccys)))
    ax2.set_xticklabels(ccys, rotation=45, ha='right')
    ax2.set_yticks(range(len(ccys)))
    ax2.set_yticklabels(ccys)
    ax2.set_xlabel('Currency j')
    ax2.set_ylabel('Currency i')
    ax2.set_title('P(j wrong | i wrong)\nError Propagation', fontweight='bold')
    plt.colorbar(im, ax=ax2)

    # 3. 삼각형 오류 분포
    ax3 = axes[1, 0]
    # 상위 10개 worst 삼각형
    sorted_tri = sorted(triangle_results,
                        key=lambda x: x['one_wrong'] + x['two_wrong'],
                        reverse=True)[:10]

    labels = ['-'.join(t['triangle']) for t in sorted_tri]
    one_wrong = [t['one_wrong'] for t in sorted_tri]
    two_wrong = [t['two_wrong'] for t in sorted_tri]
    all_wrong = [t['all_wrong'] for t in sorted_tri]

    x = np.arange(len(labels))
    width = 0.25
    ax3.barh(x - width, one_wrong, width, label='1 wrong', alpha=0.7)
    ax3.barh(x, two_wrong, width, label='2 wrong', alpha=0.7)
    ax3.barh(x + width, all_wrong, width, label='3 wrong', alpha=0.7)
    ax3.set_yticks(x)
    ax3.set_yticklabels(labels, fontsize=8)
    ax3.set_xlabel('Count')
    ax3.set_title('Top 10 Problematic Triangles', fontweight='bold')
    ax3.legend()

    # 4. 요약
    ax4 = axes[1, 1]
    ax4.axis('off')

    # 가장 문제되는 통화
    most_problematic = max(ccys, key=lambda c: problematic[c]['deviation'])
    most_prob_rate = problematic[most_problematic]['single_error_rate']

    # 가장 덜 문제되는 통화
    least_problematic = min(ccys, key=lambda c: problematic[c]['deviation'])
    least_prob_rate = problematic[least_problematic]['single_error_rate']

    # 오류 전파가 가장 강한 쌍
    cond_err_no_diag = cond_err.copy()
    np.fill_diagonal(cond_err_no_diag, 0)
    max_prop_idx = np.unravel_index(cond_err_no_diag.argmax(), cond_err.shape)
    max_prop_pair = (ccys[max_prop_idx[0]], ccys[max_prop_idx[1]])
    max_prop_val = cond_err[max_prop_idx]

    summary = f"""
    Triangle Consistency Analysis Summary
    =====================================

    🔴 Most Problematic Currency:
       {most_problematic}: {most_prob_rate:.1%} of single-errors
       (expected: {expected:.1%})

       → {most_problematic} 방향이 틀리면 삼각형이 깨짐

    🟢 Least Problematic Currency:
       {least_problematic}: {least_prob_rate:.1%} of single-errors

    🔗 Strongest Error Propagation:
       {max_prop_pair[0]} → {max_prop_pair[1]}: {max_prop_val:.1%}

       → {max_prop_pair[0]}가 틀리면 {max_prop_pair[1]}도
         {max_prop_val:.1%} 확률로 틀림

    💡 Interpretation:
       • {most_problematic}의 예측이 개선되면
         전체 삼각형 일관성이 크게 향상될 것
       • {max_prop_pair[0]}-{max_prop_pair[1]} 페어는
         연동성이 높아 함께 틀리는 경향
    """

    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("Triangle Consistency Analysis")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config()

    # Train model
    print("\n[1/4] Training model...")
    train_loader, test_loader = create_dataloaders(config)
    edge_index = fully_connected_edge_index(config.n_ccy).to(device)

    model = FXStrengthGNN(config)
    trainer = Trainer(model, config, device)
    trainer.train(train_loader, test_loader, edge_index, label="Model")

    # Get predictions
    print("\n[2/4] Getting predictions...")
    preds, targets = get_predictions(model, test_loader, edge_index, device)

    # Triangle analysis
    print("\n[3/4] Analyzing triangles...")
    triangle_results = analyze_triangle_errors(preds, targets, config)
    problematic = find_problematic_currency(triangle_results, config)
    propagation = analyze_error_propagation(preds, targets, config)

    # Visualization
    print("\n[4/4] Creating visualization...")
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    create_triangle_analysis_plot(triangle_results, problematic, propagation, config,
                                  os.path.join(exp_dir, "triangle_analysis.png"))

    # Print summary
    print("\n" + "=" * 60)
    print("TRIANGLE ANALYSIS SUMMARY")
    print("=" * 60)

    print("\n📊 Single-Error Cause Rate (어떤 통화가 삼각형을 깨뜨리나):")
    sorted_ccys = sorted(problematic.keys(),
                         key=lambda c: problematic[c]['deviation'],
                         reverse=True)
    for ccy in sorted_ccys:
        p = problematic[ccy]
        marker = "🔴" if p['is_problematic'] else "  "
        print(f"  {marker} {ccy:4s}: {p['single_error_rate']:.1%} "
              f"(deviation: {p['deviation']:+.1%})")

    print("\n🔗 Error Propagation (i가 틀리면 j도 틀릴 확률):")
    ccys = propagation['currencies']
    cond_err = propagation['conditional_error_matrix']
    # Top 5 pairs
    pairs = []
    for i in range(len(ccys)):
        for j in range(len(ccys)):
            if i != j:
                pairs.append((ccys[i], ccys[j], cond_err[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for i_ccy, j_ccy, prob in pairs[:5]:
        print(f"  {i_ccy} → {j_ccy}: {prob:.1%}")

    # Save results
    results = {
        'problematic_currencies': problematic,
        'error_propagation_top5': [
            {'from': p[0], 'to': p[1], 'probability': float(p[2])}
            for p in pairs[:10]
        ],
    }
    with open(os.path.join(exp_dir, "triangle_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Saved: triangle_analysis.png, triangle_results.json")


if __name__ == "__main__":
    main()
