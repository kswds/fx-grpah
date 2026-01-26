"""
Exp7: Big Move Analysis

Question: Does the model correctly predict the BIG moves?
- Predicting small moves correctly is less valuable
- What matters is getting the direction right when the move is large

Analysis:
1. Accuracy stratified by ACTUAL move size (not prediction size)
2. Profit contribution by move size quintile
3. Large move detection rate
"""
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

import random
import json
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

from config import Config
from dataset import create_dataloaders, fully_connected_edge_index
from models import FXStrengthGNN


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)

    def loss_fn(self, rhat, y, ds, A_param):
        mask = torch.ones(self.config.n_ccy, dtype=torch.bool, device=y.device)
        mask[self.config.usd_idx] = False
        mse = ((rhat[:, mask] - y[:, mask]) ** 2).mean()
        var_term = -ds.var(dim=1).mean()
        l1_A = A_param.abs().mean()
        return mse + self.config.lambda_var * var_term + self.config.lambda_a_l1 * l1_A

    def train_epoch(self, train_loader, edge_index):
        self.model.train()
        for xl, xm, yb in train_loader:
            xl, xm, yb = xl.to(self.device), xm.to(self.device), yb.to(self.device)
            rhat, ds, _, _ = self.model(xl, xm, edge_index)
            loss = self.loss_fn(rhat, yb, ds, self.model.A)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

    def train(self, train_loader, edge_index, epochs):
        for _ in range(epochs):
            self.train_epoch(train_loader, edge_index)


def get_predictions(model, loader, edge_index, device, config):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xl, xm, yb in loader:
            xl, xm, yb = xl.to(device), xm.to(device), yb.to(device)
            rhat, _, _, _ = model(xl, xm, edge_index)
            all_preds.append(rhat.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def analyze_by_actual_move_size(preds, targets, mask):
    """
    Analyze accuracy stratified by ACTUAL move size (target magnitude).

    Key question: When the market makes a big move, do we predict it correctly?
    """
    preds_flat = preds[:, mask].flatten()
    targets_flat = targets[:, mask].flatten()

    # Define quintiles by ACTUAL move size
    abs_targets = np.abs(targets_flat)
    quintiles = np.percentile(abs_targets, [20, 40, 60, 80])

    results = {}

    # Analyze each quintile
    quintile_labels = ['Q1 (Smallest)', 'Q2', 'Q3', 'Q4', 'Q5 (Largest)']
    quintile_ranges = [
        (0, quintiles[0]),
        (quintiles[0], quintiles[1]),
        (quintiles[1], quintiles[2]),
        (quintiles[2], quintiles[3]),
        (quintiles[3], np.inf)
    ]

    for i, (label, (low, high)) in enumerate(zip(quintile_labels, quintile_ranges)):
        in_quintile = (abs_targets >= low) & (abs_targets < high)

        preds_q = preds_flat[in_quintile]
        targets_q = targets_flat[in_quintile]

        # Direction accuracy
        correct = (np.sign(preds_q) == np.sign(targets_q))
        accuracy = correct.mean()

        # Profit contribution
        # If we bet on prediction direction, how much do we gain/lose?
        returns = np.sign(preds_q) * targets_q
        total_return = returns.sum()
        avg_return = returns.mean()

        # Sample size
        n_samples = len(preds_q)
        pct_of_total = n_samples / len(preds_flat) * 100

        results[label] = {
            'accuracy': accuracy,
            'n_samples': n_samples,
            'pct_of_total': pct_of_total,
            'avg_abs_move': abs_targets[in_quintile].mean(),
            'total_return': total_return,
            'avg_return': avg_return,
            'return_contribution_pct': 0,  # Will calculate after
        }

    # Calculate return contribution percentages
    total_abs_return = sum(abs(r['total_return']) for r in results.values())
    for label in results:
        if total_abs_return > 0:
            results[label]['return_contribution_pct'] = (
                results[label]['total_return'] / total_abs_return * 100
            )

    return results, quintiles


def analyze_extreme_moves(preds, targets, mask, percentile=90):
    """
    Focus specifically on extreme moves (top X% by actual size).

    These are the moves that matter most for trading.
    """
    preds_flat = preds[:, mask].flatten()
    targets_flat = targets[:, mask].flatten()
    abs_targets = np.abs(targets_flat)

    threshold = np.percentile(abs_targets, percentile)

    # Extreme moves (actual)
    extreme_mask = abs_targets >= threshold
    normal_mask = abs_targets < threshold

    # Accuracy on extreme vs normal
    extreme_correct = (np.sign(preds_flat[extreme_mask]) == np.sign(targets_flat[extreme_mask]))
    normal_correct = (np.sign(preds_flat[normal_mask]) == np.sign(targets_flat[normal_mask]))

    extreme_accuracy = extreme_correct.mean()
    normal_accuracy = normal_correct.mean()

    # Returns from extreme moves
    extreme_returns = np.sign(preds_flat[extreme_mask]) * targets_flat[extreme_mask]
    normal_returns = np.sign(preds_flat[normal_mask]) * targets_flat[normal_mask]

    # Statistical test (manual z-test for proportions)
    p1 = extreme_correct.mean()
    p2 = normal_correct.mean()
    n1 = len(extreme_correct)
    n2 = len(normal_correct)
    p_pooled = (extreme_correct.sum() + normal_correct.sum()) / (n1 + n2)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    z_stat = (p1 - p2) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        'threshold_percentile': percentile,
        'threshold_value': threshold,
        'extreme_accuracy': extreme_accuracy,
        'normal_accuracy': normal_accuracy,
        'extreme_count': extreme_mask.sum(),
        'normal_count': normal_mask.sum(),
        'extreme_total_return': extreme_returns.sum(),
        'normal_total_return': normal_returns.sum(),
        'extreme_avg_return': extreme_returns.mean(),
        'normal_avg_return': normal_returns.mean(),
        'z_stat': z_stat,
        'p_value': p_value,
    }


def analyze_daily_biggest_move(preds, targets, mask, config):
    """
    Each day, identify the currency with the BIGGEST actual move.
    How often do we get this one right?
    """
    n_samples = len(preds)

    daily_results = []

    for t in range(n_samples):
        pred_t = preds[t, mask]
        target_t = targets[t, mask]

        # Find biggest mover today
        abs_moves = np.abs(target_t)
        biggest_idx = np.argmax(abs_moves)

        # Did we predict direction correctly?
        pred_sign = np.sign(pred_t[biggest_idx])
        target_sign = np.sign(target_t[biggest_idx])
        correct = pred_sign == target_sign

        # Was our prediction for this currency also confident?
        pred_rank = np.argsort(np.abs(pred_t))[::-1]
        pred_confidence_rank = np.where(pred_rank == biggest_idx)[0][0] + 1  # 1-indexed

        daily_results.append({
            'actual_biggest_move': abs_moves[biggest_idx],
            'correct_direction': correct,
            'pred_confidence_rank': pred_confidence_rank,
            'was_top_3_prediction': pred_confidence_rank <= 3,
        })

    df_results = {
        'accuracy_on_biggest': np.mean([r['correct_direction'] for r in daily_results]),
        'avg_confidence_rank': np.mean([r['pred_confidence_rank'] for r in daily_results]),
        'pct_in_top3_prediction': np.mean([r['was_top_3_prediction'] for r in daily_results]),
        'n_days': n_samples,
    }

    return df_results, daily_results


def analyze_when_both_big(preds, targets, mask, pred_percentile=75, target_percentile=75):
    """
    When BOTH prediction is confident AND actual move is large,
    what's the accuracy?

    This is the "high conviction, big move" scenario.
    """
    preds_flat = preds[:, mask].flatten()
    targets_flat = targets[:, mask].flatten()

    pred_threshold = np.percentile(np.abs(preds_flat), pred_percentile)
    target_threshold = np.percentile(np.abs(targets_flat), target_percentile)

    # Different scenarios
    scenarios = {
        'both_big': (np.abs(preds_flat) >= pred_threshold) & (np.abs(targets_flat) >= target_threshold),
        'pred_big_only': (np.abs(preds_flat) >= pred_threshold) & (np.abs(targets_flat) < target_threshold),
        'target_big_only': (np.abs(preds_flat) < pred_threshold) & (np.abs(targets_flat) >= target_threshold),
        'both_small': (np.abs(preds_flat) < pred_threshold) & (np.abs(targets_flat) < target_threshold),
    }

    results = {}
    for name, mask_scenario in scenarios.items():
        if mask_scenario.sum() > 0:
            correct = (np.sign(preds_flat[mask_scenario]) == np.sign(targets_flat[mask_scenario]))
            returns = np.sign(preds_flat[mask_scenario]) * targets_flat[mask_scenario]
            results[name] = {
                'accuracy': correct.mean(),
                'count': mask_scenario.sum(),
                'pct_of_total': mask_scenario.sum() / len(preds_flat) * 100,
                'total_return': returns.sum(),
                'avg_return': returns.mean(),
            }

    return results


def main():
    seed = 42
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = Config(seed=seed, epochs=30, batch_size=128, lr=3e-4)
    config.use_skip_connection = True
    config.use_layer_norm = True

    edge_index = fully_connected_edge_index(config.n_ccy).to(device)

    print("\n" + "=" * 70)
    print("EXP7: BIG MOVE ANALYSIS")
    print("=" * 70)
    print("Question: Does the model correctly predict BIG moves?")

    # Train model
    print("\n>>> Training model...")
    train_loader, test_loader = create_dataloaders(config, macro_mode="real")
    model = FXStrengthGNN(config)
    trainer = Trainer(model, config, device)
    trainer.train(train_loader, edge_index, config.epochs)

    # Get predictions
    print(">>> Getting predictions...")
    preds, targets = get_predictions(model, test_loader, edge_index, device, config)

    mask = np.ones(config.n_ccy, dtype=bool)
    mask[config.usd_idx] = False

    print(f"Test samples: {len(preds)}")

    # Overall accuracy for reference
    preds_flat = preds[:, mask].flatten()
    targets_flat = targets[:, mask].flatten()
    overall_accuracy = (np.sign(preds_flat) == np.sign(targets_flat)).mean()
    print(f"Overall hit rate: {overall_accuracy:.1%}")

    # 1. Accuracy by actual move size quintile
    print("\n" + "=" * 70)
    print("1. ACCURACY BY ACTUAL MOVE SIZE (Target Quintile)")
    print("=" * 70)

    quintile_results, quintiles = analyze_by_actual_move_size(preds, targets, mask)

    print(f"\n{'Quintile':<15} {'Accuracy':>10} {'Avg |Move|':>12} {'Samples':>10} {'Return %':>12}")
    print("-" * 60)

    for label, r in quintile_results.items():
        print(f"{label:<15} {r['accuracy']:>9.1%} {r['avg_abs_move']:>12.4f} "
              f"{r['n_samples']:>10} {r['return_contribution_pct']:>11.1f}%")

    # Key finding
    q1_acc = quintile_results['Q1 (Smallest)']['accuracy']
    q5_acc = quintile_results['Q5 (Largest)']['accuracy']
    print(f"\n>>> Key: Q5 (biggest moves) accuracy = {q5_acc:.1%} vs Q1 (smallest) = {q1_acc:.1%}")
    print(f">>> Difference: {q5_acc - q1_acc:+.1%}")

    # 2. Extreme move analysis
    print("\n" + "=" * 70)
    print("2. EXTREME MOVE ANALYSIS (Top 10% Actual Moves)")
    print("=" * 70)

    extreme_results = analyze_extreme_moves(preds, targets, mask, percentile=90)

    print(f"\nExtreme moves (top 10% by actual size):")
    print(f"  - Accuracy: {extreme_results['extreme_accuracy']:.1%}")
    print(f"  - Sample count: {extreme_results['extreme_count']}")
    print(f"  - Total return contribution: {extreme_results['extreme_total_return']:.2f}")

    print(f"\nNormal moves (bottom 90%):")
    print(f"  - Accuracy: {extreme_results['normal_accuracy']:.1%}")
    print(f"  - Sample count: {extreme_results['normal_count']}")

    print(f"\n>>> Statistical test (extreme vs normal):")
    print(f"    z-statistic: {extreme_results['z_stat']:.2f}")
    print(f"    p-value: {extreme_results['p_value']:.6f}")

    if extreme_results['extreme_accuracy'] > extreme_results['normal_accuracy']:
        print(f">>> ✓ Model is BETTER at predicting big moves!")
    else:
        print(f">>> Model accuracy is similar or worse for big moves")

    # 3. Daily biggest mover analysis
    print("\n" + "=" * 70)
    print("3. DAILY BIGGEST MOVER ANALYSIS")
    print("=" * 70)
    print("Each day, does the model correctly predict the biggest mover?")

    daily_results, _ = analyze_daily_biggest_move(preds, targets, mask, config)

    print(f"\nAccuracy on daily biggest mover: {daily_results['accuracy_on_biggest']:.1%}")
    print(f"Avg confidence rank of biggest mover: {daily_results['avg_confidence_rank']:.1f} (out of 9)")
    print(f"% of time biggest mover was in our top-3 predictions: {daily_results['pct_in_top3_prediction']:.1%}")

    # Random baseline comparison
    random_accuracy = 0.5  # 50% for direction
    random_top3 = 3/9  # ~33% chance

    print(f"\n>>> Comparison to random baseline:")
    print(f"    Direction accuracy: {daily_results['accuracy_on_biggest']:.1%} vs {random_accuracy:.1%} (random)")
    print(f"    In top-3 prediction: {daily_results['pct_in_top3_prediction']:.1%} vs {random_top3:.1%} (random)")

    # 4. High conviction + Big move scenario
    print("\n" + "=" * 70)
    print("4. HIGH CONVICTION + BIG MOVE ANALYSIS")
    print("=" * 70)
    print("When our prediction is confident AND the actual move is large")

    scenario_results = analyze_when_both_big(preds, targets, mask)

    print(f"\n{'Scenario':<20} {'Accuracy':>10} {'% Total':>10} {'Avg Return':>12}")
    print("-" * 55)

    scenario_order = ['both_big', 'pred_big_only', 'target_big_only', 'both_small']
    scenario_labels = {
        'both_big': 'Both Big',
        'pred_big_only': 'Pred Big, Move Small',
        'target_big_only': 'Pred Small, Move Big',
        'both_small': 'Both Small',
    }

    for scenario in scenario_order:
        if scenario in scenario_results:
            r = scenario_results[scenario]
            label = scenario_labels[scenario]
            print(f"{label:<20} {r['accuracy']:>9.1%} {r['pct_of_total']:>9.1f}% {r['avg_return']:>12.4f}")

    if 'both_big' in scenario_results:
        both_big_acc = scenario_results['both_big']['accuracy']
        print(f"\n>>> Key: When both prediction and move are big: {both_big_acc:.1%} accuracy")

    # Visualization
    output_dir = "exp7_big_move_analysis"
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Accuracy by quintile
    ax = axes[0, 0]
    labels = list(quintile_results.keys())
    accs = [quintile_results[l]['accuracy'] * 100 for l in labels]
    colors = plt.cm.RdYlGn([0.2, 0.35, 0.5, 0.65, 0.8])

    bars = ax.bar(range(len(labels)), accs, color=colors)
    ax.axhline(y=overall_accuracy * 100, color='k', linestyle='--', label=f'Overall ({overall_accuracy:.1%})')
    ax.axhline(y=50, color='r', linestyle=':', label='Random (50%)')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(['Q1\n(Small)', 'Q2', 'Q3', 'Q4', 'Q5\n(Big)'])
    ax.set_ylabel('Hit Rate (%)')
    ax.set_title('Accuracy by Actual Move Size')
    ax.legend()
    ax.set_ylim(40, 80)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{acc:.1f}%', ha='center', fontsize=9)

    # 2. Return contribution by quintile
    ax = axes[0, 1]
    returns = [quintile_results[l]['total_return'] for l in labels]
    colors_ret = ['red' if r < 0 else 'green' for r in returns]

    ax.bar(range(len(labels)), returns, color=colors_ret)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(['Q1\n(Small)', 'Q2', 'Q3', 'Q4', 'Q5\n(Big)'])
    ax.set_ylabel('Total Return Contribution')
    ax.set_title('Return Contribution by Move Size')

    # 3. Extreme vs Normal
    ax = axes[1, 0]
    categories = ['Extreme (Top 10%)', 'Normal (Bottom 90%)']
    accuracies = [extreme_results['extreme_accuracy'] * 100, extreme_results['normal_accuracy'] * 100]
    counts = [extreme_results['extreme_count'], extreme_results['normal_count']]

    bars = ax.bar(categories, accuracies, color=['darkgreen', 'steelblue'])
    ax.axhline(y=50, color='r', linestyle=':', label='Random')
    ax.set_ylabel('Hit Rate (%)')
    ax.set_title('Extreme vs Normal Move Accuracy')
    ax.set_ylim(40, 80)

    for bar, acc, cnt in zip(bars, accuracies, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{acc:.1f}%\n(n={cnt})', ha='center', fontsize=9)

    # 4. Scenario matrix
    ax = axes[1, 1]
    if all(s in scenario_results for s in ['both_big', 'pred_big_only', 'target_big_only', 'both_small']):
        matrix = np.array([
            [scenario_results['both_small']['accuracy'], scenario_results['target_big_only']['accuracy']],
            [scenario_results['pred_big_only']['accuracy'], scenario_results['both_big']['accuracy']]
        ]) * 100

        im = ax.imshow(matrix, cmap='RdYlGn', vmin=50, vmax=85)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Small Move', 'Big Move'])
        ax.set_yticklabels(['Small Pred', 'Big Pred'])
        ax.set_xlabel('Actual Move Size')
        ax.set_ylabel('Prediction Confidence')
        ax.set_title('Accuracy by Prediction × Actual Size')

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{matrix[i, j]:.1f}%', ha='center', va='center',
                       fontsize=12, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Hit Rate (%)')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/big_move_analysis.png', dpi=150)
    plt.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: BIG MOVE PREDICTION CAPABILITY")
    print("=" * 70)

    print(f"""
1. ACCURACY BY MOVE SIZE:
   - Smallest moves (Q1): {quintile_results['Q1 (Smallest)']['accuracy']:.1%}
   - Largest moves (Q5):  {quintile_results['Q5 (Largest)']['accuracy']:.1%}
   - Trend: {"✓ Better on big moves" if q5_acc > q1_acc else "✗ Worse on big moves"}

2. EXTREME MOVE PERFORMANCE (Top 10%):
   - Extreme accuracy: {extreme_results['extreme_accuracy']:.1%}
   - Normal accuracy:  {extreme_results['normal_accuracy']:.1%}
   - Statistical significance: p={extreme_results['p_value']:.4f}

3. DAILY BIGGEST MOVER:
   - Direction accuracy: {daily_results['accuracy_on_biggest']:.1%} (vs 50% random)
   - In top-3 prediction: {daily_results['pct_in_top3_prediction']:.1%} (vs 33% random)

4. HIGH CONVICTION + BIG MOVE:
   - Accuracy: {scenario_results.get('both_big', {}).get('accuracy', 0):.1%}
   - Coverage: {scenario_results.get('both_big', {}).get('pct_of_total', 0):.1f}% of samples

CONCLUSION:
{"✓ Model is effective at predicting big moves" if q5_acc > q1_acc + 0.03 else "△ Model shows similar performance across move sizes"}
""")

    # Save results
    def convert(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    output = {
        'timestamp': datetime.now().isoformat(),
        'overall_accuracy': overall_accuracy,
        'quintile_analysis': convert(quintile_results),
        'extreme_analysis': convert(extreme_results),
        'daily_biggest': convert(daily_results),
        'scenario_analysis': convert(scenario_results),
    }

    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
