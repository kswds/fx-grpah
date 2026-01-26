"""
Exp5: Rigorous Confidence-Based Prediction Analysis

Key Questions:
1. Is high confidence = high accuracy statistically significant?
2. Is this robust across random seeds?
3. What is the optimal accuracy-coverage tradeoff?
4. Do baseline models also have this property?
5. What market conditions lead to high confidence?
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
from models import FXStrengthGNN, FXStrengthNoGNN


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """Simple trainer for this experiment"""
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
    """Get predictions, targets, and macro features"""
    model.eval()
    all_preds, all_targets, all_macro = [], [], []

    with torch.no_grad():
        for xl, xm, yb in loader:
            xl, xm, yb = xl.to(device), xm.to(device), yb.to(device)
            rhat, _, _, _ = model(xl, xm, edge_index)
            all_preds.append(rhat.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
            all_macro.append(xm[:, -1, :].cpu().numpy())

    return (np.concatenate(all_preds),
            np.concatenate(all_targets),
            np.concatenate(all_macro))


def bootstrap_hit_rate(preds, targets, mask, n_bootstrap=1000, confidence=0.95):
    """Bootstrap confidence interval for hit rate"""
    preds_flat = preds[:, mask].flatten()
    targets_flat = targets[:, mask].flatten()

    hits = (np.sign(preds_flat) == np.sign(targets_flat)).astype(float)
    n = len(hits)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        bootstrap_means.append(hits[idx].mean())

    bootstrap_means = np.array(bootstrap_means)
    alpha = (1 - confidence) / 2
    ci_low = np.percentile(bootstrap_means, alpha * 100)
    ci_high = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return hits.mean(), ci_low, ci_high, bootstrap_means.std()


def bootstrap_hit_rate_by_confidence(preds, targets, mask, percentile_threshold,
                                      n_bootstrap=1000, confidence=0.95):
    """Bootstrap CI for hit rate at given confidence threshold"""
    preds_flat = preds[:, mask].flatten()
    targets_flat = targets[:, mask].flatten()

    # High confidence = high |prediction|
    threshold = np.percentile(np.abs(preds_flat), percentile_threshold)
    high_conf_mask = np.abs(preds_flat) >= threshold

    if high_conf_mask.sum() == 0:
        return None, None, None, 0, 0

    hits_high = (np.sign(preds_flat[high_conf_mask]) == np.sign(targets_flat[high_conf_mask])).astype(float)
    n = len(hits_high)
    coverage = high_conf_mask.mean()

    bootstrap_means = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        bootstrap_means.append(hits_high[idx].mean())

    bootstrap_means = np.array(bootstrap_means)
    alpha = (1 - confidence) / 2
    ci_low = np.percentile(bootstrap_means, alpha * 100)
    ci_high = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return hits_high.mean(), ci_low, ci_high, coverage, bootstrap_means.std()


def threshold_sweep(preds, targets, mask, thresholds=None):
    """Sweep across confidence thresholds"""
    if thresholds is None:
        thresholds = np.arange(0, 100, 5)

    preds_flat = preds[:, mask].flatten()
    targets_flat = targets[:, mask].flatten()

    results = []
    for pct in thresholds:
        threshold = np.percentile(np.abs(preds_flat), pct)
        high_conf_mask = np.abs(preds_flat) >= threshold

        if high_conf_mask.sum() < 10:
            continue

        hits = np.sign(preds_flat[high_conf_mask]) == np.sign(targets_flat[high_conf_mask])
        hit_rate = hits.mean()
        coverage = high_conf_mask.mean()
        n_samples = high_conf_mask.sum()

        results.append({
            'percentile': pct,
            'threshold': threshold,
            'hit_rate': hit_rate,
            'coverage': coverage,
            'n_samples': n_samples,
        })

    return results


def analyze_high_confidence_conditions(preds, targets, macro, mask, config, percentile=75):
    """Analyze what conditions lead to high confidence predictions"""
    preds_flat = preds[:, mask].flatten()
    targets_flat = targets[:, mask].flatten()

    # Reshape macro to match flattened predictions
    n_samples, n_ccy = preds.shape
    n_ccy_masked = mask.sum()
    macro_expanded = np.repeat(macro, n_ccy_masked, axis=0)

    threshold = np.percentile(np.abs(preds_flat), percentile)
    high_conf_mask = np.abs(preds_flat) >= threshold
    low_conf_mask = ~high_conf_mask

    results = {}
    macro_features = config.global_features

    for i, feat in enumerate(macro_features):
        feat_name = feat.replace("Global_", "")
        high_conf_mean = macro_expanded[high_conf_mask, i].mean()
        low_conf_mean = macro_expanded[low_conf_mask, i].mean()

        # T-test
        t_stat, p_value = stats.ttest_ind(
            macro_expanded[high_conf_mask, i],
            macro_expanded[low_conf_mask, i]
        )

        results[feat_name] = {
            'high_conf_mean': high_conf_mean,
            'low_conf_mean': low_conf_mean,
            'difference': high_conf_mean - low_conf_mean,
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
        }

    return results


def run_multi_seed_analysis(seeds, config, device, edge_index):
    """Run analysis across multiple random seeds"""
    all_results = []

    for seed in seeds:
        print(f"  Seed {seed}...", end=" ")
        set_seed(seed)

        train_loader, test_loader = create_dataloaders(config, macro_mode="real")
        model = FXStrengthGNN(config)
        trainer = Trainer(model, config, device)
        trainer.train(train_loader, edge_index, config.epochs)

        preds, targets, macro = get_predictions(model, test_loader, edge_index, device, config)

        mask = np.ones(config.n_ccy, dtype=bool)
        mask[config.usd_idx] = False

        # Overall hit rate
        preds_flat = preds[:, mask].flatten()
        targets_flat = targets[:, mask].flatten()
        overall_hit = (np.sign(preds_flat) == np.sign(targets_flat)).mean()

        # High confidence (q4) hit rate
        threshold = np.percentile(np.abs(preds_flat), 75)
        high_conf_mask = np.abs(preds_flat) >= threshold
        high_conf_hit = (np.sign(preds_flat[high_conf_mask]) == np.sign(targets_flat[high_conf_mask])).mean()

        print(f"Overall={overall_hit:.4f}, HighConf={high_conf_hit:.4f}")

        all_results.append({
            'seed': seed,
            'overall_hit': overall_hit,
            'high_conf_hit': high_conf_hit,
            'improvement': high_conf_hit - overall_hit,
        })

    return all_results


def compare_with_baseline(config, device, edge_index, seed=42):
    """Compare confidence-accuracy relationship with NoGNN baseline"""
    results = {}

    for model_name, model_cls in [("GNN", FXStrengthGNN), ("NoGNN", FXStrengthNoGNN)]:
        print(f"  Training {model_name}...")
        set_seed(seed)

        train_loader, test_loader = create_dataloaders(config, macro_mode="real")
        model = model_cls(config)
        trainer = Trainer(model, config, device)
        trainer.train(train_loader, edge_index, config.epochs)

        preds, targets, _ = get_predictions(model, test_loader, edge_index, device, config)

        mask = np.ones(config.n_ccy, dtype=bool)
        mask[config.usd_idx] = False

        sweep = threshold_sweep(preds, targets, mask)
        results[model_name] = sweep

    return results


def create_visualizations(sweep_results, multi_seed_results, baseline_comparison, output_dir):
    """Create visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Accuracy vs Coverage tradeoff
    ax = axes[0, 0]
    coverages = [r['coverage'] for r in sweep_results]
    hit_rates = [r['hit_rate'] for r in sweep_results]
    ax.plot(coverages, hit_rates, 'b-o', markersize=6)
    ax.axhline(y=0.75, color='g', linestyle='--', label='75% target')
    ax.axhline(y=0.5, color='r', linestyle='--', label='Random')
    ax.set_xlabel('Coverage (fraction of predictions used)')
    ax.set_ylabel('Hit Rate')
    ax.set_title('Accuracy vs Coverage Tradeoff')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.45, 0.85)

    # Add annotations for key points
    for r in sweep_results:
        if r['percentile'] in [0, 50, 75, 90, 95]:
            ax.annotate(f"p{int(r['percentile'])}",
                       (r['coverage'], r['hit_rate']),
                       textcoords="offset points", xytext=(5, 5), fontsize=8)

    # 2. Multi-seed robustness
    ax = axes[0, 1]
    seeds = [r['seed'] for r in multi_seed_results]
    overall_hits = [r['overall_hit'] for r in multi_seed_results]
    high_conf_hits = [r['high_conf_hit'] for r in multi_seed_results]

    x = np.arange(len(seeds))
    width = 0.35
    ax.bar(x - width/2, overall_hits, width, label='Overall', alpha=0.7)
    ax.bar(x + width/2, high_conf_hits, width, label='High Confidence (q4)', alpha=0.7)
    ax.axhline(y=0.75, color='g', linestyle='--', label='75% target')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Hit Rate')
    ax.set_title('Multi-Seed Robustness')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.legend()
    ax.set_ylim(0.5, 0.85)

    # 3. Baseline comparison
    ax = axes[1, 0]
    for model_name, sweep in baseline_comparison.items():
        coverages = [r['coverage'] for r in sweep]
        hit_rates = [r['hit_rate'] for r in sweep]
        ax.plot(coverages, hit_rates, '-o', markersize=4, label=model_name)

    ax.axhline(y=0.75, color='g', linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Hit Rate')
    ax.set_title('GNN vs NoGNN: Confidence-Accuracy Relationship')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.45, 0.85)

    # 4. Improvement distribution
    ax = axes[1, 1]
    improvements = [r['improvement'] for r in multi_seed_results]
    ax.hist(improvements, bins=10, edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(improvements), color='r', linestyle='--',
               label=f'Mean: {np.mean(improvements):.4f}')
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('High Conf Hit Rate - Overall Hit Rate')
    ax.set_ylabel('Count')
    ax.set_title('Improvement from Confidence Selection')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/confidence_analysis.png', dpi=150)
    plt.close()

    print(f"Visualizations saved to {output_dir}/")


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
    print("EXP5: RIGOROUS CONFIDENCE-BASED PREDICTION ANALYSIS")
    print("=" * 70)

    # 1. Train base model and get predictions
    print("\n>>> 1. Training base model...")
    set_seed(seed)
    train_loader, test_loader = create_dataloaders(config, macro_mode="real")
    model = FXStrengthGNN(config)
    trainer = Trainer(model, config, device)
    trainer.train(train_loader, edge_index, config.epochs)

    preds, targets, macro = get_predictions(model, test_loader, edge_index, device, config)

    mask = np.ones(config.n_ccy, dtype=bool)
    mask[config.usd_idx] = False

    # 2. Bootstrap analysis for overall and high-confidence
    print("\n>>> 2. Bootstrap Confidence Intervals...")

    overall_hit, overall_ci_low, overall_ci_high, overall_std = bootstrap_hit_rate(
        preds, targets, mask, n_bootstrap=2000)
    print(f"Overall Hit Rate: {overall_hit:.4f} [{overall_ci_low:.4f}, {overall_ci_high:.4f}]")

    # q4 (75th percentile)
    q4_hit, q4_ci_low, q4_ci_high, q4_coverage, q4_std = bootstrap_hit_rate_by_confidence(
        preds, targets, mask, percentile_threshold=75, n_bootstrap=2000)
    print(f"Q4 (top 25%) Hit Rate: {q4_hit:.4f} [{q4_ci_low:.4f}, {q4_ci_high:.4f}] (coverage={q4_coverage:.2%})")

    # Statistical test: is q4 significantly better than overall?
    # Using difference of proportions
    improvement = q4_hit - overall_hit
    se_diff = np.sqrt(overall_std**2 + q4_std**2)
    z_score = improvement / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    print(f"\nImprovement: {improvement:.4f} (z={z_score:.2f}, p={p_value:.4f})")
    if p_value < 0.05:
        print("  -> Statistically significant at p<0.05!")
    else:
        print("  -> NOT statistically significant")

    # 3. Threshold sweep
    print("\n>>> 3. Threshold Sweep (Accuracy vs Coverage)...")
    sweep_results = threshold_sweep(preds, targets, mask, thresholds=np.arange(0, 100, 5))

    print(f"{'Percentile':>10} {'Hit Rate':>10} {'Coverage':>10} {'N':>8}")
    print("-" * 40)
    for r in sweep_results:
        marker = "*" if r['hit_rate'] >= 0.75 else ""
        print(f"{r['percentile']:>10} {r['hit_rate']:>10.4f} {r['coverage']:>10.2%} {r['n_samples']:>8}{marker}")

    # Find minimum threshold for 75% accuracy
    above_75 = [r for r in sweep_results if r['hit_rate'] >= 0.75]
    if above_75:
        best_coverage_75 = max(above_75, key=lambda x: x['coverage'])
        print(f"\n  -> 75%+ accuracy achievable at p{best_coverage_75['percentile']} (coverage={best_coverage_75['coverage']:.2%})")
    else:
        print("\n  -> 75% accuracy NOT achievable at any threshold")

    # 4. Multi-seed robustness
    print("\n>>> 4. Multi-Seed Robustness (5 seeds)...")
    multi_seed_results = run_multi_seed_analysis(
        seeds=[42, 123, 456, 789, 1000],
        config=config, device=device, edge_index=edge_index
    )

    avg_overall = np.mean([r['overall_hit'] for r in multi_seed_results])
    avg_high_conf = np.mean([r['high_conf_hit'] for r in multi_seed_results])
    std_overall = np.std([r['overall_hit'] for r in multi_seed_results])
    std_high_conf = np.std([r['high_conf_hit'] for r in multi_seed_results])

    print(f"\nAverage Overall Hit: {avg_overall:.4f} +/- {std_overall:.4f}")
    print(f"Average High Conf Hit: {avg_high_conf:.4f} +/- {std_high_conf:.4f}")

    # All seeds above 75%?
    all_above_75 = all(r['high_conf_hit'] >= 0.75 for r in multi_seed_results)
    print(f"All seeds achieve 75%+ in high confidence: {all_above_75}")

    # 5. Baseline comparison
    print("\n>>> 5. Baseline Comparison (GNN vs NoGNN)...")
    baseline_comparison = compare_with_baseline(config, device, edge_index, seed=42)

    # Compare at 75th percentile
    for model_name, sweep in baseline_comparison.items():
        r75 = [r for r in sweep if r['percentile'] == 75][0]
        print(f"  {model_name}: Hit@p75 = {r75['hit_rate']:.4f}")

    # 6. Analyze high-confidence conditions
    print("\n>>> 6. High-Confidence Market Conditions...")
    conditions = analyze_high_confidence_conditions(preds, targets, macro, mask, config, percentile=75)

    print(f"{'Feature':<12} {'High Conf':>10} {'Low Conf':>10} {'Diff':>8} {'p-value':>10}")
    print("-" * 52)
    for feat, r in sorted(conditions.items(), key=lambda x: abs(x[1]['t_stat']), reverse=True):
        sig = "*" if r['significant'] else ""
        print(f"{feat:<12} {r['high_conf_mean']:>10.4f} {r['low_conf_mean']:>10.4f} "
              f"{r['difference']:>+8.4f} {r['p_value']:>10.4f}{sig}")

    # Create visualizations
    output_dir = "exp5_confidence_analysis"
    create_visualizations(sweep_results, multi_seed_results, baseline_comparison, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n1. STATISTICAL SIGNIFICANCE:")
    print(f"   - High confidence hit rate: {q4_hit:.4f} [{q4_ci_low:.4f}, {q4_ci_high:.4f}]")
    print(f"   - Overall hit rate: {overall_hit:.4f} [{overall_ci_low:.4f}, {overall_ci_high:.4f}]")
    print(f"   - Improvement: {improvement:.4f} (p={p_value:.4f})")

    print(f"\n2. ROBUSTNESS (5 seeds):")
    print(f"   - High conf hit rate: {avg_high_conf:.4f} +/- {std_high_conf:.4f}")
    print(f"   - All seeds >= 75%: {all_above_75}")

    print(f"\n3. 75% ACCURACY TARGET:")
    if above_75:
        print(f"   - Achievable at {best_coverage_75['coverage']:.1%} coverage (p{best_coverage_75['percentile']})")
    else:
        print(f"   - NOT achievable")

    print(f"\n4. BASELINE COMPARISON:")
    gnn_75 = [r for r in baseline_comparison['GNN'] if r['percentile'] == 75][0]['hit_rate']
    nognn_75 = [r for r in baseline_comparison['NoGNN'] if r['percentile'] == 75][0]['hit_rate']
    print(f"   - GNN @ p75: {gnn_75:.4f}")
    print(f"   - NoGNN @ p75: {nognn_75:.4f}")
    if gnn_75 > nognn_75:
        print(f"   - GNN is better by {gnn_75 - nognn_75:.4f}")
    else:
        print(f"   - NoGNN is better by {nognn_75 - gnn_75:.4f}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'bootstrap': {
            'overall': {'hit': overall_hit, 'ci_low': overall_ci_low, 'ci_high': overall_ci_high},
            'q4': {'hit': q4_hit, 'ci_low': q4_ci_low, 'ci_high': q4_ci_high, 'coverage': q4_coverage},
            'improvement': {'value': improvement, 'z_score': z_score, 'p_value': p_value},
        },
        'threshold_sweep': sweep_results,
        'multi_seed': multi_seed_results,
        'baseline_comparison': {k: v for k, v in baseline_comparison.items()},
        'market_conditions': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                   for kk, vv in v.items()}
                              for k, v in conditions.items()},
        'summary': {
            'significant_improvement': p_value < 0.05,
            'robust_across_seeds': all_above_75,
            'achieves_75_target': len(above_75) > 0,
            'best_coverage_at_75': best_coverage_75['coverage'] if above_75 else None,
        }
    }

    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
