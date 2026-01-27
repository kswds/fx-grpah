"""
Experiment: Rank-based Loss Functions

Tests different rank-aware loss functions to improve:
1. Overall rank correlation (Spearman)
2. Top/Bottom-K accuracy
3. Long-Short strategy performance
"""
import torch
import numpy as np
from scipy.stats import spearmanr, kendalltau
from config import Config
from models import FXStrengthGNN
from dataset import create_dataloaders, fully_connected_edge_index
from train import loss_fn, rank_loss_fn, listMLE_loss, pairwise_margin_loss, topk_focused_loss


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def train_with_loss(config, train_loader, edge_index, device, loss_type='mse', epochs=30):
    """Train with specified loss type"""
    model = FXStrengthGNN(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xl, xm, yb in train_loader:
            xl, xm, yb = xl.to(device), xm.to(device), yb.to(device)
            rhat, ds, _, _ = model(xl, xm, edge_index)

            mask = torch.ones(config.n_ccy, dtype=torch.bool, device=device)
            mask[config.usd_idx] = False

            if loss_type == 'mse':
                loss = loss_fn(rhat, yb, ds, model.A, config)
            elif loss_type == 'rank_combined':
                loss, _ = rank_loss_fn(rhat, yb, ds, model.A, config,
                                       lambda_mse=1.0, lambda_list=0.5, lambda_pair=0.3, lambda_topk=0.2)
            elif loss_type == 'listmle':
                mse = ((rhat[:, mask] - yb[:, mask]) ** 2).mean()
                list_loss = listMLE_loss(rhat[:, mask], yb[:, mask])
                var_term = -ds.var(dim=1).mean()
                loss = mse + 0.5 * list_loss + config.lambda_var * var_term
            elif loss_type == 'pairwise':
                mse = ((rhat[:, mask] - yb[:, mask]) ** 2).mean()
                pair_loss = pairwise_margin_loss(rhat[:, mask], yb[:, mask])
                var_term = -ds.var(dim=1).mean()
                loss = mse + 0.5 * pair_loss + config.lambda_var * var_term
            elif loss_type == 'topk':
                mse = ((rhat[:, mask] - yb[:, mask]) ** 2).mean()
                topk_loss = topk_focused_loss(rhat[:, mask], yb[:, mask], k=2, alpha=3.0)
                var_term = -ds.var(dim=1).mean()
                loss = 0.5 * mse + 0.5 * topk_loss + config.lambda_var * var_term
            elif loss_type == 'extreme_only':
                # Only care about top-1 and bottom-1
                pred = rhat[:, mask]
                target = yb[:, mask]
                # Top-1: maximize pred of argmax(target)
                # Bottom-1: minimize pred of argmin(target)
                top_idx = target.argmax(dim=1)
                bot_idx = target.argmin(dim=1)
                B = pred.shape[0]
                top_pred = pred[torch.arange(B, device=device), top_idx]
                bot_pred = pred[torch.arange(B, device=device), bot_idx]
                # Want top_pred > bot_pred with margin
                extreme_loss = F.relu(0.5 - (top_pred - bot_pred)).mean()
                mse = ((pred - target) ** 2).mean()
                var_term = -ds.var(dim=1).mean()
                loss = 0.5 * mse + 0.5 * extreme_loss + config.lambda_var * var_term
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item() if isinstance(loss, torch.Tensor) else loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{loss_type}] Epoch {epoch:02d}: loss = {np.mean(losses):.4f}")

    return model


def evaluate_rank_metrics(model, test_loader, edge_index, config, device):
    """Evaluate rank prediction metrics"""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for xl, xm, yb in test_loader:
            xl, xm = xl.to(device), xm.to(device)
            rhat, _, _, _ = model(xl, xm, edge_index)
            all_preds.append(rhat.cpu().numpy())
            all_targets.append(yb.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Exclude USD
    mask = np.array([i != config.usd_idx for i in range(config.n_ccy)])
    preds = preds[:, mask]
    targets = targets[:, mask]
    n_ccy = preds.shape[1]

    results = {}

    # 1. Direction accuracy
    correct = (np.sign(preds) == np.sign(targets))
    results['hit_rate'] = correct.mean()

    # 2. Spearman correlation
    spearman_corrs = []
    for i in range(len(preds)):
        sp, _ = spearmanr(preds[i], targets[i])
        if not np.isnan(sp):
            spearman_corrs.append(sp)
    results['spearman'] = np.mean(spearman_corrs)

    # 3. Top-K accuracy
    for k in [1, 2, 3]:
        correct = 0
        total = 0
        for i in range(len(preds)):
            pred_topk = set(np.argsort(preds[i])[-k:])
            actual_topk = set(np.argsort(targets[i])[-k:])
            overlap = len(pred_topk & actual_topk)
            correct += overlap
            total += k
        results[f'top{k}_acc'] = correct / total

    # 4. Bottom-K accuracy
    for k in [1, 2, 3]:
        correct = 0
        total = 0
        for i in range(len(preds)):
            pred_botk = set(np.argsort(preds[i])[:k])
            actual_botk = set(np.argsort(targets[i])[:k])
            overlap = len(pred_botk & actual_botk)
            correct += overlap
            total += k
        results[f'bot{k}_acc'] = correct / total

    # 5. Strongest/Weakest exact match
    results['strongest_exact'] = np.mean([np.argmax(preds[i]) == np.argmax(targets[i]) for i in range(len(preds))])
    results['weakest_exact'] = np.mean([np.argmin(preds[i]) == np.argmin(targets[i]) for i in range(len(preds))])

    # 6. Long-Short strategy (Long Top-1, Short Bottom-1)
    returns = []
    for i in range(len(preds)):
        long_idx = np.argmax(preds[i])
        short_idx = np.argmin(preds[i])
        ret = targets[i, long_idx] - targets[i, short_idx]
        returns.append(ret)
    returns = np.array(returns)
    results['ls_return'] = returns.mean()
    results['ls_sharpe'] = returns.mean() / returns.std() * np.sqrt(252)
    results['ls_win_rate'] = (returns > 0).mean()

    # Random baseline
    results['random_top1'] = 1 / n_ccy

    return results


def main():
    print("="*70)
    print("Experiment: Rank-based Loss Functions")
    print("="*70)

    seed = 42
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    config = Config()
    train_loader, test_loader = create_dataloaders(config)
    edge_index = fully_connected_edge_index(config.n_ccy).to(device)

    print(f"Currencies ({config.n_ccy}): {config.ccys}")

    # Loss types to test
    loss_types = [
        'mse',           # Baseline MSE only
        'listmle',       # MSE + ListMLE
        'pairwise',      # MSE + Pairwise margin
        'topk',          # MSE + TopK focused
        'rank_combined', # MSE + All rank losses
        'extreme_only',  # MSE + Extreme (top1/bot1) loss
    ]

    all_results = {}

    for loss_type in loss_types:
        print(f"\n{'-'*70}")
        print(f"Training with loss: {loss_type}")
        print(f"{'-'*70}")

        set_seed(seed)
        model = train_with_loss(config, train_loader, edge_index, device, loss_type)
        results = evaluate_rank_metrics(model, test_loader, edge_index, config, device)
        all_results[loss_type] = results

    # Print comparison table
    print("\n" + "="*90)
    print("RESULTS COMPARISON")
    print("="*90)

    # Header
    metrics = ['hit_rate', 'spearman', 'top1_acc', 'top2_acc', 'bot1_acc', 'bot2_acc', 'ls_sharpe', 'ls_win_rate']
    header = f"{'Loss Type':<15}"
    for m in metrics:
        header += f" | {m:>10}"
    print(header)
    print("-"*90)

    # Rows
    for loss_type in loss_types:
        r = all_results[loss_type]
        row = f"{loss_type:<15}"
        for m in metrics:
            val = r[m]
            if 'rate' in m or 'acc' in m:
                row += f" | {val:>9.1%}"
            elif m == 'spearman':
                row += f" | {val:>10.3f}"
            else:
                row += f" | {val:>10.2f}"
        print(row)

    print("-"*90)
    print(f"Random baseline (Top-1): {all_results['mse']['random_top1']:.1%}")

    # Best model summary
    print("\n" + "="*70)
    print("BEST MODELS BY METRIC")
    print("="*70)

    for metric in ['hit_rate', 'spearman', 'top1_acc', 'bot1_acc', 'ls_sharpe']:
        best_loss = max(all_results.keys(), key=lambda x: all_results[x][metric])
        best_val = all_results[best_loss][metric]
        baseline_val = all_results['mse'][metric]
        improvement = (best_val - baseline_val) / abs(baseline_val) * 100 if baseline_val != 0 else 0

        if 'rate' in metric or 'acc' in metric:
            print(f"  {metric}: {best_loss} ({best_val:.1%}) [baseline: {baseline_val:.1%}, +{improvement:.1f}%]")
        elif metric == 'spearman':
            print(f"  {metric}: {best_loss} ({best_val:.3f}) [baseline: {baseline_val:.3f}, +{improvement:.1f}%]")
        else:
            print(f"  {metric}: {best_loss} ({best_val:.2f}) [baseline: {baseline_val:.2f}, +{improvement:.1f}%]")


if __name__ == '__main__':
    import torch.nn.functional as F
    main()
