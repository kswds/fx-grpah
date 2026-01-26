"""
Exp6b: Trading Backtest with Transaction Costs

Realistic simulation including:
1. Bid-ask spread (transaction cost)
2. Different cost tiers (retail vs institutional)
3. Break-even analysis
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


def simulate_trading_with_costs(preds, targets, mask,
                                 strategy='all',
                                 confidence_threshold=None,
                                 transaction_cost_bps=10,
                                 position_size=0.01):
    """
    Simulate trading with transaction costs.

    Args:
        transaction_cost_bps: Round-trip cost in basis points
            - Retail: 20-50 bps
            - Institutional: 5-15 bps
            - HFT/Prime: 2-5 bps
    """
    n_samples, n_ccy = preds.shape
    n_ccy_masked = mask.sum()

    cost_rate = transaction_cost_bps / 10000  # Convert bps to decimal

    daily_returns = []
    daily_returns_gross = []
    daily_costs = []
    trades_per_day = []
    wins_per_day = []

    for t in range(n_samples):
        pred_t = preds[t, mask]
        target_t = targets[t, mask]

        if strategy == 'selective' and confidence_threshold is not None:
            trade_mask = np.abs(pred_t) >= confidence_threshold
        else:
            trade_mask = np.ones(n_ccy_masked, dtype=bool)

        n_trades = trade_mask.sum()

        if n_trades == 0:
            daily_returns.append(0)
            daily_returns_gross.append(0)
            daily_costs.append(0)
            trades_per_day.append(0)
            wins_per_day.append(0)
            continue

        pred_traded = pred_t[trade_mask]
        target_traded = target_t[trade_mask]

        # Gross returns (before costs)
        gross_returns = position_size * np.sign(pred_traded) * target_traded
        gross_return = gross_returns.mean()

        # Transaction cost (applied to each trade)
        # Cost is proportional to position size and number of trades
        total_cost = position_size * cost_rate * n_trades / n_ccy_masked

        # Net return
        net_return = gross_return - total_cost

        daily_returns.append(net_return)
        daily_returns_gross.append(gross_return)
        daily_costs.append(total_cost)
        trades_per_day.append(n_trades)
        wins_per_day.append((np.sign(pred_traded) == np.sign(target_traded)).sum())

    return (np.array(daily_returns), np.array(daily_returns_gross),
            np.array(daily_costs), np.array(trades_per_day), np.array(wins_per_day))


def calculate_metrics(returns, returns_gross, costs, trades, wins):
    """Calculate trading metrics with cost breakdown"""
    active_mask = trades > 0
    active_returns = returns[active_mask]

    # Returns
    total_return_net = (1 + returns).prod() - 1
    total_return_gross = (1 + returns_gross).prod() - 1
    total_costs = costs.sum()

    cumulative_returns = np.cumprod(1 + returns)

    # Sharpe ratio
    if len(active_returns) > 0 and active_returns.std() > 0:
        sharpe = active_returns.mean() / active_returns.std() * np.sqrt(252)
    else:
        sharpe = 0

    # Max drawdown
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    # Win rate
    total_trades = trades.sum()
    total_wins = wins.sum()
    win_rate = total_wins / total_trades if total_trades > 0 else 0

    # Profit factor
    winning = returns[returns > 0].sum()
    losing = abs(returns[returns < 0].sum())
    profit_factor = winning / losing if losing > 0 else np.inf

    # Average daily return
    avg_daily_return = active_returns.mean() if len(active_returns) > 0 else 0
    avg_daily_return_gross = returns_gross[active_mask].mean() if active_mask.sum() > 0 else 0
    avg_daily_cost = costs[active_mask].mean() if active_mask.sum() > 0 else 0

    return {
        'total_return_net': total_return_net,
        'total_return_gross': total_return_gross,
        'total_costs': total_costs,
        'cumulative_returns': cumulative_returns,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_daily_return': avg_daily_return,
        'avg_daily_return_gross': avg_daily_return_gross,
        'avg_daily_cost': avg_daily_cost,
        'trading_days': active_mask.sum(),
        'total_trades': total_trades,
        'returns': returns,
    }


def find_breakeven_cost(preds, targets, mask, strategy='all', threshold=None):
    """Find the transaction cost at which strategy breaks even"""
    for cost_bps in range(1, 200):
        returns, _, _, trades, _ = simulate_trading_with_costs(
            preds, targets, mask,
            strategy=strategy,
            confidence_threshold=threshold,
            transaction_cost_bps=cost_bps
        )
        total_return = (1 + returns).prod() - 1
        if total_return <= 0:
            return cost_bps - 1
    return 200


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
    print("EXP6b: TRADING BACKTEST WITH TRANSACTION COSTS")
    print("=" * 70)

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
    print(f"Test period: ~{len(preds)/252:.1f} years")

    # Cost scenarios
    cost_scenarios = {
        'No Cost': 0,
        'HFT/Prime (3bp)': 3,
        'Institutional (10bp)': 10,
        'Retail Low (20bp)': 20,
        'Retail High (50bp)': 50,
    }

    strategies = {
        'all': None,
        'p50': 50,
        'p75': 75,
    }

    # Results table
    print("\n" + "=" * 70)
    print("RESULTS BY COST TIER")
    print("=" * 70)

    all_results = {}

    for strat_name, pct in strategies.items():
        if pct is not None:
            threshold = np.percentile(np.abs(preds[:, mask].flatten()), pct)
        else:
            threshold = None

        print(f"\n{'='*70}")
        print(f"Strategy: {strat_name.upper()}")
        print(f"{'='*70}")
        print(f"{'Cost Tier':<20} {'Net Return':>12} {'Sharpe':>10} {'Win Rate':>10} {'Profit F.':>10}")
        print("-" * 70)

        strat_results = {}

        for cost_name, cost_bps in cost_scenarios.items():
            returns, returns_gross, costs, trades, wins = simulate_trading_with_costs(
                preds, targets, mask,
                strategy='all' if pct is None else 'selective',
                confidence_threshold=threshold,
                transaction_cost_bps=cost_bps
            )

            metrics = calculate_metrics(returns, returns_gross, costs, trades, wins)
            strat_results[cost_name] = metrics

            net_ret = metrics['total_return_net'] * 100
            sharpe = metrics['sharpe_ratio']
            win_rate = metrics['win_rate'] * 100
            pf = metrics['profit_factor']

            # Color coding for profitability
            status = "✓" if net_ret > 0 else "✗"

            print(f"{cost_name:<20} {net_ret:>11.1f}% {sharpe:>10.2f} {win_rate:>9.1f}% {pf:>10.2f} {status}")

        # Find breakeven
        breakeven = find_breakeven_cost(
            preds, targets, mask,
            strategy='all' if pct is None else 'selective',
            threshold=threshold
        )
        print(f"\n>>> Break-even cost: {breakeven} bps")

        all_results[strat_name] = {
            'results': strat_results,
            'breakeven_bps': breakeven
        }

    # Detailed comparison at institutional cost (10bp)
    print("\n" + "=" * 70)
    print("DETAILED COMPARISON @ 10bp COST (INSTITUTIONAL)")
    print("=" * 70)

    print(f"\n{'Strategy':<10} {'Gross Ret':>12} {'Costs':>12} {'Net Ret':>12} {'Sharpe':>10} {'MDD':>10}")
    print("-" * 70)

    for strat_name, pct in strategies.items():
        r = all_results[strat_name]['results']['Institutional (10bp)']
        gross = r['total_return_gross'] * 100
        costs = r['total_costs'] * 100
        net = r['total_return_net'] * 100
        sharpe = r['sharpe_ratio']
        mdd = r['max_drawdown'] * 100

        print(f"{strat_name:<10} {gross:>11.1f}% {costs:>11.1f}% {net:>11.1f}% {sharpe:>10.2f} {mdd:>9.1f}%")

    # Cost impact analysis
    print("\n" + "=" * 70)
    print("COST IMPACT ANALYSIS")
    print("=" * 70)

    for strat_name in strategies.keys():
        no_cost = all_results[strat_name]['results']['No Cost']
        inst_cost = all_results[strat_name]['results']['Institutional (10bp)']

        sharpe_no = no_cost['sharpe_ratio']
        sharpe_inst = inst_cost['sharpe_ratio']
        sharpe_drop = (sharpe_no - sharpe_inst) / sharpe_no * 100 if sharpe_no > 0 else 0

        print(f"\n{strat_name}:")
        print(f"  Sharpe (no cost):    {sharpe_no:.2f}")
        print(f"  Sharpe (10bp cost):  {sharpe_inst:.2f}")
        print(f"  Sharpe degradation:  {sharpe_drop:.1f}%")
        print(f"  Break-even cost:     {all_results[strat_name]['breakeven_bps']} bps")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Cumulative returns comparison at 10bp
    ax = axes[0, 0]
    colors = {'all': 'blue', 'p50': 'green', 'p75': 'orange'}
    for strat_name in strategies.keys():
        cum_ret = all_results[strat_name]['results']['Institutional (10bp)']['cumulative_returns']
        sharpe = all_results[strat_name]['results']['Institutional (10bp)']['sharpe_ratio']
        ax.plot(cum_ret, label=f"{strat_name} (Sharpe={sharpe:.2f})", color=colors[strat_name])
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('Cumulative Returns @ 10bp Cost')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Sharpe vs Cost
    ax = axes[0, 1]
    cost_levels = [0, 3, 10, 20, 50]
    cost_names = ['No Cost', 'HFT/Prime (3bp)', 'Institutional (10bp)', 'Retail Low (20bp)', 'Retail High (50bp)']

    for strat_name in strategies.keys():
        sharpes = [all_results[strat_name]['results'][cn]['sharpe_ratio'] for cn in cost_names]
        ax.plot(cost_levels, sharpes, 'o-', label=strat_name, color=colors[strat_name])

    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    ax.axhline(y=1, color='g', linestyle='--', alpha=0.5, label='Sharpe=1')
    ax.set_xlabel('Transaction Cost (bps)')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio vs Transaction Cost')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Net Return vs Cost
    ax = axes[1, 0]
    for strat_name in strategies.keys():
        net_rets = [all_results[strat_name]['results'][cn]['total_return_net'] * 100 for cn in cost_names]
        ax.plot(cost_levels, net_rets, 'o-', label=strat_name, color=colors[strat_name])

    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    ax.set_xlabel('Transaction Cost (bps)')
    ax.set_ylabel('Total Return (%)')
    ax.set_title('Net Return vs Transaction Cost')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Break-even analysis
    ax = axes[1, 1]
    strat_names = list(strategies.keys())
    breakevens = [all_results[s]['breakeven_bps'] for s in strat_names]

    bars = ax.bar(strat_names, breakevens, color=[colors[s] for s in strat_names])
    ax.axhline(y=10, color='g', linestyle='--', label='Institutional (10bp)')
    ax.axhline(y=20, color='orange', linestyle='--', label='Retail Low (20bp)')
    ax.axhline(y=50, color='r', linestyle='--', label='Retail High (50bp)')

    for bar, be in zip(bars, breakevens):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{be}bp', ha='center', fontsize=10)

    ax.set_ylabel('Break-even Cost (bps)')
    ax.set_title('Break-even Transaction Cost by Strategy')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('exp6_trading_backtest/trading_with_costs.png', dpi=150)
    plt.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: PRACTICAL VIABILITY")
    print("=" * 70)

    inst_viable = sum(1 for s in strategies.keys()
                      if all_results[s]['results']['Institutional (10bp)']['sharpe_ratio'] > 1)
    retail_viable = sum(1 for s in strategies.keys()
                        if all_results[s]['results']['Retail Low (20bp)']['sharpe_ratio'] > 1)

    print(f"""
QUESTION: Is the strategy profitable after transaction costs?

1. AT INSTITUTIONAL LEVEL (10bp):
   - All strategies: {inst_viable}/{len(strategies)} have Sharpe > 1
   - p75 selective Sharpe: {all_results['p75']['results']['Institutional (10bp)']['sharpe_ratio']:.2f}
   - VERDICT: {"✓ VIABLE" if all_results['p75']['results']['Institutional (10bp)']['sharpe_ratio'] > 1 else "✗ NOT VIABLE"}

2. AT RETAIL LEVEL (20bp):
   - All strategies: {retail_viable}/{len(strategies)} have Sharpe > 1
   - p75 selective Sharpe: {all_results['p75']['results']['Retail Low (20bp)']['sharpe_ratio']:.2f}
   - VERDICT: {"✓ VIABLE" if all_results['p75']['results']['Retail Low (20bp)']['sharpe_ratio'] > 1 else "✗ MARGINAL"}

3. BREAK-EVEN COSTS:
   - all:  {all_results['all']['breakeven_bps']} bps
   - p50:  {all_results['p50']['breakeven_bps']} bps
   - p75:  {all_results['p75']['breakeven_bps']} bps

4. RECOMMENDATION:
   - Institutional/HFT: Strong viability with Sharpe > {all_results['p75']['results']['Institutional (10bp)']['sharpe_ratio']:.1f}
   - Retail: {"Viable" if all_results['p75']['results']['Retail Low (20bp)']['sharpe_ratio'] > 0.5 else "Not recommended"} (Sharpe {all_results['p75']['results']['Retail Low (20bp)']['sharpe_ratio']:.2f})
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
        'test_period_years': len(preds) / 252,
        'strategies': {}
    }

    for strat_name in strategies.keys():
        output['strategies'][strat_name] = {
            'breakeven_bps': all_results[strat_name]['breakeven_bps'],
            'by_cost': {}
        }
        for cost_name, metrics in all_results[strat_name]['results'].items():
            output['strategies'][strat_name]['by_cost'][cost_name] = convert({
                k: v for k, v in metrics.items()
                if k not in ['cumulative_returns', 'returns']
            })

    with open('exp6_trading_backtest/results_with_costs.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to exp6_trading_backtest/")


if __name__ == "__main__":
    main()
