"""
Network Visualization for Currency & Macro Relationships

Reproduces:
  - causal_network.png           (macro→currency causal network from Ridge coefficients)
  - confidence_trading_graph.png (left: cumulative returns, right: pair reliability network)
  - trading_macro_network.png    (left: cumulative returns, right: currency+macro network)

Uses Ridge model coefficients to determine:
  - Macro → Currency links: Ridge coefficient magnitude per currency per macro factor
  - Currency pair reliability: confidence × accuracy per pair
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import Ridge
from itertools import combinations
from datetime import datetime

from config import Config
from dataset import load_data, build_features, normalize_data

MACRO_FEATURES = ["Global_VIX", "Global_Gold", "Global_Oil",
                  "Global_Copper", "Global_US2Y", "Global_IronOre"]
MACRO_SHORT = ["VIX", "Gold", "Oil", "Copper", "US2Y", "IronOre"]


def prepare_data_and_model():
    """Load data, train Ridge, return predictions, actuals, and model."""
    config = Config(seed=42, lookback=20)
    df = load_data(config)
    X_local_base, X_macro_full, Y = build_features(df, config)

    macro_idx = [config.global_features.index(f) for f in MACRO_FEATURES]
    X_macro = X_macro_full[:, macro_idx]

    n_total = len(X_local_base)
    split_idx = int(n_total * 0.8)
    X_local_scaled, X_macro_scaled, Y_raw, _ = normalize_data(
        X_local_base, X_macro, Y, train_idx=split_idx)

    L = config.lookback
    n = n_total - L
    split = int(n * 0.8)

    X_local_list, X_macro_list, Y_list = [], [], []
    for idx in range(n):
        X_local_list.append(X_local_scaled[idx + L - 1])
        X_macro_list.append(X_macro_scaled[idx + L - 1])
        Y_list.append(Y_raw[idx + L])

    X_local = np.stack(X_local_list)
    X_macro_arr = np.stack(X_macro_list)
    Y = np.stack(Y_list)

    X = np.concatenate([X_local.reshape(n, -1), X_macro_arr], axis=1)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, Y_train)
    pred = ridge.predict(X_test)

    return pred, Y_test, ridge, config


def get_macro_coefficients(ridge, config):
    """Extract macro→currency coefficient matrix from Ridge model.

    Ridge features: [N*3 local features, M macro features]
    Ridge coef: [N_ccy, N*3 + M] → last M columns are macro coefficients per currency.
    """
    n_local = config.n_ccy * 3  # 3 local features per currency
    n_macro = len(MACRO_FEATURES)
    # coef shape: [n_ccy, n_features]
    macro_coef = ridge.coef_[:, n_local:]  # [n_ccy, n_macro]
    return macro_coef


def compute_pair_reliability(pred, actual, config, exclude_usd=False):
    """Compute confidence × accuracy for each currency pair."""
    pairs = []
    for i, j in combinations(range(config.n_ccy), 2):
        if exclude_usd and (i == config.usd_idx or j == config.usd_idx):
            continue
        pairs.append((i, j))

    reliability = {}
    for i, j in pairs:
        pred_diff = pred[:, i] - pred[:, j]
        actual_diff = actual[:, i] - actual[:, j]
        confidence = np.mean(np.abs(pred_diff))
        accuracy = np.mean(np.sign(pred_diff) == np.sign(actual_diff))
        reliability[(i, j)] = confidence * accuracy

    return reliability, pairs


def confidence_trading_returns(pred, actual, config, exclude_usd=False):
    """Compute cumulative returns for different confidence filters."""
    pairs = []
    for i, j in combinations(range(config.n_ccy), 2):
        if exclude_usd and (i == config.usd_idx or j == config.usd_idx):
            continue
        pairs.append((i, j))

    pred_pairs = np.stack([pred[:, i] - pred[:, j] for i, j in pairs], axis=1)
    actual_pairs = np.stack([actual[:, i] - actual[:, j] for i, j in pairs], axis=1)

    T, P = pred_pairs.shape
    fracs = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
    results = {}

    for frac in fracs:
        n_select = max(1, int(P * frac))
        daily = np.zeros(T)
        for t in range(T):
            conf = np.abs(pred_pairs[t])
            top_idx = np.argsort(conf)[-n_select:]
            signals = np.sign(pred_pairs[t, top_idx])
            daily[t] = (signals * actual_pairs[t, top_idx]).mean()
        results[frac] = np.cumsum(daily) * 100
    return results


# ============ Plotting Functions ============

def _network_layout(config):
    """Fixed circular layout for currencies + grid for macros."""
    ccys = config.ccys
    n = len(ccys)
    pos = {}

    # Currencies in circle
    for i, c in enumerate(ccys):
        angle = 2 * np.pi * i / n - np.pi / 2
        pos[c] = (2.5 * np.cos(angle), 2.5 * np.sin(angle))

    # Macros in inner region
    macro_positions = [
        (-0.8, 0.6), (0.0, 0.8), (0.8, 0.6),
        (-0.8, -0.2), (0.0, 0.0), (0.8, -0.2),
    ]
    for i, m in enumerate(MACRO_SHORT):
        pos[m] = macro_positions[i]

    return pos


HIGH_SIGNAL_CCYS = {"NOK", "NZD", "SEK"}


def plot_causal_network(ridge, config, output_name=None):
    """Plot macro→currency causal network (top 30% reliable pairs)."""
    macro_coef = get_macro_coefficients(ridge, config)  # [n_ccy, n_macro]
    pos = _network_layout(config)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Determine threshold for top 30%
    all_abs = np.abs(macro_coef).flatten()
    threshold = np.percentile(all_abs, 70)

    # Draw macro→currency edges
    for i, c in enumerate(config.ccys):
        for j, m in enumerate(MACRO_SHORT):
            coef = macro_coef[i, j]
            if np.abs(coef) < threshold:
                continue
            x0, y0 = pos[m]
            x1, y1 = pos[c]
            color = '#e74c3c' if coef > 0 else '#3498db'
            alpha = min(1.0, np.abs(coef) / np.max(all_abs) * 1.5)
            width = np.abs(coef) / np.max(all_abs) * 2.5
            style = '-' if coef > 0 else '--'
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        alpha=alpha, lw=width, linestyle=style))

    # Draw currency-currency strong pairs (from coefficient correlation)
    ccy_corr = np.corrcoef(macro_coef)
    corr_threshold = np.percentile(np.abs(ccy_corr[np.triu_indices(config.n_ccy, k=1)]), 70)
    for i in range(config.n_ccy):
        for j in range(i+1, config.n_ccy):
            if np.abs(ccy_corr[i, j]) > corr_threshold:
                c1, c2 = config.ccys[i], config.ccys[j]
                x0, y0 = pos[c1]
                x1, y1 = pos[c2]
                ax.plot([x0, x1], [y0, y1], color='#2ecc71', alpha=0.4,
                        linewidth=1.5, linestyle='-')

    # Draw nodes
    for c in config.ccys:
        x, y = pos[c]
        color = '#e74c3c' if c in HIGH_SIGNAL_CCYS else '#1abc9c'
        circle = plt.Circle((x, y), 0.3, color=color, ec='white', lw=2, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, c, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=6)

    for m in MACRO_SHORT:
        x, y = pos[m]
        rect = plt.Rectangle((x-0.4, y-0.2), 0.8, 0.4, color='#e67e22',
                              ec='white', lw=1.5, zorder=5)
        ax.add_patch(rect)
        ax.text(x, y, m, ha='center', va='center', fontsize=7,
                fontweight='bold', color='white', zorder=6)

    ax.set_title("Currency & Macro Causal Network\n"
                 "(Only top 30% reliable pairs shown, arrows indicate prediction direction)",
                 fontsize=13, fontweight='bold')

    # Legend
    legend_items = [
        mpatches.Patch(color='#e74c3c', label=f'High-signal CCY ({", ".join(HIGH_SIGNAL_CCYS)})'),
        mpatches.Patch(color='#1abc9c', label='Other CCY'),
        mpatches.Patch(color='#e67e22', label='Macro Factor'),
        plt.Line2D([0], [0], color='#e74c3c', lw=2, label='Macro→CCY (+)'),
        plt.Line2D([0], [0], color='#3498db', lw=2, linestyle='--', label='Macro→CCY (-)'),
        plt.Line2D([0], [0], color='#2ecc71', lw=1.5, label='Strong CCY↔CCY'),
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    if output_name:
        out_path = os.path.join(os.path.dirname(__file__), '..', 'results', output_name)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
    plt.close()


def plot_confidence_trading_graph(pred, actual, config, output_name=None):
    """Plot left: cumulative returns, right: pair reliability network."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Cumulative returns
    results = confidence_trading_returns(pred, actual, config, exclude_usd=False)
    fracs = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(fracs)))
    n_pairs = 55

    for i, frac in enumerate(fracs):
        label = f"All ({n_pairs} pairs)" if frac == 1.0 else f"Top {int(frac*100)}%"
        ax1.plot(results[frac], label=label, color=colors[i], linewidth=1.2)

    ax1.set_title("Confidence-Filtered Trading Returns", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: Pair reliability network
    reliability, pairs = compute_pair_reliability(pred, actual, config)
    rel_values = np.array(list(reliability.values()))
    rel_max = rel_values.max()

    # Circular layout
    n = config.n_ccy
    pos = {}
    for i, c in enumerate(config.ccys):
        angle = 2 * np.pi * i / n - np.pi / 2
        pos[c] = (2.5 * np.cos(angle), 2.5 * np.sin(angle))

    ax2.set_xlim(-3.5, 3.5)
    ax2.set_ylim(-3.5, 3.5)
    ax2.set_aspect('equal')
    ax2.axis('off')

    # Draw edges (color and width by reliability)
    cmap = plt.cm.Reds
    for (i, j), rel in reliability.items():
        c1, c2 = config.ccys[i], config.ccys[j]
        x0, y0 = pos[c1]
        x1, y1 = pos[c2]
        norm_rel = rel / rel_max
        color = cmap(0.3 + 0.7 * norm_rel)
        width = 0.5 + 3.5 * norm_rel
        ax2.plot([x0, x1], [y0, y1], color=color, alpha=0.6, linewidth=width)

    # Draw nodes
    for c in config.ccys:
        x, y = pos[c]
        circle = plt.Circle((x, y), 0.3, color='#1abc9c', ec='white', lw=2, zorder=5)
        ax2.add_patch(circle)
        ax2.text(x, y, c, ha='center', va='center', fontsize=9,
                 fontweight='bold', color='white', zorder=6)

    ax2.set_title("Currency Pair Reliability\n(darker/thicker = higher confidence × accuracy)",
                  fontsize=11, fontweight='bold')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(rel_values.min(), rel_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, fraction=0.04, pad=0.05)
    cbar.set_label("Confidence × Accuracy", fontsize=8)

    plt.tight_layout()
    if output_name:
        out_path = os.path.join(os.path.dirname(__file__), '..', 'results', output_name)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
    plt.close()


def plot_trading_macro_network(pred, actual, ridge, config, output_name=None):
    """Plot left: confidence-filtered returns, right: currency + macro network."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: same as confidence trading
    results = confidence_trading_returns(pred, actual, config, exclude_usd=False)
    fracs = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(fracs)))
    n_pairs = 55

    for i, frac in enumerate(fracs):
        label = f"All ({n_pairs} pairs)" if frac == 1.0 else f"Top {int(frac*100)}%"
        ax1.plot(results[frac], label=label, color=colors[i], linewidth=1.2)

    ax1.set_title("Confidence-Filtered Trading Returns", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: currency + macro network
    macro_coef = get_macro_coefficients(ridge, config)
    reliability, pairs = compute_pair_reliability(pred, actual, config)
    rel_values = np.array(list(reliability.values()))
    rel_max = rel_values.max()
    rel_threshold = np.percentile(rel_values, 50)

    pos = _network_layout(config)

    ax2.set_xlim(-3.5, 3.5)
    ax2.set_ylim(-3.5, 3.5)
    ax2.set_aspect('equal')
    ax2.axis('off')

    # Currency pair edges (strong ones)
    cmap = plt.cm.Reds
    for (i, j), rel in reliability.items():
        c1, c2 = config.ccys[i], config.ccys[j]
        x0, y0 = pos[c1]
        x1, y1 = pos[c2]
        norm_rel = rel / rel_max
        if rel > rel_threshold:
            color = cmap(0.3 + 0.7 * norm_rel)
            width = 0.5 + 3.0 * norm_rel
            label = "Strong CCY pair" if (i, j) == pairs[0] else None
        else:
            color = cmap(0.15)
            width = 0.3
            label = "Weak CCY pair" if (i, j) == pairs[-1] else None
        ax2.plot([x0, x1], [y0, y1], color=color, alpha=0.4, linewidth=width)

    # Macro→CCY links (top connections only)
    all_abs = np.abs(macro_coef).flatten()
    threshold = np.percentile(all_abs, 70)
    for i, c in enumerate(config.ccys):
        for j, m in enumerate(MACRO_SHORT):
            if np.abs(macro_coef[i, j]) < threshold:
                continue
            x0, y0 = pos[m]
            x1, y1 = pos[c]
            ax2.plot([x0, x1], [y0, y1], color='#3498db', alpha=0.5,
                     linewidth=1.0, linestyle='--')

    # Draw nodes
    for c in config.ccys:
        x, y = pos[c]
        color = '#e74c3c' if c in HIGH_SIGNAL_CCYS else '#1abc9c'
        circle = plt.Circle((x, y), 0.3, color=color, ec='white', lw=2, zorder=5)
        ax2.add_patch(circle)
        ax2.text(x, y, c, ha='center', va='center', fontsize=8,
                 fontweight='bold', color='white', zorder=6)

    for m in MACRO_SHORT:
        x, y = pos[m]
        rect = plt.Rectangle((x-0.35, y-0.18), 0.7, 0.36, color='#e67e22',
                              ec='white', lw=1.5, zorder=5)
        ax2.add_patch(rect)
        ax2.text(x, y, m, ha='center', va='center', fontsize=7,
                 fontweight='bold', color='white', zorder=6)

    ax2.set_title("Currency + Macro Network\n"
                  "(darker = higher confidence × accuracy)",
                  fontsize=11, fontweight='bold')

    legend_items = [
        mpatches.Patch(color='#e74c3c', label=f'High-signal CCY ({", ".join(HIGH_SIGNAL_CCYS)})'),
        mpatches.Patch(color='#1abc9c', label='Other CCY'),
        mpatches.Patch(color='#e67e22', label='Macro Factor'),
        plt.Line2D([0], [0], color='#c0392b', lw=2.5, label='Strong CCY pair'),
        plt.Line2D([0], [0], color='#e8c6c6', lw=0.5, label='Weak CCY pair'),
        plt.Line2D([0], [0], color='#3498db', lw=1, linestyle='--', label='Macro↔CCY link'),
    ]
    ax2.legend(handles=legend_items, loc='lower left', fontsize=7, framealpha=0.9)

    plt.tight_layout()
    if output_name:
        out_path = os.path.join(os.path.dirname(__file__), '..', 'results', output_name)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
    plt.close()


def save_network_results(ridge, pred, actual, config):
    """Save Ridge coefficients, pair reliability, and macro→currency links as JSON."""
    macro_coef = get_macro_coefficients(ridge, config)
    reliability, pairs = compute_pair_reliability(pred, actual, config)

    results = {
        'timestamp': datetime.now().isoformat(),
        'macro_to_currency_coefficients': {},
        'pair_reliability': {},
        'macro_features': MACRO_SHORT,
        'currencies': config.ccys,
    }

    # Macro → Currency coefficient matrix
    for i, c in enumerate(config.ccys):
        results['macro_to_currency_coefficients'][c] = {
            m: round(float(macro_coef[i, j]), 6)
            for j, m in enumerate(MACRO_SHORT)
        }

    # Pair reliability (confidence × accuracy)
    for (i, j), rel in sorted(reliability.items(), key=lambda x: x[1], reverse=True):
        pair_name = f"{config.ccys[i]}/{config.ccys[j]}"
        pred_diff = pred[:, i] - pred[:, j]
        actual_diff = actual[:, i] - actual[:, j]
        results['pair_reliability'][pair_name] = {
            'reliability': round(float(rel), 6),
            'hit_rate': round(float(np.mean(np.sign(pred_diff) == np.sign(actual_diff))), 4),
            'avg_confidence': round(float(np.mean(np.abs(pred_diff))), 6),
        }

    # Currency-currency similarity (macro coefficient correlation)
    ccy_corr = np.corrcoef(macro_coef)
    results['currency_macro_similarity'] = {}
    for i in range(config.n_ccy):
        for j in range(i+1, config.n_ccy):
            pair_name = f"{config.ccys[i]}/{config.ccys[j]}"
            results['currency_macro_similarity'][pair_name] = round(float(ccy_corr[i, j]), 4)

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'eval_network_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    return results


def main():
    print("=" * 60)
    print("Network Visualization Evaluation")
    print("=" * 60)

    pred, Y_test, ridge, config = prepare_data_and_model()
    print(f"Test period: {len(Y_test)} days, {config.n_ccy} currencies")

    # 1. Causal network
    print("\n[1/4] Causal network...")
    plot_causal_network(ridge, config, output_name="causal_network.png")

    # 2. Confidence trading + reliability graph
    print("[2/4] Confidence trading graph...")
    plot_confidence_trading_graph(pred, Y_test, config,
                                 output_name="confidence_trading_graph.png")

    # 3. Trading + macro network
    print("[3/4] Trading macro network...")
    plot_trading_macro_network(pred, Y_test, ridge, config,
                              output_name="trading_macro_network.png")

    # 4. Save numerical results
    print("[4/4] Saving network results...")
    save_network_results(ridge, pred, Y_test, config)

    print("\nDone.")


if __name__ == "__main__":
    main()
