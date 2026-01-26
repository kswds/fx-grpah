"""
Yoonsik Hong's FXRP Model (arxiv 2508.14784)
"Graph Learning for Foreign Exchange Rate Prediction and Statistical Arbitrage"

Key differences from our model:
1. Edge-level prediction (FX rate) vs Node-level (currency strength)
2. Only uses Interest Rates as node features
3. Currency Value features from MLE
4. Different GNN architecture (node + edge message passing)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class EdgeGNNLayer(nn.Module):
    """
    GNN layer that handles both node and edge features.
    Based on equations (10) and (11) in the paper.
    """
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        # Node update: aggregates from neighbors
        # n_i^l = (1/|N(i)|) * sum_j SLP([n_i^{l-1}; e_ji^{l-1}; n_j^{l-1}])
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Edge update: uses updated node features
        # e_ij^l = SLP([n_i^l; e_ij^{l-1}; n_j^l])
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

    def forward(self, node_feat, edge_feat, edge_index):
        """
        Args:
            node_feat: [N, node_dim]
            edge_feat: [E, edge_dim]
            edge_index: [2, E]
        Returns:
            new_node_feat: [N, hidden_dim]
            new_edge_feat: [E, hidden_dim]
        """
        src, dst = edge_index
        N = node_feat.size(0)
        E = edge_index.size(1)

        # Node update: aggregate from neighbors
        # For each node i, aggregate messages from all j where (j,i) is an edge
        # Message: [n_i, e_ji, n_j]

        # Create messages for each edge (j -> i)
        msg_input = torch.cat([
            node_feat[dst],      # n_i (destination)
            edge_feat,           # e_ji
            node_feat[src],      # n_j (source)
        ], dim=-1)

        messages = self.node_mlp(msg_input)  # [E, hidden_dim]

        # Aggregate messages for each node (mean aggregation)
        new_node_feat = torch.zeros(N, self.hidden_dim, device=node_feat.device)
        counts = torch.zeros(N, 1, device=node_feat.device)

        # For edge (src -> dst), the message goes to dst
        new_node_feat.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.hidden_dim), messages)
        counts.scatter_add_(0, dst.unsqueeze(-1), torch.ones(E, 1, device=node_feat.device))
        counts = counts.clamp(min=1)
        new_node_feat = new_node_feat / counts

        # Edge update: using new node features
        edge_input = torch.cat([
            new_node_feat[src],  # n_i^l (source)
            edge_feat,           # e_ij^{l-1}
            new_node_feat[dst],  # n_j^l (destination)
        ], dim=-1)

        new_edge_feat = self.edge_mlp(edge_input)

        return new_node_feat, new_edge_feat


class YoonsikFXRP(nn.Module):
    """
    Yoonsik's Foreign Exchange Rate Prediction model.

    Predicts FX rate changes at edge level using:
    - Node features: IR temporal changes, Currency Value temporal changes
    - Edge features: FX rate temporal changes
    """
    def __init__(self, n_ccy, node_feat_dim, edge_feat_dim, hidden_dim=64, n_layers=3):
        super().__init__()

        self.n_ccy = n_ccy
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Input projection
        self.node_input = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_input = nn.Linear(edge_feat_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            EdgeGNNLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])

        # Output layer (edge-level prediction)
        self.edge_output = nn.Linear(hidden_dim, 1)

    def forward(self, node_feat, edge_feat, edge_index):
        """
        Args:
            node_feat: [N, node_feat_dim] - IR and currency value features
            edge_feat: [E, edge_feat_dim] - FX rate features
            edge_index: [2, E] - edge connections
        Returns:
            pred: [E] - predicted FX rate log changes
        """
        # Project inputs
        h_node = self.node_input(node_feat)
        h_edge = self.edge_input(edge_feat)

        # GNN layers
        for layer in self.gnn_layers:
            h_node, h_edge = layer(h_node, h_edge, edge_index)

        # Edge-level output
        pred = self.edge_output(h_edge).squeeze(-1)

        return pred


def compute_currency_values(fx_rates, ccys):
    """
    Compute currency values via MLE (least squares).

    Solves:
        log V_i - log V_j = log X_ij  for all (i,j) edges
        (1/|C|) * sum(log V_i) = 0    (normalization)

    Args:
        fx_rates: dict {(i,j): X_ij} or matrix [N, N]
        ccys: list of currency names
    Returns:
        log_values: [N] log currency values
    """
    N = len(ccys)

    if isinstance(fx_rates, dict):
        # Build from dictionary
        equations = []
        targets = []
        for (i, j), rate in fx_rates.items():
            if i < j and rate > 0:
                eq = np.zeros(N)
                eq[i] = 1
                eq[j] = -1
                equations.append(eq)
                targets.append(np.log(rate))
    else:
        # Build from matrix
        equations = []
        targets = []
        for i in range(N):
            for j in range(i+1, N):
                if fx_rates[i, j] > 0:
                    eq = np.zeros(N)
                    eq[i] = 1
                    eq[j] = -1
                    equations.append(eq)
                    targets.append(np.log(fx_rates[i, j]))

    # Add normalization constraint
    norm_eq = np.ones(N) / N
    equations.append(norm_eq)
    targets.append(0)

    A = np.array(equations)
    b = np.array(targets)

    # Least squares solution
    log_values, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return log_values


class YoonsikDataProcessor:
    """
    Process data for Yoonsik's model.

    Features:
    - Node: IR temporal changes, Currency Value temporal changes
    - Edge: FX rate temporal changes
    """
    def __init__(self, ccys, lookback_windows=[1, 3, 5, 10, 15, 20]):
        self.ccys = ccys
        self.lookback_windows = lookback_windows
        self.n_ccy = len(ccys)

    def build_edge_index(self):
        """Build fully connected edge index (excluding self-loops)"""
        edges = []
        for i in range(self.n_ccy):
            for j in range(self.n_ccy):
                if i != j:
                    edges.append([i, j])
        return torch.tensor(edges, dtype=torch.long).T

    def compute_temporal_features(self, series, windows):
        """
        Compute average temporal log differences over multiple windows.

        Args:
            series: [T] time series
            windows: list of lookback windows
        Returns:
            features: [T, len(windows)]
        """
        T = len(series)
        features = np.zeros((T, len(windows)))

        log_series = np.log(series + 1e-10)
        log_diff = np.diff(log_series, prepend=log_series[0])

        for w_idx, w in enumerate(windows):
            for t in range(T):
                start = max(0, t - w + 1)
                features[t, w_idx] = log_diff[start:t+1].mean()

        return features

    def process_data(self, fx_matrix, ir_matrix):
        """
        Process FX and IR data into model inputs.

        Args:
            fx_matrix: [T, N, N] FX rates (X_tij)
            ir_matrix: [T, N] Interest rates (Y_t,1,i)
        Returns:
            node_features: [T, N, node_dim]
            edge_features: [T, E, edge_dim]
            targets: [T, E] FX rate log changes
        """
        T, N, _ = fx_matrix.shape
        E = N * (N - 1)  # fully connected

        # Build edge list
        edge_list = [(i, j) for i in range(N) for j in range(N) if i != j]

        # Compute currency values for each time step
        currency_values = np.zeros((T, N))
        for t in range(T):
            currency_values[t] = compute_currency_values(fx_matrix[t], self.ccys)

        # Node features: IR changes + Currency Value changes
        node_feat_list = []
        for i in range(N):
            ir_feat = self.compute_temporal_features(ir_matrix[:, i], self.lookback_windows)
            cv_feat = self.compute_temporal_features(np.exp(currency_values[:, i]), self.lookback_windows)
            node_feat = np.concatenate([ir_feat, cv_feat], axis=-1)  # [T, 2*len(windows)]
            node_feat_list.append(node_feat)

        node_features = np.stack(node_feat_list, axis=1)  # [T, N, 2*len(windows)]

        # Edge features: FX rate changes
        edge_feat_list = []
        for (i, j) in edge_list:
            fx_series = fx_matrix[:, i, j]
            fx_feat = self.compute_temporal_features(fx_series, self.lookback_windows)
            edge_feat_list.append(fx_feat)

        edge_features = np.stack(edge_feat_list, axis=1)  # [T, E, len(windows)]

        # Targets: FX rate log changes (t-1 to t)
        targets = np.zeros((T, E))
        for e_idx, (i, j) in enumerate(edge_list):
            log_fx = np.log(fx_matrix[:, i, j] + 1e-10)
            targets[:, e_idx] = np.diff(log_fx, prepend=log_fx[0])

        return node_features, edge_features, targets


class YoonsikTrainer:
    """Trainer for Yoonsik's model"""

    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 3e-4))

    def train_epoch(self, node_feat, edge_feat, targets, edge_index):
        """Train for one epoch"""
        self.model.train()

        node_feat = torch.tensor(node_feat, dtype=torch.float32, device=self.device)
        edge_feat = torch.tensor(edge_feat, dtype=torch.float32, device=self.device)
        targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        edge_index = edge_index.to(self.device)

        T = node_feat.size(0)
        batch_size = self.config.get('batch_size', 64)
        indices = np.random.permutation(T)

        total_loss = 0
        n_batches = 0

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            batch_idx = indices[start:end]

            batch_loss = 0
            for t in batch_idx:
                pred = self.model(node_feat[t], edge_feat[t], edge_index)
                loss = F.mse_loss(pred, targets[t])
                batch_loss += loss

            batch_loss = batch_loss / len(batch_idx)

            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += batch_loss.item()
            n_batches += 1

        return total_loss / n_batches

    def evaluate(self, node_feat, edge_feat, targets, edge_index):
        """Evaluate model"""
        self.model.eval()

        node_feat = torch.tensor(node_feat, dtype=torch.float32, device=self.device)
        edge_feat = torch.tensor(edge_feat, dtype=torch.float32, device=self.device)
        targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        edge_index = edge_index.to(self.device)

        T = node_feat.size(0)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for t in range(T):
                pred = self.model(node_feat[t], edge_feat[t], edge_index)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(targets[t].cpu().numpy())

        preds = np.array(all_preds)
        targets = np.array(all_targets)

        # MSE
        mse = ((preds - targets) ** 2).mean()

        # Direction accuracy (for each edge)
        correct = (np.sign(preds) == np.sign(targets))
        hit_rate = correct.mean()

        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'hit_rate': hit_rate,
            'preds': preds,
            'targets': targets,
        }


if __name__ == "__main__":
    # Test the model
    print("Testing Yoonsik's FXRP Model...")

    # Dummy data
    N = 10  # currencies
    T = 100  # time steps

    ccys = ['USD', 'EUR', 'JPY', 'GBP', 'CAD', 'AUD', 'CHF', 'NZD', 'SEK', 'NOK']

    # Random FX matrix
    fx_matrix = np.random.uniform(0.5, 2.0, (T, N, N))
    for t in range(T):
        for i in range(N):
            fx_matrix[t, i, i] = 1.0  # X_ii = 1

    # Random IR matrix
    ir_matrix = np.random.uniform(0.01, 0.05, (T, N))

    # Process data
    processor = YoonsikDataProcessor(ccys)
    node_feat, edge_feat, targets = processor.process_data(fx_matrix, ir_matrix)
    edge_index = processor.build_edge_index()

    print(f"Node features shape: {node_feat.shape}")  # [T, N, node_dim]
    print(f"Edge features shape: {edge_feat.shape}")  # [T, E, edge_dim]
    print(f"Targets shape: {targets.shape}")          # [T, E]

    # Create model
    model = YoonsikFXRP(
        n_ccy=N,
        node_feat_dim=node_feat.shape[-1],
        edge_feat_dim=edge_feat.shape[-1],
        hidden_dim=64,
        n_layers=3
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Test forward pass
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    node_t = torch.tensor(node_feat[0], dtype=torch.float32, device=device)
    edge_t = torch.tensor(edge_feat[0], dtype=torch.float32, device=device)
    edge_index = edge_index.to(device)

    pred = model(node_t, edge_t, edge_index)
    print(f"Prediction shape: {pred.shape}")  # [E]

    print("Test passed!")
