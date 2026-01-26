"""
Exp2: SOTA Model Comparison

Compares our FXStrengthGNN against:
- Basic baselines (MLP, GRU)
- Tuned SOTA models (iTransformer, PatchTST, TimeMixer)

Results: exp/exp2_sota_comparison/results.json
"""
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import json
from datetime import datetime
import torch
import torch.nn as nn

from config import Config
from dataset import create_dataloaders, fully_connected_edge_index
from models import BASELINE_MODELS, FXStrengthGNN
from train import Trainer
from utils import set_seed, get_device, save_results

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# Tuned SOTA models
class PatchTSTTuned(nn.Module):
    """PatchTST with tuned params: patch_len=4, stride=2, depth=2, dim=64"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_ccy = config.n_ccy
        self.usd_idx = config.usd_idx
        self.hidden = 64
        self.scales = [1, 2, 4]

        input_dim = config.local_dim + config.macro_dim
        patch_len, stride = 4, 2
        num_patches = (config.lookback - patch_len) // stride + 1

        self.patch_len = patch_len
        self.stride = stride
        self.patch_embed = nn.Linear(patch_len * input_dim, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(64 * num_patches, 1)
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim), requires_grad=False)

    def forward(self, xl, xm, edge_index_single):
        B, L, N, local_dim = xl.shape
        xm_exp = xm.unsqueeze(2).expand(-1, -1, N, -1)
        x = torch.cat([xl, xm_exp], dim=-1)
        input_dim = x.size(-1)
        x = x.permute(0, 2, 1, 3).reshape(B * N, L, input_dim)

        patches = []
        for i in range(0, L - self.patch_len + 1, self.stride):
            patch = x[:, i:i + self.patch_len, :].reshape(B * N, -1)
            patches.append(patch)
        patches = torch.stack(patches, dim=1)
        patches = self.patch_embed(patches)
        out = self.transformer(patches)
        out = out.reshape(B * N, -1)
        ds = self.head(out).squeeze(-1)
        ds = ds.view(B, N)
        ds = ds - ds.mean(dim=1, keepdim=True)
        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]
        z_ccy = torch.zeros(B, N, self.hidden, device=xl.device)
        m_msg = torch.zeros(B, N, self.hidden, device=xl.device)
        return rhat, ds, z_ccy, m_msg


class TimeMixerTuned(nn.Module):
    """TimeMixer with tuned params: dim=64, scales=[1,2,4]"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_ccy = config.n_ccy
        self.usd_idx = config.usd_idx
        self.hidden = 64
        self.scales = [1, 2, 4]

        input_dim = config.local_dim + config.macro_dim
        L = config.lookback

        self.time_mixers = nn.ModuleList([
            nn.Sequential(nn.Linear(L // s, 64), nn.GELU(), nn.Linear(64, L // s))
            for s in self.scales
        ])
        self.feature_mixer = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, input_dim)
        )
        self.proj = nn.Linear(input_dim * sum(L // s for s in self.scales), 1)
        self.A = nn.Parameter(torch.zeros(config.n_ccy, config.macro_dim), requires_grad=False)

    def forward(self, xl, xm, edge_index_single):
        B, L, N, local_dim = xl.shape
        xm_exp = xm.unsqueeze(2).expand(-1, -1, N, -1)
        x = torch.cat([xl, xm_exp], dim=-1)
        input_dim = x.size(-1)
        x = x.permute(0, 2, 1, 3).reshape(B * N, L, input_dim)

        outputs = []
        for i, scale in enumerate(self.scales):
            x_scale = x[:, ::scale, :]
            x_t = x_scale.permute(0, 2, 1)
            x_t = self.time_mixers[i](x_t)
            x_t = x_t.permute(0, 2, 1)
            x_f = self.feature_mixer(x_t)
            outputs.append(x_f.reshape(B * N, -1))

        out = torch.cat(outputs, dim=-1)
        ds = self.proj(out).squeeze(-1)
        ds = ds.view(B, N)
        ds = ds - ds.mean(dim=1, keepdim=True)
        rhat = ds - ds[:, self.usd_idx:self.usd_idx + 1]
        z_ccy = torch.zeros(B, N, self.hidden, device=xl.device)
        m_msg = torch.zeros(B, N, self.hidden, device=xl.device)
        return rhat, ds, z_ccy, m_msg


TUNED_MODELS = {
    "patchtst": PatchTSTTuned,
    "timemixer": TimeMixerTuned,
}


def run_model(model_name, model_cls, config, device, edge_index, seed=42):
    """Train and evaluate a single model"""
    set_seed(seed)
    train_loader, test_loader = create_dataloaders(config, macro_mode="real")
    model = model_cls(config)
    trainer = Trainer(model, config, device)
    metrics = trainer.train(train_loader, test_loader, edge_index, label=model_name.upper())
    return metrics


def main():
    print("=" * 70)
    print("EXP2: SOTA MODEL COMPARISON")
    print("=" * 70)

    seed = 42
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    config = Config(seed=seed, epochs=30, batch_size=128, lr=3e-4)
    edge_index = fully_connected_edge_index(config.n_ccy).to(device)

    results = {}

    # Basic baselines
    for name in ["mlp", "gru"]:
        print(f"\n>>> Running {name}...")
        metrics = run_model(name, BASELINE_MODELS[name], config, device, edge_index, seed)
        results[name] = {
            "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
            "hit": float(metrics["hit"]),
        }

    # Tuned SOTA
    for name, cls in TUNED_MODELS.items():
        print(f"\n>>> Running {name}...")
        metrics = run_model(name, cls, config, device, edge_index, seed)
        results[name] = {
            "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
            "hit": float(metrics["hit"]),
        }

    # Ours
    print(f"\n>>> Running ours (FXStrengthGNN)...")
    metrics = run_model("ours", FXStrengthGNN, config, device, edge_index, seed)
    results["ours"] = {
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "hit": float(metrics["hit"]),
    }

    # Summary
    print("\n" + "=" * 70)
    print("SOTA COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Model':<15} {'RMSE':>10} {'MAE':>10} {'Hit Rate':>10}")
    print("-" * 45)

    model_order = ["mlp", "gru", "patchtst", "timemixer", "ours"]
    for name in model_order:
        r = results[name]
        print(f"{name:<15} {r['rmse']:>10.4f} {r['mae']:>10.4f} {r['hit']:>10.1%}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "seed": seed,
        },
        "results": results,
    }

    save_results(output, OUTPUT_DIR, "results.json")
    print(f"\nResults saved to {OUTPUT_DIR}/results.json")


if __name__ == "__main__":
    main()
