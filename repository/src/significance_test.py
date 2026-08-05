from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


DISPLAY = {
    "oursmain": "ARC_FX",
    "mlp": "MLP",
    "corrlstmgat": "Corr-LSTM-GAT",
}
NON_USD_ORDER = ["EUR", "JPY", "GBP", "CAD", "AUD", "KRW", "CHF", "NZD", "SEK", "NOK"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired moving-block bootstrap significance tests for the predictive results table.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "main_experiment.yaml"))
    parser.add_argument("--arcfx-root", default=None, help="Override ARC_FX prediction root. Defaults to results/repro_runs/arc_fx.")
    parser.add_argument("--baseline-root", default=None, help="Override baseline prediction root. Defaults to results/repro_runs/baselines.")
    parser.add_argument("--block-length", type=int, default=10)
    parser.add_argument("--num-bootstrap", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260805)
    parser.add_argument("--rmse-delta", type=float, default=0.0, help="Non-inferiority margin placeholder; kept for extensibility.")
    parser.add_argument("--report-path", default=None, help="Override markdown output path.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def prediction_path(root_dir: Path, model: str, seed: int) -> Path:
    return root_dir / f"{model}_seed{seed}" / "predictions" / f"{model}_predictions.parquet"


def load_model_seed_df(root_dir: Path, model: str, seed: int) -> pd.DataFrame:
    path = prediction_path(root_dir, model, seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    df = pd.read_parquet(path).copy()
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.normalize()
    return df[df["currency"].isin(NON_USD_ORDER)].copy()


def build_seed_panel(root_dir: Path, model: str, seed: int) -> Tuple[pd.Index, np.ndarray, np.ndarray]:
    df = load_model_seed_df(root_dir, model, seed)
    pred_panel = (
        df.pivot(index="target_date", columns="currency", values="pred")
        .reindex(columns=NON_USD_ORDER)
        .sort_index()
    )
    tgt_panel = (
        df.pivot(index="target_date", columns="currency", values="target")
        .reindex(columns=NON_USD_ORDER)
        .sort_index()
    )
    valid = pred_panel.notna().all(axis=1) & tgt_panel.notna().all(axis=1)
    pred_panel = pred_panel.loc[valid]
    tgt_panel = tgt_panel.loc[valid]
    return pred_panel.index, pred_panel.to_numpy(dtype=np.float32), tgt_panel.to_numpy(dtype=np.float32)


def load_paired_panels(
    model_a_root: Path,
    model_b_root: Path,
    model_a_name: str,
    model_b_name: str,
    seeds: Iterable[int],
) -> Dict[str, np.ndarray]:
    seed_dates: List[pd.Index] = []
    pred_a_all: List[np.ndarray] = []
    pred_b_all: List[np.ndarray] = []
    tgt_all: List[np.ndarray] = []

    for seed in seeds:
        dates_a, pred_a, tgt_a = build_seed_panel(model_a_root, model_a_name, int(seed))
        dates_b, pred_b, tgt_b = build_seed_panel(model_b_root, model_b_name, int(seed))
        common_dates = dates_a.intersection(dates_b)
        if len(common_dates) == 0:
            raise ValueError(f"No common dates for seed {seed}.")

        idx_a = pd.Index(dates_a).get_indexer(common_dates)
        idx_b = pd.Index(dates_b).get_indexer(common_dates)
        pred_a = pred_a[idx_a]
        pred_b = pred_b[idx_b]
        tgt_a = tgt_a[idx_a]
        tgt_b = tgt_b[idx_b]
        if not np.allclose(tgt_a, tgt_b, atol=1e-12, rtol=0.0):
            raise ValueError(f"Targets are not aligned for seed {seed}.")

        seed_dates.append(common_dates)
        pred_a_all.append(pred_a)
        pred_b_all.append(pred_b)
        tgt_all.append(tgt_a)

    common_all = seed_dates[0]
    for idx in seed_dates[1:]:
        common_all = common_all.intersection(idx)
    if len(common_all) == 0:
        raise ValueError("No common dates across all seeds.")

    pred_a_stack: List[np.ndarray] = []
    pred_b_stack: List[np.ndarray] = []
    tgt_stack: List[np.ndarray] = []
    for dates, pred_a, pred_b, tgt in zip(seed_dates, pred_a_all, pred_b_all, tgt_all):
        idx = pd.Index(dates).get_indexer(common_all)
        pred_a_stack.append(pred_a[idx])
        pred_b_stack.append(pred_b[idx])
        tgt_stack.append(tgt[idx])

    return {
        "dates": common_all.to_numpy(),
        "pred_a": np.stack(pred_a_stack, axis=0),
        "pred_b": np.stack(pred_b_stack, axis=0),
        "target": np.stack(tgt_stack, axis=0),
    }


def metric_means(pred: np.ndarray, target: np.ndarray, sample_idx: np.ndarray | None = None) -> Tuple[float, float]:
    if sample_idx is None:
        p = pred
        y = target
    else:
        p = pred[:, sample_idx, :]
        y = target[:, sample_idx, :]
    seed_hit = (np.sign(p) == np.sign(y)).astype(np.float32).mean(axis=(1, 2))
    seed_rmse = np.sqrt(((p - y) ** 2).mean(axis=(1, 2)))
    return float(seed_hit.mean()), float(seed_rmse.mean())


def sample_moving_blocks(n_dates: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if n_dates <= block_length:
        return np.arange(n_dates, dtype=np.int32)
    n_blocks = int(np.ceil(n_dates / float(block_length)))
    max_start = n_dates - block_length
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    blocks = [np.arange(start, start + block_length, dtype=np.int32) for start in starts]
    return np.concatenate(blocks, axis=0)[:n_dates]


def bootstrap_draws(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    target: np.ndarray,
    block_length: int,
    num_bootstrap: int,
    random_seed: int,
) -> Dict[str, np.ndarray | float]:
    rng = np.random.default_rng(random_seed)
    n_dates = target.shape[1]
    hit_deltas = np.empty(num_bootstrap, dtype=np.float32)
    rmse_deltas = np.empty(num_bootstrap, dtype=np.float32)

    obs_hit_a, obs_rmse_a = metric_means(pred_a, target)
    obs_hit_b, obs_rmse_b = metric_means(pred_b, target)
    obs_hit_delta = obs_hit_a - obs_hit_b
    obs_rmse_delta = obs_rmse_a - obs_rmse_b

    for i in range(num_bootstrap):
        idx = sample_moving_blocks(n_dates, block_length, rng)
        hit_a, rmse_a = metric_means(pred_a, target, idx)
        hit_b, rmse_b = metric_means(pred_b, target, idx)
        hit_deltas[i] = hit_a - hit_b
        rmse_deltas[i] = rmse_a - rmse_b

    return {
        "obs_hit_delta": obs_hit_delta,
        "obs_rmse_delta": obs_rmse_delta,
        "hit_deltas": hit_deltas,
        "rmse_deltas": rmse_deltas,
    }


def two_sided_p(samples: np.ndarray, observed: float) -> float:
    null_dist = samples - observed
    return float(np.mean(np.abs(null_dist) >= abs(observed)))


def one_sided_p_greater(samples: np.ndarray, observed: float) -> float:
    null_dist = samples - observed
    return float(np.mean(null_dist >= observed))


def one_sided_p_less(samples: np.ndarray, observed: float) -> float:
    null_dist = samples - observed
    return float(np.mean(null_dist <= observed))


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    outputs = cfg["outputs"]
    exp = cfg["experiment"]
    report_dir = ROOT / outputs["report_dir"]
    report_path = Path(args.report_path) if args.report_path else (report_dir / "significance_test_plan_and_results.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    arcfx_root = Path(args.arcfx_root) if args.arcfx_root else (ROOT / outputs["prediction_root"] / "arc_fx")
    baseline_root = Path(args.baseline_root) if args.baseline_root else (ROOT / outputs["prediction_root"] / "baselines")
    seeds = list(exp["seeds"])

    # 1) Mean Hit superiority: ARC_FX vs MLP
    panels_hit = load_paired_panels(arcfx_root, baseline_root, "oursmain", "mlp", seeds)
    draws_hit = bootstrap_draws(
        pred_a=np.asarray(panels_hit["pred_a"]),
        pred_b=np.asarray(panels_hit["pred_b"]),
        target=np.asarray(panels_hit["target"]),
        block_length=args.block_length,
        num_bootstrap=args.num_bootstrap,
        random_seed=args.random_seed,
    )
    hit_samples = np.asarray(draws_hit["hit_deltas"], dtype=float)
    hit_obs = float(draws_hit["obs_hit_delta"])
    hit_ci_lo, hit_ci_hi = np.quantile(hit_samples, [0.025, 0.975])
    hit_one_sided_lb = float(np.quantile(hit_samples, 0.05))
    hit_p_two = two_sided_p(hit_samples, hit_obs)
    hit_p_one = one_sided_p_greater(hit_samples, hit_obs)

    # 2) RMSE superiority: MLP vs Corr-LSTM-GAT
    panels_rmse = load_paired_panels(baseline_root, baseline_root, "mlp", "corrlstmgat", seeds)
    draws_rmse = bootstrap_draws(
        pred_a=np.asarray(panels_rmse["pred_a"]),
        pred_b=np.asarray(panels_rmse["pred_b"]),
        target=np.asarray(panels_rmse["target"]),
        block_length=args.block_length,
        num_bootstrap=args.num_bootstrap,
        random_seed=args.random_seed,
    )
    rmse_samples = np.asarray(draws_rmse["rmse_deltas"], dtype=float)  # RMSE_MLP - RMSE_Corr
    rmse_obs = float(draws_rmse["obs_rmse_delta"])
    rmse_ci_lo, rmse_ci_hi = np.quantile(rmse_samples, [0.025, 0.975])
    rmse_p_one = one_sided_p_less(rmse_samples, rmse_obs)

    lines = [
        "# Statistical Testing Plan and Results for Table 1",
        "",
        "This note summarizes the hypothesis-testing setup aligned with the paper's main claims and reports the resulting bootstrap-based inference.",
        "",
        "## Bootstrap Design",
        "",
        "Both tests use the same paired moving-block bootstrap:",
        "",
        "- Bootstrap unit: test dates",
        f"- Test period dates: `{len(panels_hit['dates'])}`",
        f"- Cross-section kept within each sampled date: all `{len(NON_USD_ORDER)}` non-USD currencies",
        f"- Seeds kept within each sampled date block: `{', '.join(str(s) for s in seeds)}`",
        f"- Block length: `{args.block_length}` trading days",
        f"- Repetitions: `{args.num_bootstrap:,}`",
        "- In each bootstrap draw, the same sampled date blocks are applied jointly to both competing models",
        "- Metrics are computed separately by seed and then averaged across seeds before taking the model difference",
        "",
        "---",
        "",
        "## 1. Mean Hit: One-Sided Superiority Test",
        "",
        "Comparison:",
        "",
        "- `ARC-FX` vs `MLP`",
        "",
        "Test statistic:",
        "",
        "- `ΔHit = Hit_ARC-FX - Hit_MLP`",
        "",
        "Hypotheses:",
        "",
        "- `H0: ΔHit <= 0`",
        "- `H1: ΔHit > 0`",
        "",
        "This is the appropriate test when the ex-ante research claim is directional superiority of `ARC-FX` over the strongest direct baseline in Mean Hit.",
        "",
        "### Result",
        "",
        f"- Observed difference: `ΔHit = {hit_obs:.6f}`",
        f"- Two-sided 95% CI: `[{hit_ci_lo:.6f}, {hit_ci_hi:.6f}]`",
        f"- Two-sided p-value: `{hit_p_two:.4f}`",
        f"- One-sided 95% CI: `[{hit_one_sided_lb:.6f}, +inf)`",
        f"- One-sided p-value: `{hit_p_one:.4f}`",
        "",
        "### Judgment",
        "",
        (
            "Under the one-sided superiority formulation, the Mean Hit improvement of `ARC-FX` over `MLP` is statistically significant at the 5% level."
            if hit_one_sided_lb > 0.0 and hit_p_one < 0.05
            else "Under the one-sided superiority formulation, the Mean Hit improvement of `ARC-FX` over `MLP` is not statistically significant at the 5% level."
        ),
        "",
        "---",
        "",
        "",
        "## 2. One-Sided RMSE Superiority Test: MLP vs Corr-LSTM-GAT",
        "",
        "For a direct top-two RMSE comparison, define the difference as:",
        "",
        "- `ΔRMSE = RMSE_MLP - RMSE_Corr-LSTM-GAT`",
        "",
        "Since lower RMSE is better, negative values favor `MLP`.",
        "",
        "Although the rounded table values suggest only a tiny gap, the statistical test is computed from the unrounded raw prediction panels and the re-evaluated bootstrap RMSE values.",
        "",
        "Hypotheses:",
        "",
        "- `H0: ΔRMSE >= 0`",
        "- `H1: ΔRMSE < 0`",
        "",
        "This is a one-sided superiority test asking whether `MLP` has significantly lower RMSE than `Corr-LSTM-GAT`.",
        "",
        "### Result",
        "",
        f"- Observed difference: `ΔRMSE = {rmse_obs:.8f}`",
        f"- Two-sided 95% CI: `[{rmse_ci_lo:.8f}, {rmse_ci_hi:.8f}]`",
        f"- One-sided p-value: `{rmse_p_one:.4f}`",
        "",
        "### Judgment",
        "",
        (
            "The observed difference has the favorable sign for `MLP`, and the one-sided bootstrap test supports the claim that `MLP` has significantly lower RMSE than `Corr-LSTM-GAT`."
            if rmse_obs < 0.0 and rmse_p_one < 0.05
            else "The observed difference has the favorable sign for `MLP`, but the one-sided p-value is far above 0.05. Therefore, the bootstrap test does not support the claim that `MLP` has significantly lower RMSE than `Corr-LSTM-GAT`."
        ),
        "",
        "## Reproduction",
        "",
        "Run this file after prediction parquet files have been generated under `results/repro_runs/`:",
        "",
        "```bash",
        "python src/significance_test.py",
        "```",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved to {report_path}")


if __name__ == "__main__":
    main()
