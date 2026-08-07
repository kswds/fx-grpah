from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data_pipeline import prepare_data
from metrics import compute_metrics, save_dataframe


DISPLAY = {
    "oursmain": "ARC_FX",
    "mlp": "MLP",
    "transformer": "Transformer",
    "gnn": "GNN",
    "corrlstmgat": "Corr-LSTM-GAT",
    "fxrp": "FXRP",
}
THRESHOLDS_BP = [1, 3, 5, 10]


def public_model_name(model: str) -> str:
    return DISPLAY.get(model, model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the main predictive-results markdown table from saved prediction files.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "main_experiment.yaml"))
    parser.add_argument("--fx-data-path", default=None, help="Optional override for the processed FX CSV path.")
    parser.add_argument("--nonfx-data-path", default=None, help="Optional override for the processed non-FX CSV path.")
    parser.add_argument("--arcfx-root", default=None, help="Optional override for ARC_FX prediction root.")
    parser.add_argument("--baseline-root", default=None, help="Optional override for baseline prediction root.")
    parser.add_argument("--report-path", default=None, help="Optional override for markdown output path.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_data_path(root: Path, configured: str, override: str | None) -> str:
    if override:
        return str(Path(override).resolve())
    return str((root / configured).resolve())


def fmt(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {0.0 if pd.isna(std) else std:.{digits}f}"


def tail_score(train_values: pd.Series, full_values: pd.Series) -> pd.Series:
    train_arr = pd.to_numeric(train_values, errors="coerce").to_numpy(dtype=float)
    full_arr = pd.to_numeric(full_values, errors="coerce").to_numpy(dtype=float)
    train_arr = train_arr[np.isfinite(train_arr)]
    sorted_train = np.sort(train_arr)
    out = np.full(len(full_arr), np.nan)
    valid = np.isfinite(full_arr)
    out[valid] = np.searchsorted(sorted_train, full_arr[valid], side="right") / float(len(sorted_train))
    return pd.Series(out, index=full_values.index)


def build_simple_q85_flags(cfg: Dict[str, Any], fx_data_path: str | None, nonfx_data_path: str | None) -> pd.DataFrame:
    currencies = list(cfg["experiment"]["currencies"])
    prepared = prepare_data(
        resolve_data_path(ROOT, cfg["data"]["fx_data_path"], fx_data_path),
        resolve_data_path(ROOT, cfg["data"]["nonfx_data_path"], nonfx_data_path),
        currencies,
        int(cfg["experiment"]["lookback"]),
        list(cfg["experiment"]["split"]),
        include_regime_onehot=False,
    )
    merged = prepared.merged.copy()
    merged["target_date"] = pd.to_datetime(merged["Date"]).dt.normalize()
    train_end = int(len(merged) * cfg["experiment"]["split"][0])
    train = merged.iloc[:train_end].copy()
    non_usd = [c for c in currencies if c != "USD"]

    target_cols = [f"TargetRet_{c}" for c in non_usd]
    fx_vol = merged[target_cols].apply(pd.to_numeric, errors="coerce").abs().mean(axis=1)
    train_fx_vol = train[target_cols].apply(pd.to_numeric, errors="coerce").abs().mean(axis=1)

    vix_abs = pd.to_numeric(merged.get("Global_VIX_change", 0.0), errors="coerce").abs()
    train_vix_abs = pd.to_numeric(train.get("Global_VIX_change", 0.0), errors="coerce").abs()

    us10y_up = pd.to_numeric(merged.get("Global_US10Y_change", 0.0), errors="coerce")
    train_us10y_up = pd.to_numeric(train.get("Global_US10Y_change", 0.0), errors="coerce")

    flags = pd.DataFrame({"target_date": merged["target_date"]})
    flags["vix_stress"] = tail_score(train_vix_abs, vix_abs) >= 0.85
    flags["fx_stress"] = tail_score(train_fx_vol, fx_vol) >= 0.85
    flags["yield_stress"] = tail_score(train_us10y_up, us10y_up) >= 0.85
    return flags.drop_duplicates("target_date")


def prediction_path(root_dir: Path, model: str, seed: int) -> Path:
    return root_dir / f"{model}_seed{seed}" / "predictions" / f"{model}_predictions.parquet"


def load_prediction_df(path: Path, model: str, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.normalize()
    df["model"] = model
    df["seed"] = int(seed)
    return df


def prediction_panel(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[str]]:
    currencies = sorted(df["currency"].unique().tolist(), key=lambda x: (x != "USD", x))
    dates = sorted(df["target_date"].unique().tolist())
    pred_panel = df.pivot(index="target_date", columns="currency", values="pred").reindex(index=dates, columns=currencies)
    tgt_panel = df.pivot(index="target_date", columns="currency", values="target").reindex(index=dates, columns=currencies)
    valid = pred_panel.notna().all(axis=1) & tgt_panel.notna().all(axis=1)
    pred_panel = pred_panel.loc[valid]
    tgt_panel = tgt_panel.loc[valid]
    return pred_panel.to_numpy(dtype=float), tgt_panel.to_numpy(dtype=float), list(pred_panel.index), currencies


def class_from_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.int8)
    out[x > threshold] = 1
    out[x < -threshold] = -1
    return out


def macro_f1(labels: List[int], pred_cls: np.ndarray, tgt_cls: np.ndarray) -> float:
    vals: List[float] = []
    for label in labels:
        tp = float(np.sum((pred_cls == label) & (tgt_cls == label)))
        fp = float(np.sum((pred_cls == label) & (tgt_cls != label)))
        fn = float(np.sum((pred_cls != label) & (tgt_cls == label)))
        denom = 2.0 * tp + fp + fn
        if denom > 0:
            vals.append((2.0 * tp) / denom)
    return float(np.mean(vals)) if vals else np.nan


def evaluate_predictions(cfg: Dict[str, Any], arcfx_root: Path, baseline_root: Path, fx_data_path: str | None, nonfx_data_path: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    flags = build_simple_q85_flags(cfg, fx_data_path, nonfx_data_path).set_index("target_date")
    currencies = list(cfg["experiment"]["currencies"])
    non_usd = [c for c in currencies if c != "USD"]
    scenario_cols = ["vix_stress", "fx_stress", "yield_stress"]

    prepared = prepare_data(
        resolve_data_path(ROOT, cfg["data"]["fx_data_path"], fx_data_path),
        resolve_data_path(ROOT, cfg["data"]["nonfx_data_path"], nonfx_data_path),
        currencies,
        int(cfg["experiment"]["lookback"]),
        list(cfg["experiment"]["split"]),
        include_regime_onehot=False,
    )
    train_end = int(len(prepared.merged) * cfg["experiment"]["split"][0])
    abs_train = np.abs(prepared.y_raw[:train_end, 1:].reshape(-1))
    abs_train = abs_train[np.isfinite(abs_train)]
    train_q80 = float(np.quantile(abs_train, 0.80))

    model_order = ["oursmain"] + list(cfg["experiment"]["baseline_models"])
    rows: List[Dict[str, float | int | str]] = []
    nontrivial_rows: List[Dict[str, float | int | str]] = []

    for seed in cfg["experiment"]["seeds"]:
        for model in model_order:
            root_dir = arcfx_root if model == "oursmain" else baseline_root
            df = load_prediction_df(prediction_path(root_dir, model, int(seed)), model, int(seed))
            pred_np, tgt_np, dates, currencies_order = prediction_panel(df)
            non_usd_mask = np.array([c != "USD" for c in currencies_order], dtype=bool)
            metrics = compute_metrics(pred_np, tgt_np, non_usd_mask)

            pred_nu = pred_np[:, non_usd_mask]
            tgt_nu = tgt_np[:, non_usd_mask]
            extreme_mask = np.abs(tgt_nu) >= train_q80
            extreme_hit = float(np.mean((np.sign(pred_nu) == np.sign(tgt_nu))[extreme_mask]))

            sub_all = df[df["currency"].isin(non_usd)].copy()
            aligned_flags = flags.reindex(pd.Index(pd.to_datetime(dates).normalize())).fillna(False)
            stress_mean_hits = []
            stress_extreme_q80_hits = []
            for scenario in scenario_cols:
                sdates = aligned_flags.index[aligned_flags[scenario]]
                sub = sub_all[sub_all["target_date"].isin(sdates)].copy()
                pred_vals = sub["pred"].to_numpy(dtype=float)
                tgt_vals = sub["target"].to_numpy(dtype=float)
                stress_mean_hits.append(float(np.mean(np.sign(pred_vals) == np.sign(tgt_vals))))
                thr = float(np.nanquantile(np.abs(tgt_vals), 0.80))
                ex_mask = np.abs(tgt_vals) >= thr
                stress_extreme_q80_hits.append(float(np.mean((np.sign(pred_vals) == np.sign(tgt_vals))[ex_mask])))

            rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "mean_hit": float(metrics["hit_ratio"]),
                    "extreme_hit": extreme_hit,
                    "stress_mean_hit_avg": float(np.mean(stress_mean_hits)),
                    "stress_extreme_hit_avg_q80": float(np.mean(stress_extreme_q80_hits)),
                    "rmse_x1e3": float(metrics["rmse"] * 1000.0),
                }
            )

            pred_flat = sub_all["pred"].to_numpy(dtype=float)
            tgt_flat = sub_all["target"].to_numpy(dtype=float)
            sign_equal = np.sign(pred_flat) == np.sign(tgt_flat)
            for bp in THRESHOLDS_BP:
                thr = bp / 10000.0
                both_flat = (np.abs(pred_flat) <= thr) & (np.abs(tgt_flat) <= thr)
                keep = ~both_flat
                pred_cls = class_from_threshold(pred_flat, thr)
                tgt_cls = class_from_threshold(tgt_flat, thr)
                nontrivial_rows.append(
                    {
                        "model": model,
                        "seed": int(seed),
                        "threshold_bp": int(bp),
                        "nontrivial_hit": float(np.mean(sign_equal[keep])) if keep.any() else np.nan,
                        "macro_f1": macro_f1([-1, 0, 1], pred_cls[keep], tgt_cls[keep]) if keep.any() else np.nan,
                    }
                )

    detail = pd.DataFrame(rows)
    agg = detail.groupby("model", as_index=False)[
        ["mean_hit", "extreme_hit", "stress_mean_hit_avg", "stress_extreme_hit_avg_q80", "rmse_x1e3"]
    ].agg(["mean", "std"])
    agg.columns = ["_".join([c for c in col if c]).strip("_") for col in agg.columns.to_flat_index()]
    agg = agg.rename(columns={"model_": "model"})

    nontrivial_detail = pd.DataFrame(nontrivial_rows)
    nontrivial_agg = (
        nontrivial_detail.groupby(["model", "threshold_bp"], as_index=False)[["nontrivial_hit", "macro_f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    nontrivial_agg.columns = ["_".join([c for c in col if c]).strip("_") for col in nontrivial_agg.columns.to_flat_index()]
    nontrivial_agg = nontrivial_agg.rename(columns={"model_": "model", "threshold_bp_": "threshold_bp"})
    return detail, agg, nontrivial_detail, nontrivial_agg


def build_report(cfg: Dict[str, Any], agg: pd.DataFrame, nontrivial_agg: pd.DataFrame, arcfx_root: Path, baseline_root: Path) -> str:
    lines = [
        "# Main Predictive Results",
        "",
        "This report reproduces the headline predictive model comparison used in the anonymous GitHub release.",
        "",
        "Evaluation setup:",
        "- All models use the same shared training objective for this comparison: the active directional core on non-USD targets only.",
        "- Active set definition: `A = {(b, i) : |y_{b,i}| >= tau}`.",
        "- Threshold definition: `tau = Q0.40(|y_train_norm|)`.",
        "- Core loss on the active set: `mean softplus(- rhat_{b,i} * sign(y_{b,i}))`.",
        f"- Seed set: `{', '.join(str(int(seed)) for seed in cfg['experiment']['seeds'])}`. Reported numbers are mean ± standard deviation across these seeds.",
        "- `Extreme Hit` is the sign hit ratio computed on the pooled train-based extreme subset, where `|y|` exceeds the training `Q0.80` threshold over non-USD targets.",
        "- `Stress Mean Hit Avg.` is the simple average of sign hit ratios across the three stress scenarios `{VIX stress, FX stress, Yield stress}`.",
        "- `Stress Extreme Hit Avg. @ Q80` is computed inside each stress scenario using that scenario's realized `Q0.80` absolute-return subset, and then averaged across the three scenarios.",
        "- Stress scenarios are defined from training-sample percentile rules: `VIX stress = train-rank(|VIX change|) >= 0.85`, `FX stress = train-rank(cross-currency mean absolute FX return) >= 0.85`, and `Yield stress = train-rank(US10Y change) >= 0.85`.",
        "- `Corr-LSTM-GAT` and `FXRP` are included as reference baselines.",
        "",
        "## Headline Predictive Metrics",
        "",
        "| Model | Mean Hit ↑ | Extreme Hit ↑ | Stress Mean Hit Avg. ↑ | Stress Extreme Hit Avg. @ Q80 ↑ | RMSE ×10³ ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for model in ["oursmain"] + list(cfg["experiment"]["baseline_models"]):
        row = agg[agg["model"] == model].iloc[0]
        vals = [
            DISPLAY[model],
            fmt(float(row["mean_hit_mean"]), float(row["mean_hit_std"]), 4),
            fmt(float(row["extreme_hit_mean"]), float(row["extreme_hit_std"]), 4),
            fmt(float(row["stress_mean_hit_avg_mean"]), float(row["stress_mean_hit_avg_std"]), 4),
            fmt(float(row["stress_extreme_hit_avg_q80_mean"]), float(row["stress_extreme_hit_avg_q80_std"]), 4),
            fmt(float(row["rmse_x1e3_mean"]), float(row["rmse_x1e3_std"]), 3),
        ]
        if model == "oursmain":
            vals = [f"**{v}**" for v in vals]
        lines.append("| " + " | ".join(vals) + " |")

    lines.extend(
        [
            "",
            "## Non-Trivial Directional Hit And F1",
            "",
            "The neutral-band thresholds below are basis-point bands (`±1bp`, `±3bp`, `±5bp`, `±10bp`).",
            "",
            "- `Non-trivial hit`: excludes only the observations where both the prediction and the realized return fall inside the same neutral band, then measures sign hit on the remaining subset.",
            "- `Macro-F1`: computed on the same subset using ternary labels `{down, flat, up}` induced by the threshold.",
            "",
            "### Non-Trivial Directional Hit",
            "",
            "| Model | ±1bp | ±3bp | ±5bp | ±10bp |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in ["oursmain"] + list(cfg["experiment"]["baseline_models"]):
        row = [DISPLAY[model]]
        for bp in THRESHOLDS_BP:
            cur = nontrivial_agg[(nontrivial_agg["model"] == model) & (nontrivial_agg["threshold_bp"] == bp)].iloc[0]
            row.append(fmt(float(cur["nontrivial_hit_mean"]), float(cur["nontrivial_hit_std"]), 4))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "### Macro-F1",
            "",
            "| Model | ±1bp | ±3bp | ±5bp | ±10bp |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in ["oursmain"] + list(cfg["experiment"]["baseline_models"]):
        row = [DISPLAY[model]]
        for bp in THRESHOLDS_BP:
            cur = nontrivial_agg[(nontrivial_agg["model"] == model) & (nontrivial_agg["threshold_bp"] == bp)].iloc[0]
            row.append(fmt(float(cur["macro_f1_mean"]), float(cur["macro_f1_std"]), 4))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    outputs = cfg["outputs"]
    arcfx_root = Path(args.arcfx_root) if args.arcfx_root else (ROOT / outputs["prediction_root"] / "arc_fx")
    baseline_root = Path(args.baseline_root) if args.baseline_root else (ROOT / outputs["prediction_root"] / "baselines")
    report_path = Path(args.report_path) if args.report_path else (ROOT / outputs["report_dir"] / "final_overall_prediction.md")

    detail, agg, nontrivial_detail, nontrivial_agg = evaluate_predictions(
        cfg,
        arcfx_root,
        baseline_root,
        args.fx_data_path,
        args.nonfx_data_path,
    )
    detail["model"] = detail["model"].map(public_model_name)
    agg["model"] = agg["model"].map(public_model_name)
    nontrivial_detail["model"] = nontrivial_detail["model"].map(public_model_name)
    nontrivial_agg["model"] = nontrivial_agg["model"].map(public_model_name)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataframe(detail, report_path.parent / "model_comparison_detail.csv")
    save_dataframe(agg, report_path.parent / "model_comparison_aggregate.csv")
    save_dataframe(nontrivial_detail, report_path.parent / "nontrivial_directional_detail.csv")
    save_dataframe(nontrivial_agg, report_path.parent / "nontrivial_directional_aggregate.csv")
    report_path.write_text(build_report(cfg, agg, nontrivial_agg, arcfx_root, baseline_root), encoding="utf-8")
    print(f"saved report to {report_path}")


if __name__ == "__main__":
    main()
