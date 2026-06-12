import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from exp_utils import save_csv, save_json, save_predictions
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(__file__))
    from exp_utils import save_csv, save_json, save_predictions


COMPONENT_MODELS = ["NoMacro", "NoGraph", "StaticGraph", "NoDirect", "Ours"]

COMPONENT_TO_INTERNAL = {
    "NoMacro": "NoMacro",
    "NoGraph": "NoGraph",
    "StaticGraph": "StaticGraph",
    "NoDirect": "PureGraphFX",  # graph-only, no direct branch
    "Ours": "Ours",
}

DISPLAY_NAMES = {
    "NoMacro": "NoMacroFX",
    "NoGraph": "NoGraphFX",
    "StaticGraph": "StaticGraphFX",
    "NoDirect": "GraphOnly(PureGraphFX)",
    "Ours": "MACRO-HyGraph",
}

METRIC_COLS = ["rmse", "mae", "hit_ccy", "hit_pair", "extreme_rmse", "extreme_hit", "ic", "sharpe"]


def ensure_component_dirs(base_dir: str | Path) -> dict[str, Path]:
    base = Path(base_dir)
    out = {
        "base": base,
        "metrics": base / "metrics",
        "predictions": base / "predictions",
        "configs": base / "configs",
        "tables": base / "tables",
        "checkpoints": base / "checkpoints",
    }
    for p in out.values():
        if isinstance(p, Path):
            p.mkdir(parents=True, exist_ok=True)
    return out


def long_prediction_df(pred, target, dates_df: pd.DataFrame, model: str, lookback: str, seed: int, ccys: list[str]) -> pd.DataFrame:
    rows = []
    for t in range(pred.shape[0]):
        for i, c in enumerate(ccys):
            rows.append(
                {
                    "date": dates_df.iloc[t]["target_date"],
                    "input_end_date": dates_df.iloc[t]["input_end_date"],
                    "target_date": dates_df.iloc[t]["target_date"],
                    "model": model,
                    "lookback": str(lookback),
                    "seed": int(seed),
                    "currency": c,
                    "pred": float(pred[t, i]),
                    "target": float(target[t, i]),
                }
            )
    return pd.DataFrame(rows)


def summarize_component_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, d, lb), g in raw_df.groupby(["model", "display_name", "lookback"]):
        row = {"model": m, "display_name": d, "lookback": lb}
        for c in METRIC_COLS:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_std"] = float(g[c].std(ddof=0))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["lookback", "model"]).reset_index(drop=True)
    return out


def component_improvement(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lb, g in summary_df.groupby("lookback"):
        d = {r["model"]: r for _, r in g.iterrows()}
        if "Ours" not in d:
            continue
        ours = d["Ours"]
        for m in ["NoMacro", "NoGraph", "StaticGraph", "NoDirect"]:
            if m not in d:
                continue
            b = d[m]
            rmse_imp = (float(b["rmse_mean"]) - float(ours["rmse_mean"])) / (float(b["rmse_mean"]) + 1e-12) * 100.0
            mae_imp = (float(b["mae_mean"]) - float(ours["mae_mean"])) / (float(b["mae_mean"]) + 1e-12) * 100.0
            rows.append(
                {
                    "lookback": lb,
                    "baseline_model": m,
                    "baseline_display_name": b["display_name"],
                    "ours_rmse_mean": float(ours["rmse_mean"]),
                    "baseline_rmse_mean": float(b["rmse_mean"]),
                    "improvement_rmse_pct": float(rmse_imp),
                    "improvement_mae_pct": float(mae_imp),
                    "improvement_hit_ccy": float(ours["hit_ccy_mean"] - b["hit_ccy_mean"]),
                    "improvement_hit_pair": float(ours["hit_pair_mean"] - b["hit_pair_mean"]),
                    "improvement_extreme_hit": float(ours["extreme_hit_mean"] - b["extreme_hit_mean"]),
                }
            )
    return pd.DataFrame(rows)


def save_component_outputs(
    dirs: dict[str, Path],
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
) -> None:
    save_csv(dirs["metrics"] / "component_ablation_raw.csv", raw_df)
    save_csv(dirs["tables"] / "component_ablation_summary.csv", summary_df)
    save_csv(dirs["tables"] / "ours_component_improvement.csv", improvement_df)
    md = summary_df.copy()
    for c in METRIC_COLS:
        md[c] = md.apply(lambda r: f"{r[f'{c}_mean']:.6f} ± {r[f'{c}_std']:.6f}", axis=1)
    md_cols = ["model", "display_name", "lookback"] + METRIC_COLS
    (dirs["tables"] / "component_ablation_summary.md").write_text(md[md_cols].to_markdown(index=False), encoding="utf-8")

