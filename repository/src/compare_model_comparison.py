from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two model-comparison aggregate CSV files.")
    parser.add_argument("--reference", required=True, help="Reference aggregate CSV path.")
    parser.add_argument("--candidate", required=True, help="Candidate aggregate CSV path.")
    parser.add_argument("--output", default=None, help="Optional output CSV path for the merged comparison.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ref = pd.read_csv(args.reference)
    cand = pd.read_csv(args.candidate)

    merged = ref.merge(cand, on="model", suffixes=("_ref", "_cand"))
    metric_roots = [
        "mean_hit",
        "extreme_hit",
        "stress_mean_hit_avg",
        "stress_extreme_hit_avg_q80",
        "rmse_x1e3",
    ]
    for root in metric_roots:
        ref_col = f"{root}_mean_ref"
        cand_col = f"{root}_mean_cand"
        if ref_col in merged.columns and cand_col in merged.columns:
            merged[f"{root}_delta"] = merged[cand_col] - merged[ref_col]

    cols = ["model"]
    for root in metric_roots:
        for suffix in ("ref", "cand"):
            for tail in ("mean", "std"):
                col = f"{root}_{tail}_{suffix}"
                if col in merged.columns:
                    cols.append(col)
        delta_col = f"{root}_delta"
        if delta_col in merged.columns:
            cols.append(delta_col)
    merged = merged[cols]

    print(merged.to_string(index=False))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)
        print(f"\nsaved comparison to {out_path}")


if __name__ == "__main__":
    main()
