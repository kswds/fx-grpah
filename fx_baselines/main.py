import math
import numpy as np
import torch

from config import Config
from dataset import process_data, build_tensors_raw
from models import fully_connected_row_stochastic, build_granger_proxy_graph
from train import train_and_evaluate
from utils import set_seed, mean_ci, save_results

def run_one_seed(seed: int, base_config: Config | None = None):
    config = base_config if base_config is not None else Config()
    config.seed = seed
    set_seed(seed)

    print("\n" + "=" * 70)
    print(f"Seed = {seed} | Device = {config.device}")
    print("=" * 70)

    raw_data = process_data(config)
    X_local_raw, X_macro_raw, Y_raw = build_tensors_raw(raw_data, config)

    # split index for train-only graph building
    total_len = len(X_local_raw)
    valid_len = total_len - config.lookback
    train_split_idx = int(valid_len * 0.8) + config.lookback

    device = torch.device(config.device)
    W_fc = fully_connected_row_stochastic(config.n_ccy, device=device)

    if config.use_granger_graph:
        Wg_np = build_granger_proxy_graph(Y_raw, train_split_idx=train_split_idx, config=config)
        W_g = torch.tensor(Wg_np, device=device)
    else:
        W_g = W_fc

    # (1) Baseline MLP (strong GRU + macro injection)
    res_mlp = train_and_evaluate(
        config, X_local_raw, X_macro_raw, Y_raw,
        model_name="mlp",
        macro_mode="real",
        label="BASELINE: MLP (GRU + Hetero-Macro)"
    )

    # (2) Static FC Graph + Macro
    res_fc = train_and_evaluate(
        config, X_local_raw, X_macro_raw, Y_raw,
        model_name="static_fc",
        macro_mode="real",
        label="GNN: STATIC-FC graph (WITH MACRO)",
        W_graph=W_fc
    )

    # (3) Granger + ShockProp + Macro
    res_gsp = train_and_evaluate(
        config, X_local_raw, X_macro_raw, Y_raw,
        model_name="granger_shockprop",
        macro_mode="real",
        label="GNN: GRANGER + SHOCK-PROP (WITH MACRO)",
        W_graph=W_g
    )

    # (4) Granger + ShockProp, macro zero (sanity/ablation)
    res_gsp0 = train_and_evaluate(
        config, X_local_raw, X_macro_raw, Y_raw,
        model_name="granger_shockprop",
        macro_mode="zero",
        label="GNN: GRANGER + SHOCK-PROP (MACRO=ZERO)",
        W_graph=W_g
    )

    summary = {
        "fc_minus_mlp_hit": res_fc["hit"] - res_mlp["hit"],
        "gsp_minus_fc_hit": res_gsp["hit"] - res_fc["hit"],
        "macro_contribution_hit": res_gsp["hit"] - res_gsp0["hit"],
        "fc_minus_mlp_rmse": res_fc["rmse"] - res_mlp["rmse"],
        "gsp_minus_fc_rmse": res_gsp["rmse"] - res_fc["rmse"],
        "macro_contribution_rmse": res_gsp["rmse"] - res_gsp0["rmse"],
    }

    return {
        "seed": seed,
        "metrics": {
            "mlp_with_macro": res_mlp,
            "static_fc_with_macro": res_fc,
            "granger_shockprop_with_macro": res_gsp,
            "granger_shockprop_macro_zero": res_gsp0,
        },
        "summary": summary,
        "graph_info": {
            "use_granger_graph": config.use_granger_graph,
            "granger_topk": config.granger_topk,
            "granger_min_weight": config.granger_min_weight,
            "shockprop_steps": config.shockprop_steps,
        }
    }

def main():
    seeds = [42]
    base_config = Config()
    runs = [run_one_seed(s, base_config=base_config) for s in seeds]

    def get_metric(model_key: str, metric_key: str):
        return [r["metrics"][model_key][metric_key] for r in runs]

    agg = {
        "BASELINE: MLP (GRU + Hetero-Macro)": {
            "hit_mean_ci": mean_ci(get_metric("mlp_with_macro", "hit")),
            "rmse_mean_ci": mean_ci(get_metric("mlp_with_macro", "rmse")),
            "hit_ext_mean_ci": mean_ci(get_metric("mlp_with_macro", "hit_ext")),
            "w_hit_mean_ci": mean_ci(get_metric("mlp_with_macro", "weighted_hit")),
        },
        "GNN: STATIC-FC graph (WITH MACRO)": {
            "hit_mean_ci": mean_ci(get_metric("static_fc_with_macro", "hit")),
            "rmse_mean_ci": mean_ci(get_metric("static_fc_with_macro", "rmse")),
            "hit_ext_mean_ci": mean_ci(get_metric("static_fc_with_macro", "hit_ext")),
            "w_hit_mean_ci": mean_ci(get_metric("static_fc_with_macro", "weighted_hit")),
        },
        "GNN: GRANGER + SHOCK-PROP (WITH MACRO)": {
            "hit_mean_ci": mean_ci(get_metric("granger_shockprop_with_macro", "hit")),
            "rmse_mean_ci": mean_ci(get_metric("granger_shockprop_with_macro", "rmse")),
            "hit_ext_mean_ci": mean_ci(get_metric("granger_shockprop_with_macro", "hit_ext")),
            "w_hit_mean_ci": mean_ci(get_metric("granger_shockprop_with_macro", "weighted_hit")),
        },
        "GNN: GRANGER + SHOCK-PROP (MACRO=ZERO)": {
            "hit_mean_ci": mean_ci(get_metric("granger_shockprop_macro_zero", "hit")),
            "rmse_mean_ci": mean_ci(get_metric("granger_shockprop_macro_zero", "rmse")),
            "hit_ext_mean_ci": mean_ci(get_metric("granger_shockprop_macro_zero", "hit_ext")),
            "w_hit_mean_ci": mean_ci(get_metric("granger_shockprop_macro_zero", "weighted_hit")),
        },
        "DIFFS": {
            "fc_minus_mlp_hit_mean_ci": mean_ci([r["summary"]["fc_minus_mlp_hit"] for r in runs]),
            "gsp_minus_fc_hit_mean_ci": mean_ci([r["summary"]["gsp_minus_fc_hit"] for r in runs]),
            "macro_contribution_hit_mean_ci": mean_ci([r["summary"]["macro_contribution_hit"] for r in runs]),
            "fc_minus_mlp_rmse_mean_ci": mean_ci([r["summary"]["fc_minus_mlp_rmse"] for r in runs]),
            "gsp_minus_fc_rmse_mean_ci": mean_ci([r["summary"]["gsp_minus_fc_rmse"] for r in runs]),
            "macro_contribution_rmse_mean_ci": mean_ci([r["summary"]["macro_contribution_rmse"] for r in runs]),
        }
    }

    experiment_results = {
        "experiment_info": {
            "note": "RAW-Y FX forecasting: targets are RAW USD-base log returns. NO Y normalization. Inputs scaled using TRAIN-only stats. Granger-proxy graph built using TRAIN lag-1 correlations. Shock propagation: propagate -> macro -> propagate.",
            "lookback": base_config.lookback,
            "rv_window": base_config.rv_window,
            "epochs": base_config.epochs,
            "lr": base_config.lr,
            "batch_size": base_config.batch_size,
            "hidden": base_config.hidden,
            "dropout": base_config.dropout,
            "use_granger_graph": base_config.use_granger_graph,
            "granger_topk": base_config.granger_topk,
            "granger_min_weight": base_config.granger_min_weight,
            "shockprop_steps": base_config.shockprop_steps,
            "extreme_percentile": base_config.extreme_percentile,
            "n_runs": len(seeds),
            "seeds": seeds,
        },
        "runs": runs,
        "aggregate": agg
    }

    save_results(experiment_results, output_dir="results", filename="results_all_compare_rawY.json")

if __name__ == "__main__":
    main()
