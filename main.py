"""
Main entry point for FX Strength GNN experiments

Usage:
    # Baseline mode (reproduces original kswds results)
    python main.py --mode baseline

    # Enhanced mode (with our improvements)
    python main.py --mode enhanced --skip_connection --layer_norm

    # Select specific currencies/macros for ablation
    python main.py --currencies USD,EUR,JPY,GBP --macros VIX,Oil,SP500
"""
import argparse

from config import Config
from dataset import create_dataloaders, fully_connected_edge_index
from models import create_model
from train import Trainer
from utils import set_seed, get_device


# Available currencies and macros (full set)
ALL_CURRENCIES = ["USD", "EUR", "JPY", "GBP", "CAD", "AUD", "CHF", "NZD", "SEK", "NOK"]
ALL_MACROS = ["Gold", "VIX", "Oil", "US10Y", "Copper", "SP500", "US2Y"]


def parse_args():
    parser = argparse.ArgumentParser(description="FX Strength GNN")

    # Mode selection
    parser.add_argument("--mode", type=str, default="baseline",
                        choices=["baseline", "enhanced"],
                        help="baseline: original kswds | enhanced: with our improvements")

    # Data
    parser.add_argument("--data_path", type=str, default="factor_final_daily.csv")

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lookback", type=int, default=20)

    # Model
    parser.add_argument("--gnn_type", type=str, default="gat",
                        choices=["gcn", "sage", "gat"])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)

    # Enhanced mode options
    parser.add_argument("--skip_connection", action="store_true",
                        help="Use skip connection in GNN")
    parser.add_argument("--layer_norm", action="store_true",
                        help="Use layer normalization")
    parser.add_argument("--magnitude_head", action="store_true",
                        help="Use magnitude prediction head")

    # Loss weights
    parser.add_argument("--lambda_var", type=float, default=0.005)
    parser.add_argument("--lambda_a_l1", type=float, default=1e-4)

    # Currency and macro selection (for ablation studies)
    parser.add_argument("--currencies", type=str, default=None,
                        help="Comma-separated currencies to use (e.g., USD,EUR,JPY). Default: all 10")
    parser.add_argument("--macros", type=str, default=None,
                        help="Comma-separated macros to use (e.g., VIX,Oil,SP500). Default: all 7")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Device
    device = get_device()
    print(f"Device: {device}")

    # Parse currencies and macros
    if args.currencies:
        currencies = [c.strip() for c in args.currencies.split(",")]
        # Validate currencies
        for c in currencies:
            if c not in ALL_CURRENCIES:
                raise ValueError(f"Unknown currency: {c}. Available: {ALL_CURRENCIES}")
        # USD must be included for relative strength calculation
        if "USD" not in currencies:
            print("Warning: USD not in currencies, adding automatically (required for relative strength)")
            currencies = ["USD"] + currencies
    else:
        currencies = ALL_CURRENCIES

    if args.macros:
        macros = [m.strip() for m in args.macros.split(",")]
        # Validate macros
        for m in macros:
            if m not in ALL_MACROS:
                raise ValueError(f"Unknown macro: {m}. Available: {ALL_MACROS}")
        global_features = [f"Global_{m}" for m in macros]
    else:
        global_features = [f"Global_{m}" for m in ALL_MACROS]

    # Create config
    config = Config(
        file_path=args.data_path,
        seed=args.seed,
        lookback=args.lookback,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        gnn_type=args.gnn_type,
        hidden=args.hidden,
        heads=args.heads,
        lambda_var=args.lambda_var,
        lambda_a_l1=args.lambda_a_l1,
        ccys=currencies,
        global_features=global_features,
    )

    # Enhanced mode options
    if args.mode == "enhanced":
        config.use_skip_connection = args.skip_connection
        config.use_layer_norm = args.layer_norm
        config.use_magnitude_head = args.magnitude_head
        print(f"\n=== Enhanced Mode ===")
        print(f"Skip Connection: {config.use_skip_connection}")
        print(f"Layer Norm: {config.use_layer_norm}")
        print(f"Magnitude Head: {config.use_magnitude_head}")
    else:
        print(f"\n=== Baseline Mode (Original kswds) ===")

    # Show selected currencies and macros
    print(f"Currencies ({config.n_ccy}): {config.ccys}")
    print(f"Macros ({config.macro_dim}): {[f.replace('Global_', '') for f in config.global_features]}")

    # Edge index
    edge_index = fully_connected_edge_index(config.n_ccy).to(device)

    print("\n>>> Start Training...")

    # WITH MACRO
    set_seed(args.seed)
    train_loader, test_loader = create_dataloaders(config, macro_mode="real")
    model_macro = create_model(config)
    trainer_macro = Trainer(model_macro, config, device)
    res_macro = trainer_macro.train(train_loader, test_loader, edge_index, label="WITH MACRO")

    # WITHOUT MACRO
    set_seed(args.seed)
    train_loader_zero, test_loader_zero = create_dataloaders(config, macro_mode="zero")
    model_zero = create_model(config)
    trainer_zero = Trainer(model_zero, config, device)
    res_zero = trainer_zero.train(train_loader_zero, test_loader_zero, edge_index, label="WITHOUT MACRO")

    # Summary
    print("\n===== Macro Contribution (Hetero Edges) =====")
    print(f"Δ RMSE : {res_zero['rmse'] - res_macro['rmse']:.4f}")
    print(f"Δ MAE  : {res_zero['mae'] - res_macro['mae']:.4f}")
    print(f"Δ Hit  : {res_macro['hit'] - res_zero['hit']:.4f}")
    print(f"Δ MUR  : {res_macro['mur'] - res_zero['mur']:.4f}")
    print(f"HS(mean Var across factors): {res_macro['hs_mean']:.6f}")


if __name__ == "__main__":
    main()
