import argparse
import os
import yaml
from pathlib import Path

from src.dec.assets import load_fixed_assets
from src.dec.training import run_dec_single_lr_single_seed

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DEC learning rate sweep experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML file"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print("Loaded config successfully.")
    print(f"Experiment name: {config['experiment']['name']}")

    # fixed assets  
    assets = load_fixed_assets()

    X = assets["X"]
    train_idx = assets["train_idx"]
    val_idx = assets["val_idx"]

    # experiment parameters
    candidate_lrs = [float(lr) for lr in config["sweep"]["learning_rates"]]
    seeds = config["sweep"]["seeds"]

    experiment_root = Path(config['experiment']['output_dir'])
    ae_checkpoint_path = config['experiment']['ae_checkpoint_path']

    experiment_root.mkdir(parents=True, exist_ok=True)

    # controlled LR x seed loops
    for lr in candidate_lrs:
        lr_dir = experiment_root / f"lr_{lr:.0e}"
        lr_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            run_dir = lr_dir / f"seed_{seed}"
            
            # safe skip
            if run_dir.exists() and any(run_dir.iterdir()):
                print(f"[SKIP] LR={lr}, Seed={seed} already completed")
                continue

            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[RUN] LR={lr}, Seed={seed}")
            print(f"Saving to: {run_dir}")

            run_dec_single_lr_single_seed(
                X=X,
                train_idx=train_idx,
                val_idx=val_idx,
                ae_checkpoint_path=ae_checkpoint_path,
                lr=lr,
                seed=seed,
                config=config,
                output_dir=str(run_dir)
            )

if __name__ == "__main__":
    main()