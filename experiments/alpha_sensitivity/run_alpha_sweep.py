import argparse
import os
import yaml
from pathlib import Path

from src.dec.assets import load_fixed_assets
from src.dec.training import run_dec_single_alpha_single_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DEC alpha sensitivity experiments"
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

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("Loaded config successfully.")
    print(f"Experiment: {config['experiment']['name']}")

    # fixed assets
    assets = load_fixed_assets()

    X = assets["X"]
    train_idx = assets["train_idx"]
    val_idx = assets["val_idx"]

    alphas = config["sweep"]["alphas"]
    seeds = config["sweep"]["seeds"]

    experiment_root = Path(config["experiment"]["output_dir"])
    ae_checkpoint_path = config["experiment"]["ae_checkpoint_path"]

    experiment_root.mkdir(parents=True, exist_ok=True)

    for alpha in alphas:
        alpha_dir = experiment_root / f"alpha_{alpha}"
        alpha_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            run_dir = alpha_dir / f"seed_{seed}"

            if run_dir.exists() and any(run_dir.iterdir()):
                print(f"[SKIP] alpha={alpha}, seed={seed}")
                continue

            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[RUN] alpha={alpha}, seed={seed}")

            run_dec_single_alpha_single_seed(
                X=X,
                train_idx=train_idx,
                val_idx=val_idx,
                ae_checkpoint_path=ae_checkpoint_path,
                alpha=alpha,
                seed=seed,
                config=config,
                output_dir=str(run_dir),
            )


if __name__ == "__main__":
    main()
