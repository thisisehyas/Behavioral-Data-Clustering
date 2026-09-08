import argparse
import os
import yaml
from pathlib import Path

from src.dec.assets import load_features
from src.dec.training import run_dec_final_single_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run final DEC k-sensitivity experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("Loaded config successfully.")
    print(f"Experiment name: {config['experiment']['name']}")

    # Load full dataset
    X = load_features()

    k_values = config["sweep"]["k_values"]
    seed = config["sweep"]["seed"]

    experiment_root = Path(
        config["experiment"]["output_dir"]
    )
    ae_checkpoint_path = config["experiment"]["ae_checkpoint_path"]

    experiment_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Run final DEC configuration across different values of k
    for k in k_values:

        run_dir = experiment_root / f"k_{k}"

        if run_dir.exists() and any(run_dir.iterdir()):
            print(f"[SKIP] k={k}")
            continue

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(f"\n[RUN] k={k}, seed={seed}")
        print(f"Saving to: {run_dir}")

        # Use a copy so that changing k does not modify
        # the original configuration for subsequent runs.
        run_config = config.copy()
        run_config["dec"] = config["dec"].copy()
        run_config["dec"]["num_clusters"] = k

        run_dec_final_single_seed(
            X=X,
            ae_checkpoint_path=ae_checkpoint_path,
            seed=seed,
            config=run_config,
            output_dir=str(run_dir),
        )


if __name__ == "__main__":
    main()