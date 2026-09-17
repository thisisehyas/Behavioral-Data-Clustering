import argparse
import os
import yaml
from pathlib import Path

from src.dec.assets import load_features
from src.dec.training import run_idec_single_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run IDEC diagnostic experiment"
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
    print(
        f"Experiment name: {config['experiment']['name']}"
    )

    # Load full dataset
    X = load_features()

    seed = config["sweep"]["seed"]

    experiment_root = Path(
        config["experiment"]["output_dir"]
    )

    ae_checkpoint_path = config["experiment"][
        "ae_checkpoint_path"
    ]

    experiment_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_dir = experiment_root / f"seed_{seed}"

    if run_dir.exists() and any(run_dir.iterdir()):
        print(f"[SKIP] seed={seed}")
        return

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"\n[RUN] seed={seed}")
    print(f"Saving to: {run_dir}")

    run_idec_single_seed(
        X=X,
        ae_checkpoint_path=ae_checkpoint_path,
        seed=seed,
        config=config,
        output_dir=str(run_dir),
    )


if __name__ == "__main__":
    main()