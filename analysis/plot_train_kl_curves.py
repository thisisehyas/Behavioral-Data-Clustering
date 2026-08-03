from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_train_kl_curves(
    runs_root: Path,
    figures_dir: Path,
    prefix: str,
    parameter_name: str,
):
    """
    Plot the mean training KL curve across seeds for each
    hyperparameter value.
    """

    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))

    parameter_dirs = sorted(
        [
            d for d in runs_root.iterdir()
            if d.is_dir()
        ]
    )

    for parameter_dir in parameter_dirs:

        parameter_value = parameter_dir.name.replace(prefix, "")

        seed_curves = []

        seed_dirs = sorted(
            [
                d for d in parameter_dir.iterdir()
                if d.is_dir()
            ]
        )

        for seed_dir in seed_dirs:

            train_kl_path = seed_dir / "train_kl.npy"

            if train_kl_path.exists():
                seed_curves.append(np.load(train_kl_path))

        if len(seed_curves) == 0:
            print(f"Skipping {parameter_dir.name}: no data")
            continue

        mean_curve = np.mean(seed_curves, axis=0)
        std_curve = np.std(seed_curves, axis=0)

        epochs = np.arange(len(mean_curve))

        plt.plot(
            epochs,
            mean_curve,
            linewidth=2,
            label=f"{parameter_name}={parameter_value}",
        )

        plt.fill_between(
            epochs,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training KL")
    plt.title(f"Training KL vs Epoch Across {parameter_name} Values")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    output_path = figures_dir / f"train_kl_curves_by_{parameter_name}.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure to: {output_path}")