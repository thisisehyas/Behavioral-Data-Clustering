from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_kl_curves(
    runs_root: Path,
    figures_dir: Path,
    parameter_prefix: str,
    parameter_name: str,
    output_filename: str,
    show_train: bool = True,
):
    """
    Plot mean KL curves across seeds for each parameter value.
    """

    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))

    parameter_dirs = sorted(
        [
            d
            for d in runs_root.iterdir()
            if d.is_dir()
        ]
    )

    for parameter_dir in parameter_dirs:

        parameter_value = parameter_dir.name.replace(
            parameter_prefix,
            "",
        )

        train_curves = []
        val_curves = []

        seed_dirs = sorted(
            [
                d
                for d in parameter_dir.iterdir()
                if d.is_dir()
            ]
        )

        for seed_dir in seed_dirs:

            train_path = seed_dir / "train_kl.npy"
            val_path = seed_dir / "val_kl.npy"

            if train_path.exists():
                train_curves.append(np.load(train_path))

            if val_path.exists():
                val_curves.append(np.load(val_path))

        if len(val_curves) == 0:
            print(f"Skipping {parameter_dir.name}: no KL curves")
            continue

        mean_val = np.mean(val_curves, axis=0)
        epochs = np.arange(len(mean_val))

        if show_train and len(train_curves) > 0:

            mean_train = np.mean(train_curves, axis=0)

            plt.plot(
                epochs,
                mean_train,
                linestyle="--",
                linewidth=2,
                label=f"{parameter_name}={parameter_value} (train)",
            )

        plt.plot(
            epochs,
            mean_val,
            linewidth=2,
            label=f"{parameter_name}={parameter_value} (val)",
        )

    plt.xlabel("Epoch")
    plt.ylabel("KL divergence")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.title(
        f"DEC training dynamics across {parameter_name} values"
    )

    plt.legend()

    output_path = figures_dir / output_filename

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure to: {output_path}")