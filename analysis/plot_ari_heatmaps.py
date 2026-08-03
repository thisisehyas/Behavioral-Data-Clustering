from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_ari_heatmaps(
    aggregated_dir: Path,
    figures_dir: Path,
    parameter_name: str,
):
    """
    Plot one ARI heatmap for every hyperparameter value.
    """

    figures_dir.mkdir(parents=True, exist_ok=True)

    ari_files = sorted(
        aggregated_dir.glob(
            f"ari_matrix_{parameter_name}_*.npy"
        )
    )

    if not ari_files:
        raise RuntimeError("No ARI matrix files found.")

    for ari_path in ari_files:

        parameter_value = ari_path.stem.replace(
            f"ari_matrix_{parameter_name}_",
            "",
        )

        ari_matrix = np.load(ari_path)

        plt.figure(figsize=(4, 4))

        im = plt.imshow(
            ari_matrix,
            vmin=0,
            vmax=1,
            cmap="coolwarm",
        )

        plt.colorbar(
            im,
            fraction=0.046,
            pad=0.04,
        )

        plt.title(
            f"Pairwise ARI between seeds ({parameter_name} = {parameter_value})"
        )

        n = ari_matrix.shape[0]

        plt.xticks(range(n))
        plt.yticks(range(n))

        plt.xlabel("Seed")
        plt.ylabel("Seed")

        output_path = (
            figures_dir
            / f"ari_heatmap_{parameter_name}_{parameter_value}.png"
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Saved ARI heatmap to: {output_path}")