from pathlib import Path
import json

import matplotlib.pyplot as plt


RANK_NAMES = [
    "smallest",
    "second_smallest",
    "middle",
    "second_largest",
    "largest",
]


def plot_cluster_sizes(
    input_path: Path,
    figures_dir: Path,
    parameter_name: str,
):
    """
    Plot mean cluster sizes (with std) by cluster rank across
    hyperparameter values.
    """

    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r") as f:
        aggregated = json.load(f)

    parameter_values = sorted(
        float(value)
        for value in aggregated.keys()
    )

    plt.figure(figsize=(7, 5))

    for rank in RANK_NAMES:

        means = []
        stds = []

        for parameter_value in parameter_values:

            stats = aggregated[str(parameter_value)][rank]

            means.append(stats["mean"])
            stds.append(stats["std"])

        plt.errorbar(
            parameter_values,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=rank.replace("_", " ").title(),
        )

    plt.xlabel(parameter_name.replace("_", " ").title())
    plt.ylabel("Cluster size")
    plt.title(
        f"Cluster Size by Rank Across {parameter_name.replace('_', ' ').title()} Values"
    )

    plt.xticks(
        parameter_values,
        [str(v) for v in parameter_values],
    )

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    plt.tight_layout()

    output_path = (
        figures_dir /
        f"cluster_sizes_vs_{parameter_name}.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure to: {output_path}")