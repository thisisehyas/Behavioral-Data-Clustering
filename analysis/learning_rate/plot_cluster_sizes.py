from pathlib import Path

from analysis.plot_cluster_sizes import plot_cluster_sizes


def main():
    plot_cluster_sizes(
        input_path=Path(
            "analysis/learning_rate/aggregated/aggregated_cluster_sizes.json"
        ),
        figures_dir=Path(
            "analysis/learning_rate/figures"
        ),
        parameter_name="learning_rate",
    )


if __name__ == "__main__":
    main()