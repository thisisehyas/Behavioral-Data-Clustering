from pathlib import Path

from analysis.plot_cluster_sizes import plot_cluster_sizes


def main():
    plot_cluster_sizes(
        input_path=Path(
            "analysis/update_interval/aggregated/aggregated_cluster_sizes.json"
        ),
        figures_dir=Path(
            "analysis/update_interval/figures"
        ),
        parameter_name="update_interval",
    )


if __name__ == "__main__":
    main()