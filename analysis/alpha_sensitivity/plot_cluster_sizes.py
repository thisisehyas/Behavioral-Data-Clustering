from pathlib import Path

from analysis.plot_cluster_sizes import plot_cluster_sizes


def main():
    plot_cluster_sizes(
        input_path=Path(
            "analysis/alpha_sensitivity/aggregated/aggregated_cluster_sizes.json"
        ),
        figures_dir=Path(
            "analysis/alpha_sensitivity/figures"
        ),
        parameter_name="alpha",
    )


if __name__ == "__main__":
    main()