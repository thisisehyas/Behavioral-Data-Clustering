from pathlib import Path

from analysis.plot_ari_heatmaps import plot_ari_heatmaps


def main():

    plot_ari_heatmaps(
        aggregated_dir=Path(
            "analysis/update_interval/aggregated"
        ),
        figures_dir=Path(
            "analysis/update_interval/figures"
        ),
        parameter_name="update_interval",
    )


if __name__ == "__main__":
    main()