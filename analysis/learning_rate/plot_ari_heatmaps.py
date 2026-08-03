from pathlib import Path

from analysis.plot_ari_heatmaps import plot_ari_heatmaps


def main():

    plot_ari_heatmaps(
        aggregated_dir=Path(
            "analysis/learning_rate/aggregated"
        ),
        figures_dir=Path(
            "analysis/learning_rate/figures"
        ),
        parameter_name="lr",
    )


if __name__ == "__main__":
    main()