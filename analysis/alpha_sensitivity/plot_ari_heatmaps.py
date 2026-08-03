from pathlib import Path

from analysis.plot_ari_heatmaps import plot_ari_heatmaps


def main():

    plot_ari_heatmaps(
        aggregated_dir=Path(
            "analysis/alpha_sensitivity/aggregated"
        ),
        figures_dir=Path(
            "analysis/alpha_sensitivity/figures"
        ),
        parameter_name="alpha",
    )


if __name__ == "__main__":
    main()