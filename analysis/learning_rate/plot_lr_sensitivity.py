from pathlib import Path

from analysis.plot_sensitivity import plot_sensitivity


PLOTS = [
    (
        "silhouette",
        "Silhouette score",
        "DEC Performance vs Learning Rate",
        "silhouette_vs_learning_rate.png",
    ),
    (
        "mean_entropy",
        "Mean assignment entropy",
        "Assignment Entropy vs Learning Rate",
        "entropy_vs_learning_rate.png",
    ),
]


def main():
    plot_sensitivity(
        aggregated_metrics_path=Path(
            "analysis/learning_rate/aggregated/aggregated_lr_metrics.json"
        ),
        figures_dir=Path(
            "analysis/learning_rate/figures"
        ),
        parameter_name="learning_rate",
        plots=PLOTS,
        log_x_axis=True,
    )


if __name__ == "__main__":
    main()