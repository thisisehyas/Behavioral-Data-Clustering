from pathlib import Path

from analysis.plot_sensitivity import plot_sensitivity


PLOTS = [
    (
        "final_val_kl",
        "Final validation KL",
        "Validation KL vs Update Interval",
        "final_val_kl_vs_update_interval.png",
    ),
    (
        "final_train_kl",
        "Final train KL",
        "Training KL vs Update Interval",
        "final_train_kl_vs_update_interval.png",
    ),
    (
        "silhouette",
        "Silhouette score",
        "Silhouette vs Update Interval",
        "silhouette_vs_update_interval.png",
    ),
    (
        "mean_entropy",
        "Mean assignment entropy",
        "Assignment entropy vs Update Interval",
        "mean_entropy_vs_update_interval.png",
    ),
    (
        "davies_bouldin",
        "Davies-Bouldin index",
        "Davies-Bouldin vs Update Interval",
        "davies_bouldin_vs_update_interval.png",
    ),
    (
        "calinski_harabasz",
        "Calinski-Harabasz score",
        "Calinski-Harabasz vs Update Interval",
        "calinski_harabasz_vs_update_interval.png",
    ),
]


def main():
    plot_sensitivity(
        aggregated_metrics_path=Path(
            "analysis/update_interval/aggregated/aggregated_update_interval_metrics.json"
        ),
        figures_dir=Path(
            "analysis/update_interval/figures"
        ),
        parameter_name="update_interval",
        plots=PLOTS,
        log_x_axis=False,
    )


if __name__ == "__main__":
    main()