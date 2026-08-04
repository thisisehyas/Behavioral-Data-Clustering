from pathlib import Path

from analysis.plot_sensitivity import plot_sensitivity


PLOTS = [
    (
        "final_val_kl",
        "Final validation KL",
        "Validation KL vs Alpha",
        "final_val_kl_vs_alpha.png",
    ),
    (
        "final_train_kl",
        "Final train KL",
        "Training KL vs Alpha",
        "final_train_kl_vs_alpha.png",
    ),
    (
        "silhouette",
        "Silhouette score",
        "Silhouette vs Alpha",
        "silhouette_vs_alpha.png",
    ),
    (
        "mean_entropy",
        "Mean assignment entropy",
        "Assignment entropy vs Alpha",
        "mean_entropy_vs_alpha.png",
    ),
    (
        "davies_bouldin",
        "Davies-Bouldin index",
        "Davies-Bouldin vs Alpha",
        "davies_bouldin_vs_alpha.png",
    ),
    (
        "calinski_harabasz",
        "Calinski-Harabasz score",
        "Calinski-Harabasz vs Alpha",
        "calinski_harabasz_vs_alpha.png",
    ),
]


def main():
    plot_sensitivity(
        aggregated_metrics_path=Path(
            "analysis/alpha_sensitivity/aggregated/aggregated_alpha_metrics.json"
        ),
        figures_dir=Path(
            "analysis/alpha_sensitivity/figures"
        ),
        parameter_name="alpha",
        plots=PLOTS,
        log_x_axis=False,
    )


if __name__ == "__main__":
    main()