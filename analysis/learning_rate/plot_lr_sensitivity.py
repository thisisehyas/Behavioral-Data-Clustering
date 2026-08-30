from pathlib import Path

from analysis.plot_sensitivity import plot_sensitivity


PLOTS = [
    (
        "silhouette",
        "Silhouette score",
        "Silhouette vs Learning Rate",
        "silhouette_vs_learning_rate.png",
    ),
    (
        "davies_bouldin",
        "Davies-Bouldin index",
        "Davies-Bouldin Index vs Learning Rate",
        "davies_bouldin_vs_learning_rate.png",
    ),
    (
        "calinski_harabasz",
        "Calinski-Harabasz score",
        "Calinski-Harabasz Score vs Learning Rate",
        "calinski_harabasz_vs_learning_rate.png",
    ),
    (
        "mean_entropy",
        "Mean assignment entropy",
        "Assignment Entropy vs Learning Rate",
        "mean_entropy_vs_learning_rate.png",
    ),
    (
        "final_train_kl",
        "Final training KL",
        "Training KL vs Learning Rate",
        "final_train_kl_vs_learning_rate.png",
    ),
    (
        "final_val_kl",
        "Final validation KL",
        "Validation KL vs Learning Rate",
        "final_val_kl_vs_learning_rate.png",
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