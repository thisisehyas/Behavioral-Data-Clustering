import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

AGG_METRICS_PATH = Path("analysis/alpha_sensitivity/aggregated/aggregated_alpha_metrics.json")
FIGURES_DIR = Path("analysis/alpha_sensitivity/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

with open(AGG_METRICS_PATH, "r") as f:
    aggregated = json.load(f)

def extract_alpha_metric(aggregated, metric_key):
    alphas = []
    means = []
    stds = []

    for alpha_str, metrics in aggregated.items():
        alphas.append(float(alpha_str))
        means.append(metrics[metric_key]["mean"])
        stds.append(metrics[metric_key]["std"])

    order = np.argsort(alphas)
    return (
        np.array(alphas)[order],
        np.array(means)[order],
        np.array(stds)[order],
    )

def plot_alpha_curve(
        alphas,
        means,
        stds,
        ylabel,
        title,
        output_name,
):
    plt.figure(figsize=(6, 4))

    plt.errorbar(
        alphas,
        means,
        yerr=stds,
        fmt="o-",
        capsize=4,
    )

    plt.xlabel("Alpha")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True, linestyle="--", alpha=0.4)

    output_path = FIGURES_DIR / output_name

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure to: {output_path}")

PLOTS = [
    (
        "final_val_kl",
        "Final validation KL",
        "Validation KL vs alpha",
        "final_val_kl_vs_alpha.png",
    ),
    (
        "final_train_kl",
        "Final train KL",
        "Training KL vs alpha",
        "final_train_kl_vs_alpha.png",
    ),
    (
        "silhouette",
        "Silhouette score",
        "Silhouette vs alpha",
        "silhouette_vs_alpha.png",
    ),
     (
        "mean_entropy",
        "Mean assignment entropy",
        "Assignment entropy vs alpha",
        "mean_entropy_vs_alpha.png",
    ),
    (
        "davies_bouldin",
        "Davies-Bouldin index",
        "Davies-Bouldin index vs alpha",
        "davies_bouldin_vs_alpha.png",
    ),
    (
        "calinski_harabasz",
        "Calinski-Harabasz score",
        "Calinski-Harabasz score vs alpha",
        "calinski_harabasz_vs_alpha.png",
    ),
]

def main():
    for metric_key, ylabel, title, filename in PLOTS:
        alphas, means, stds = extract_alpha_metric(
            aggregated,
            metric_key,
        )

        plot_alpha_curve(
            alphas,
            means,
            stds,
            ylabel=ylabel,
            title=title,
            output_name=filename,
        )

if __name__ == "__main__":
    main()

