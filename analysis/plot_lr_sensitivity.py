import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

AGG_METRICS_PATH = Path("analysis/aggregated/aggregated_lr_metrics.json")
FIGURES_DIR = Path("analysis/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

with open(AGG_METRICS_PATH, "r") as f:
    aggregated = json.load(f)


def extract_lr_metric(aggregated, metric_key):
    lrs = []
    means = []
    stds = []

    for lr_str, metrics in aggregated.items():
        lrs.append(float(lr_str))
        means.append(metrics[metric_key]["mean"])
        stds.append(metrics[metric_key]["std"])

    order = np.argsort(lrs)
    return (
        np.array(lrs)[order],
        np.array(means)[order],
        np.array(stds)[order],
    )


def plot_lr_curve(lrs, means, stds, ylabel, title, output_name):
    plt.figure(figsize=(6, 4))

    plt.errorbar(
        lrs,
        means,
        yerr=stds,
        fmt="o-",
        capsize=4,
    )

    plt.xscale("log")
    plt.xlabel("Learning rate (log scale)")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    output_path = FIGURES_DIR / output_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure to: {output_path}")


# ============================
# Figure 1: LR vs Silhouette
# ============================
lrs, means, stds = extract_lr_metric(aggregated, "silhouette")
plot_lr_curve(
    lrs,
    means,
    stds,
    ylabel="Silhouette score",
    title="DEC performance vs learning rate",
    output_name="silhouette_vs_lr.png",
)

# ============================
# Figure 2: LR vs Entropy
# ============================
lrs, means, stds = extract_lr_metric(aggregated, "mean_entropy")
plot_lr_curve(
    lrs,
    means,
    stds,
    ylabel="Mean assignment entropy",
    title="Assignment entropy vs learning rate",
    output_name="entropy_vs_lr.png",
)
