import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def extract_metric(
    aggregated: dict,
    metric_key: str,
):
    """
    Extract parameter values, means and standard deviations for one metric.
    """

    parameter_values = []
    means = []
    stds = []

    for parameter, metrics in aggregated.items():
        parameter_values.append(float(parameter))
        means.append(metrics[metric_key]["mean"])
        stds.append(metrics[metric_key]["std"])

    order = np.argsort(parameter_values)

    return (
        np.array(parameter_values)[order],
        np.array(means)[order],
        np.array(stds)[order],
    )


def plot_curve(
    parameter_values,
    means,
    stds,
    parameter_name,
    ylabel,
    title,
    output_path,
    log_x_axis=False,
):
    plt.figure(figsize=(6, 4))

    plt.errorbar(
        parameter_values,
        means,
        yerr=stds,
        fmt="o-",
        capsize=4,
    )

    if log_x_axis:
        plt.xscale("log")

    plt.xlabel(parameter_name.replace("_", " ").title())
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(
        True,
        which="both" if log_x_axis else "major",
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure to: {output_path}")


def plot_sensitivity(
    aggregated_metrics_path: Path,
    figures_dir: Path,
    parameter_name: str,
    plots: list,
    log_x_axis: bool = False,
):
    """
    Generate all sensitivity plots for one experiment.
    """

    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(aggregated_metrics_path, "r") as f:
        aggregated = json.load(f)

    for metric_key, ylabel, title, filename in plots:

        values, means, stds = extract_metric(
            aggregated,
            metric_key,
        )

        plot_curve(
            parameter_values=values,
            means=means,
            stds=stds,
            parameter_name=parameter_name,
            ylabel=ylabel,
            title=title,
            output_path=figures_dir / filename,
            log_x_axis=log_x_axis,
        )