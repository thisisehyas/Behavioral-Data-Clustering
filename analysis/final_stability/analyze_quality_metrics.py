from pathlib import Path

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# Configuration

SEEDS = [42, 43, 44, 45, 46]

ROOT = Path(__file__).resolve().parents[2]

# Stability experiment outputs
RUNS_DIR = ROOT / "experiments" / "final_stability" / "runs"

# Analysis outputs
AGGREGATED_DIR = ROOT / "analysis" / "final_stability" / "aggregated"
FIGURES_DIR = ROOT / "analysis" / "final_stability" / "figures"

# Create output directories if necessary
AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Helper functions

def load_run_outputs(seed):
    """Load labels, soft assignments, and KL trajectory for one seed."""

    run_dir = RUNS_DIR / f"seed_{seed}"

    z_path = run_dir / "Z_full.npy"
    labels_path = run_dir / "full_labels.npy"
    q_path = run_dir / "q_full.npy"
    kl_path = run_dir / "train_kl.npy"

    for path in [z_path, labels_path, q_path, kl_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found for seed {seed}:\n{path}"
            )

    Z = np.load(z_path)
    labels = np.load(labels_path)
    q_full = np.load(q_path)
    train_kl = np.load(kl_path)

    return Z, labels, q_full, train_kl


def validate_run(seed, Z, labels, q_full, train_kl):
    """Validate dimensions and basic properties of one run."""

    n_samples = Z.shape[0]

    if labels.ndim != 1:
        raise ValueError(
            f"Seed {seed}: labels must be 1-dimensional, "
            f"got shape {labels.shape}"
        )

    if len(labels) != n_samples:
        raise ValueError(
            f"Seed {seed}: labels contain {len(labels)} samples, "
            f"but Z contains {n_samples}"
        )

    if q_full.ndim != 2:
        raise ValueError(
            f"Seed {seed}: q_full must be 2-dimensional, "
            f"got shape {q_full.shape}"
        )

    if q_full.shape[0] != n_samples:
        raise ValueError(
            f"Seed {seed}: q_full contains {q_full.shape[0]} samples, "
            f"but Z contains {n_samples}"
        )

    if q_full.shape[1] != len(np.unique(labels)):
        raise ValueError(
            f"Seed {seed}: q_full has {q_full.shape[1]} clusters, "
            f"but labels contain {len(np.unique(labels))} clusters"
        )

    if train_kl.ndim != 1:
        raise ValueError(
            f"Seed {seed}: train_kl must be 1-dimensional, "
            f"got shape {train_kl.shape}"
        )

    if len(train_kl) == 0:
        raise ValueError(
            f"Seed {seed}: train_kl is empty"
        )


def compute_entropy(q_full):
    """
    Compute per-sample assignment entropy.

    H(q_i) = -sum_j q_ij * log(q_ij)

    A small epsilon is used to avoid log(0).
    """

    eps = 1e-12

    q = np.clip(q_full, eps, 1.0)

    entropy = -np.sum(q * np.log(q), axis=1)

    return entropy


def compute_cluster_sizes(labels):
    """Return cluster counts and percentages."""

    unique, counts = np.unique(labels, return_counts=True)

    total = len(labels)

    sizes = {
        int(cluster): {
            "count": int(count),
            "percentage": float(100.0 * count / total),
        }
        for cluster, count in zip(unique, counts)
    }

    return sizes


def compute_quality_metrics(Z, labels, q_full, train_kl):
    """Compute all quality metrics for one run."""

    # Hard clustering quality

    silhouette = silhouette_score(
        Z,
        labels,
        metric="euclidean",
    )

    davies_bouldin = davies_bouldin_score(
        Z,
        labels,
    )

    calinski_harabasz = calinski_harabasz_score(
        Z,
        labels,
    )

    # Assignment uncertainty

    entropy = compute_entropy(q_full)

    mean_entropy = float(np.mean(entropy))
    std_entropy = float(np.std(entropy))

    # Training objective

    final_kl = float(train_kl[-1])

    # Cluster sizes

    cluster_sizes = compute_cluster_sizes(labels)

    counts = [info["count"] for info in cluster_sizes.values()]

    min_cluster_size = int(min(counts))
    max_cluster_size = int(max(counts))

    min_cluster_percentage = float(
        100.0 * min_cluster_size / len(labels)
    )

    max_cluster_percentage = float(
        100.0 * max_cluster_size / len(labels)
    )

    size_ratio = float(
        max_cluster_size / min_cluster_size
    )

    return {
        "silhouette": float(silhouette),
        "davies_bouldin": float(davies_bouldin),
        "calinski_harabasz": float(calinski_harabasz),
        "mean_entropy": mean_entropy,
        "std_entropy": std_entropy,
        "final_kl": final_kl,
        "min_cluster_size": min_cluster_size,
        "max_cluster_size": max_cluster_size,
        "min_cluster_percentage": min_cluster_percentage,
        "max_cluster_percentage": max_cluster_percentage,
        "cluster_size_ratio": size_ratio,
        "cluster_sizes": cluster_sizes,
    }


# Main analysis

def main():

    print("=" * 80)
    print("FINAL DEC STABILITY, LATENT-SPACE QUALITY METRICS ACROSS SEEDS")
    print("=" * 80)

    print("\nLoading DEC latent embeddings for each seed...")

    all_results = {}

    # Analyze each seed

    for seed in SEEDS:

        print("\n" + "-" * 80)
        print(f"Analyzing seed {seed}")
        print("-" * 80)

        Z, labels, q_full, train_kl = load_run_outputs(seed)

        validate_run(
            seed,
            Z,
            labels,
            q_full,
            train_kl,
        )

        n_clusters = len(np.unique(labels))

        print(f"Samples: {len(labels):,}")
        print(f"Clusters: {n_clusters}")
        print(f"KL epochs: {len(train_kl)}")

        print("\nComputing quality metrics...")

        metrics = compute_quality_metrics(
            Z,
            labels,
            q_full,
            train_kl,
        )

        all_results[str(seed)] = {
            "seed": seed,
            "n_samples": int(len(labels)),
            "n_features": int(Z.shape[1]),
            "n_clusters": int(n_clusters),
            **metrics,
        }

        print(f"Silhouette:             {metrics['silhouette']:.4f}")
        print(f"Davies-Bouldin:         {metrics['davies_bouldin']:.4f}")
        print(
            f"Calinski-Harabasz:      "
            f"{metrics['calinski_harabasz']:.2f}"
        )
        print(
            f"Mean entropy:            "
            f"{metrics['mean_entropy']:.4f}"
        )
        print(
            f"Std entropy:             "
            f"{metrics['std_entropy']:.4f}"
        )
        print(
            f"Final KL:                "
            f"{metrics['final_kl']:.6f}"
        )
        print(
            f"Min cluster:             "
            f"{metrics['min_cluster_size']:,} "
            f"({metrics['min_cluster_percentage']:.2f}%)"
        )
        print(
            f"Max cluster:             "
            f"{metrics['max_cluster_size']:,} "
            f"({metrics['max_cluster_percentage']:.2f}%)"
        )
        print(
            f"Max/min cluster ratio:   "
            f"{metrics['cluster_size_ratio']:.2f}"
        )

    # Create summary DataFrame

    rows = []

    for seed in SEEDS:

        result = all_results[str(seed)]

        rows.append({
            "seed": seed,
            "silhouette": result["silhouette"],
            "davies_bouldin": result["davies_bouldin"],
            "calinski_harabasz": result["calinski_harabasz"],
            "mean_entropy": result["mean_entropy"],
            "std_entropy": result["std_entropy"],
            "final_kl": result["final_kl"],
            "min_cluster_size": result["min_cluster_size"],
            "min_cluster_percentage": result[
                "min_cluster_percentage"
            ],
            "max_cluster_size": result["max_cluster_size"],
            "max_cluster_percentage": result[
                "max_cluster_percentage"
            ],
            "cluster_size_ratio": result["cluster_size_ratio"],
        })

    df = pd.DataFrame(rows)

    # Print comparison table

    print("\n")
    print("=" * 80)
    print("QUALITY METRICS COMPARISON")
    print("=" * 80)

    print(
        df.to_string(
            index=False,
            formatters={
                "silhouette": "{:.4f}".format,
                "davies_bouldin": "{:.4f}".format,
                "calinski_harabasz": "{:.2f}".format,
                "mean_entropy": "{:.4f}".format,
                "std_entropy": "{:.4f}".format,
                "final_kl": "{:.6f}".format,
                "min_cluster_percentage": "{:.2f}".format,
                "max_cluster_percentage": "{:.2f}".format,
                "cluster_size_ratio": "{:.2f}".format,
            },
        )
    )

    # Summary statistics across seeds

    metric_names = [
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "mean_entropy",
        "std_entropy",
        "final_kl",
    ]

    summary = {}

    for metric in metric_names:

        values = df[metric].to_numpy(dtype=float)

        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    print("\n")
    print("=" * 80)
    print("SUMMARY ACROSS FIVE SEEDS")
    print("=" * 80)

    for metric in metric_names:

        stats = summary[metric]

        print(
            f"{metric:25s} "
            f"mean={stats['mean']:.6f}  "
            f"std={stats['std']:.6f}  "
            f"min={stats['min']:.6f}  "
            f"max={stats['max']:.6f}"
        )

    # Save CSV

    csv_path = (
        AGGREGATED_DIR /
        "quality_metrics_by_seed.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    print(
        f"\nSaved CSV results to:\n{csv_path}"
    )

    # Save JSON

    json_path = (
        AGGREGATED_DIR /
        "quality_metrics_by_seed.json"
    )

    json_data = {
        "experiment": "final_stability",
        "seeds": SEEDS,
        "feature_space": {
            "n_samples": all_results["42"]["n_samples"],
            "n_features": all_results["42"]["n_features"],
            "description": (
                "DEC latent representation (Z_full) used for "
                "clustering and internal clustering evaluation."
            ),
        },
        "metrics": all_results,
        "summary_across_seeds": summary,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_data,
            f,
            indent=2,
        )

    print(
        f"Saved JSON results to:\n{json_path}"
    )

    # Create figure

    print("\nCreating quality metrics figure...")

    figure_path = (
        FIGURES_DIR /
        "quality_metrics_by_seed.png"
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
    )

    # Silhouette

    axes[0, 0].plot(
        df["seed"],
        df["silhouette"],
        marker="o",
    )

    axes[0, 0].set_title(
        "Silhouette Score"
    )

    axes[0, 0].set_xlabel("Seed")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_xticks(SEEDS)
    axes[0, 0].grid(True, alpha=0.3)

    # Davies-Bouldin

    axes[0, 1].plot(
        df["seed"],
        df["davies_bouldin"],
        marker="o",
    )

    axes[0, 1].set_title(
        "Davies-Bouldin Index"
    )

    axes[0, 1].set_xlabel("Seed")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_xticks(SEEDS)
    axes[0, 1].grid(True, alpha=0.3)

    # Calinski-Harabasz

    axes[1, 0].plot(
        df["seed"],
        df["calinski_harabasz"],
        marker="o",
    )

    axes[1, 0].set_title(
        "Calinski-Harabasz Index"
    )

    axes[1, 0].set_xlabel("Seed")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_xticks(SEEDS)
    axes[1, 0].grid(True, alpha=0.3)

    # Mean entropy

    axes[1, 1].plot(
        df["seed"],
        df["mean_entropy"],
        marker="o",
    )

    axes[1, 1].set_title(
        "Mean Assignment Entropy"
    )

    axes[1, 1].set_xlabel("Seed")
    axes[1, 1].set_ylabel("Entropy")
    axes[1, 1].set_xticks(SEEDS)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        "Final DEC Latent-Space Quality Metrics Across Random Seeds",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved figure to:\n{figure_path}"
    )

    # Complete

    print("\n")
    print("=" * 80)
    print("QUALITY METRICS ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()