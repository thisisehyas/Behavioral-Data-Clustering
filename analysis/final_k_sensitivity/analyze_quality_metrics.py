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

K_VALUES = [2, 3, 4, 5, 6]

ROOT = Path(__file__).resolve().parents[2]

# k-sensitivity experiment outputs
RUNS_DIR = ROOT / "experiments" / "final_k_sensitivity" / "runs"

# Analysis outputs
AGGREGATED_DIR = (
    ROOT / "analysis" / "final_k_sensitivity" / "aggregated"
)

FIGURES_DIR = (
    ROOT / "analysis" / "final_k_sensitivity" / "figures"
)

# Create output directories if necessary
AGGREGATED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Helper functions

def load_run_outputs(k):
    """Load labels, soft assignments, and KL trajectory for one k."""

    run_dir = RUNS_DIR / f"k_{k}"

    z_path = run_dir / "Z_full.npy"
    labels_path = run_dir / "full_labels.npy"
    q_path = run_dir / "q_full.npy"
    kl_path = run_dir / "train_kl.npy"

    for path in [
        z_path,
        labels_path,
        q_path,
        kl_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found for k={k}:\n{path}"
            )

    Z = np.load(z_path)
    labels = np.load(labels_path)
    q_full = np.load(q_path)
    train_kl = np.load(kl_path)

    return Z, labels, q_full, train_kl


def validate_run(k, Z, labels, q_full, train_kl):
    """Validate dimensions and basic properties of one run."""

    n_samples = Z.shape[0]

    if labels.ndim != 1:
        raise ValueError(
            f"k={k}: labels must be 1-dimensional, "
            f"got shape {labels.shape}"
        )

    if len(labels) != n_samples:
        raise ValueError(
            f"k={k}: labels contain {len(labels)} samples, "
            f"but Z contains {n_samples}"
        )

    if q_full.ndim != 2:
        raise ValueError(
            f"k={k}: q_full must be 2-dimensional, "
            f"got shape {q_full.shape}"
        )

    if q_full.shape[0] != n_samples:
        raise ValueError(
            f"k={k}: q_full contains {q_full.shape[0]} samples, "
            f"but Z contains {n_samples}"
        )

    n_clusters = len(np.unique(labels))

    if q_full.shape[1] != n_clusters:
        raise ValueError(
            f"k={k}: q_full has {q_full.shape[1]} clusters, "
            f"but labels contain {n_clusters} clusters"
        )

    if train_kl.ndim != 1:
        raise ValueError(
            f"k={k}: train_kl must be 1-dimensional, "
            f"got shape {train_kl.shape}"
        )

    if len(train_kl) == 0:
        raise ValueError(
            f"k={k}: train_kl is empty"
        )


def compute_entropy(q_full):
    """
    Compute per-sample assignment entropy.

    H(q_i) = -sum_j q_ij * log(q_ij)

    A small epsilon is used to avoid log(0).
    """

    eps = 1e-12

    q = np.clip(
        q_full,
        eps,
        1.0,
    )

    entropy = -np.sum(
        q * np.log(q),
        axis=1,
    )

    return entropy


def compute_quality_metrics(
    Z,
    labels,
    q_full,
    train_kl,
    k,
):
    """Compute clustering quality and assignment uncertainty metrics."""

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

    mean_entropy = float(
        np.mean(entropy)
    )

    std_entropy = float(
        np.std(entropy)
    )

    # Normalize entropy by its theoretical maximum log(k)
    # so that assignment uncertainty can be compared
    # across different numbers of clusters.

    normalized_entropy = mean_entropy / np.log(k)

    # Training objective

    final_kl = float(
        train_kl[-1]
    )

    return {
        "silhouette": float(silhouette),
        "davies_bouldin": float(davies_bouldin),
        "calinski_harabasz": float(calinski_harabasz),
        "mean_entropy": mean_entropy,
        "normalized_mean_entropy": float(
            normalized_entropy
        ),
        "std_entropy": std_entropy,
        "final_kl": final_kl,
    }


# Main analysis

def main():

    print("=" * 80)
    print(
        "FINAL DEC K-SENSITIVITY, "
        "LATENT-SPACE QUALITY METRICS ACROSS K"
    )
    print("=" * 80)

    print("\nLoading DEC latent embeddings for each k...")

    all_results = {}

    # Analyze each k

    for k in K_VALUES:

        print("\n" + "-" * 80)
        print(f"Analyzing k={k}")
        print("-" * 80)

        Z, labels, q_full, train_kl = load_run_outputs(k)

        validate_run(
            k,
            Z,
            labels,
            q_full,
            train_kl,
        )

        n_clusters = len(
            np.unique(labels)
        )

        print(
            f"Samples: {len(labels):,}"
        )

        print(
            f"Clusters: {n_clusters}"
        )

        print(
            f"Latent dimensions: {Z.shape[1]}"
        )

        print(
            f"KL epochs: {len(train_kl)}"
        )

        print("\nComputing quality metrics...")

        metrics = compute_quality_metrics(
            Z,
            labels,
            q_full,
            train_kl,
            k,
        )

        all_results[str(k)] = {
            "k": k,
            "n_samples": int(
                len(labels)
            ),
            "n_features": int(
                Z.shape[1]
            ),
            "n_clusters": int(
                n_clusters
            ),
            **metrics,
        }

        print(
            f"Silhouette:             "
            f"{metrics['silhouette']:.4f}"
        )

        print(
            f"Davies-Bouldin:         "
            f"{metrics['davies_bouldin']:.4f}"
        )

        print(
            f"Calinski-Harabasz:      "
            f"{metrics['calinski_harabasz']:.2f}"
        )

        print(
            f"Mean entropy:            "
            f"{metrics['mean_entropy']:.4f}"
        )

        print(
            f"Normalized entropy:      "
            f"{metrics['normalized_mean_entropy']:.4f}"
        )

        print(
            f"Std entropy:             "
            f"{metrics['std_entropy']:.4f}"
        )

        print(
            f"Final KL:                "
            f"{metrics['final_kl']:.6f}"
        )

    # Create summary DataFrame

    rows = []

    for k in K_VALUES:

        result = all_results[str(k)]

        rows.append({
            "k": k,
            "silhouette": result[
                "silhouette"
            ],
            "davies_bouldin": result[
                "davies_bouldin"
            ],
            "calinski_harabasz": result[
                "calinski_harabasz"
            ],
            "mean_entropy": result[
                "mean_entropy"
            ],
            "normalized_mean_entropy": result[
                "normalized_mean_entropy"
            ],
            "std_entropy": result[
                "std_entropy"
            ],
            "final_kl": result[
                "final_kl"
            ],
        })

    df = pd.DataFrame(rows)

    # Print comparison table

    print("\n")
    print("=" * 80)
    print("QUALITY METRICS COMPARISON ACROSS K")
    print("=" * 80)

    print(
        df.to_string(
            index=False,
            formatters={
                "silhouette": "{:.4f}".format,
                "davies_bouldin": "{:.4f}".format,
                "calinski_harabasz": "{:.2f}".format,
                "mean_entropy": "{:.4f}".format,
                "normalized_mean_entropy": "{:.4f}".format,
                "std_entropy": "{:.4f}".format,
                "final_kl": "{:.6f}".format,
            },
        )
    )

    # Save CSV

    csv_path = (
        AGGREGATED_DIR
        / "quality_metrics_by_k.csv"
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
        AGGREGATED_DIR
        / "quality_metrics_by_k.json"
    )

    json_data = {
        "experiment": "final_k_sensitivity",
        "k_values": K_VALUES,
        "feature_space": {
            "n_samples": all_results["5"][
                "n_samples"
            ],
            "n_features": all_results["5"][
                "n_features"
            ],
            "description": (
                "DEC latent representation (Z_full) "
                "used for clustering and internal "
                "clustering evaluation."
            ),
        },
        "entropy": {
            "mean_entropy": (
                "Raw assignment entropy. "
                "Its scale depends on k."
            ),
            "normalized_mean_entropy": (
                "Mean assignment entropy divided "
                "by log(k), allowing comparison "
                "across different numbers of clusters."
            ),
        },
        "metrics": all_results,
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

    print(
        "\nCreating quality metrics figure..."
    )

    figure_path = (
        FIGURES_DIR
        / "quality_metrics_by_k.png"
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
    )

    # Silhouette

    axes[0, 0].plot(
        df["k"],
        df["silhouette"],
        marker="o",
    )

    axes[0, 0].set_title(
        "Silhouette Score"
    )

    axes[0, 0].set_xlabel(
        "Number of Clusters (k)"
    )

    axes[0, 0].set_ylabel(
        "Score"
    )

    axes[0, 0].set_xticks(
        K_VALUES
    )

    axes[0, 0].grid(
        True,
        alpha=0.3,
    )

    # Davies-Bouldin

    axes[0, 1].plot(
        df["k"],
        df["davies_bouldin"],
        marker="o",
    )

    axes[0, 1].set_title(
        "Davies-Bouldin Index"
    )

    axes[0, 1].set_xlabel(
        "Number of Clusters (k)"
    )

    axes[0, 1].set_ylabel(
        "Score"
    )

    axes[0, 1].set_xticks(
        K_VALUES
    )

    axes[0, 1].grid(
        True,
        alpha=0.3,
    )

    # Calinski-Harabasz

    axes[1, 0].plot(
        df["k"],
        df["calinski_harabasz"],
        marker="o",
    )

    axes[1, 0].set_title(
        "Calinski-Harabasz Index"
    )

    axes[1, 0].set_xlabel(
        "Number of Clusters (k)"
    )

    axes[1, 0].set_ylabel(
        "Score"
    )

    axes[1, 0].set_xticks(
        K_VALUES
    )

    axes[1, 0].grid(
        True,
        alpha=0.3,
    )

    # Normalized entropy

    axes[1, 1].plot(
        df["k"],
        df["normalized_mean_entropy"],
        marker="o",
    )

    axes[1, 1].set_title(
        "Normalized Mean Assignment Entropy"
    )

    axes[1, 1].set_xlabel(
        "Number of Clusters (k)"
    )

    axes[1, 1].set_ylabel(
        "Normalized Entropy"
    )

    axes[1, 1].set_xticks(
        K_VALUES
    )

    axes[1, 1].grid(
        True,
        alpha=0.3,
    )

    fig.suptitle(
        "Final DEC Latent-Space Quality Metrics Across k",
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
    print(
        "K-SENSITIVITY QUALITY METRICS ANALYSIS COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()