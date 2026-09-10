import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# Configuration

CONDITIONS = ["frozen", "trainable"]

ROOT = Path(__file__).resolve().parents[2]

# Encoder ablation experiment outputs
RUNS_DIR = (
    ROOT
    / "experiments"
    / "encoder_ablation"
    / "runs"
)

# Analysis outputs
AGGREGATED_DIR = (
    ROOT
    / "analysis"
    / "encoder_ablation"
    / "aggregated"
)

FIGURES_DIR = (
    ROOT
    / "analysis"
    / "encoder_ablation"
    / "figures"
)

AGGREGATED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Helper functions

def load_run_outputs(condition):
    """Load final outputs for one encoder condition."""

    run_dir = RUNS_DIR / condition

    z_path = run_dir / "Z_full.npy"
    labels_path = run_dir / "full_labels.npy"
    q_path = run_dir / "q_full.npy"

    for path in [
        z_path,
        labels_path,
        q_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found for "
                f"condition={condition}:\n{path}"
            )

    Z = np.load(z_path)
    labels = np.load(labels_path)
    q_full = np.load(q_path)

    return Z, labels, q_full


def validate_run(
    condition,
    Z,
    labels,
    q_full,
):
    """Validate dimensions and basic properties."""

    n_samples = Z.shape[0]

    if Z.ndim != 2:
        raise ValueError(
            f"{condition}: Z_full must be 2-dimensional, "
            f"got shape {Z.shape}"
        )

    if labels.ndim != 1:
        raise ValueError(
            f"{condition}: labels must be 1-dimensional, "
            f"got shape {labels.shape}"
        )

    if len(labels) != n_samples:
        raise ValueError(
            f"{condition}: labels contain {len(labels)} "
            f"samples, but Z contains {n_samples}"
        )

    if q_full.ndim != 2:
        raise ValueError(
            f"{condition}: q_full must be 2-dimensional, "
            f"got shape {q_full.shape}"
        )

    if q_full.shape[0] != n_samples:
        raise ValueError(
            f"{condition}: q_full contains "
            f"{q_full.shape[0]} samples, "
            f"but Z contains {n_samples}"
        )

    n_clusters = len(np.unique(labels))

    if q_full.shape[1] != n_clusters:
        raise ValueError(
            f"{condition}: q_full has "
            f"{q_full.shape[1]} clusters, "
            f"but labels contain {n_clusters} clusters"
        )


def compute_entropy(q_full):
    """
    Compute per-sample assignment entropy.

    H(q_i) = -sum_j q_ij * log(q_ij)
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
):
    """Compute latent-space clustering quality metrics."""

    n_clusters = len(
        np.unique(labels)
    )

    silhouette = silhouette_score(
        Z,
        labels,
        metric="euclidean",
    )

    davies_bouldin = davies_bouldin_score(
        Z,
        labels,
    )

    calinski_harabasz = (
        calinski_harabasz_score(
            Z,
            labels,
        )
    )

    entropy = compute_entropy(
        q_full
    )

    mean_entropy = float(
        np.mean(entropy)
    )

    std_entropy = float(
        np.std(entropy)
    )

    normalized_entropy = (
        mean_entropy
        / np.log(n_clusters)
    )

    return {
        "silhouette": float(silhouette),
        "davies_bouldin": float(
            davies_bouldin
        ),
        "calinski_harabasz": float(
            calinski_harabasz
        ),
        "mean_entropy": mean_entropy,
        "normalized_mean_entropy": float(
            normalized_entropy
        ),
        "std_entropy": std_entropy,
    }


# Main analysis

def main():

    print("=" * 80)
    print(
        "DEC ENCODER TRAINABILITY ABLATION "
        "QUALITY ANALYSIS"
    )
    print("=" * 80)

    all_results = {}

    for condition in CONDITIONS:

        print(
            f"\nAnalyzing condition: {condition}"
        )

        Z, labels, q_full = (
            load_run_outputs(condition)
        )

        validate_run(
            condition,
            Z,
            labels,
            q_full,
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

        metrics = compute_quality_metrics(
            Z,
            labels,
            q_full,
        )

        all_results[condition] = {
            "condition": condition,
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

    # Create comparison DataFrame

    rows = []

    for condition in CONDITIONS:

        result = all_results[condition]

        rows.append({
            "condition": condition,
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
        })

    df = pd.DataFrame(rows)

    # Save CSV

    csv_path = (
        AGGREGATED_DIR
        / "quality_metrics_comparison.csv"
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
        / "quality_metrics_comparison.json"
    )

    json_data = {
        "experiment": (
            "encoder_ablation"
        ),
        "conditions": CONDITIONS,
        "feature_space": {
            "description": (
                "Final DEC latent representation "
                "(Z_full) used for clustering and "
                "internal clustering evaluation."
            )
        },
        "entropy": {
            "mean_entropy": (
                "Raw assignment entropy."
            ),
            "normalized_mean_entropy": (
                "Mean assignment entropy divided "
                "by log(k), where k is the number "
                "of clusters."
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

    # Create comparison figure

    print(
        "\nCreating quality metrics figure..."
    )

    figure_path = (
        FIGURES_DIR
        / "quality_metrics_comparison.png"
    )

    x = np.arange(
        len(CONDITIONS)
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8),
    )

    metric_specs = [
        (
            "silhouette",
            "Silhouette Score",
        ),
        (
            "davies_bouldin",
            "Davies-Bouldin Index",
        ),
        (
            "calinski_harabasz",
            "Calinski-Harabasz Index",
        ),
        (
            "normalized_mean_entropy",
            "Normalized Mean Assignment Entropy",
        ),
    ]

    for ax, (metric, title) in zip(
        axes.flat,
        metric_specs,
    ):

        values = df[metric].values

        ax.bar(
            x,
            values,
        )

        ax.set_title(title)

        ax.set_xticks(x)

        ax.set_xticklabels(
            CONDITIONS
        )

        ax.set_ylabel("Score")

        ax.grid(
            axis="y",
            alpha=0.3,
        )

    fig.suptitle(
        "DEC Encoder Trainability Ablation: "
        "Clustering Quality",
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

    print("\n")
    print("=" * 80)
    print(
        "ENCODER TRAINABILITY QUALITY ANALYSIS COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()