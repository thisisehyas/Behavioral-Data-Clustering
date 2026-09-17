import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)

from src.dec.assets import load_features


# Paths

TRAINABLE_DIR = Path(
    "experiments/encoder_ablation/runs/trainable"
)

IDEC_DIR = Path(
    "experiments/idec_diagnostic/runs/seed_42"
)

PREVIOUS_ANALYSIS_DIR = Path(
    "analysis/encoder_ablation/aggregated"
)

OUTPUT_DIR = Path(
    "analysis/idec_diagnostic"
)

AGGREGATED_DIR = OUTPUT_DIR / "aggregated"
FIGURES_DIR = OUTPUT_DIR / "figures"


# Utility functions

def load_run(run_dir):
    """
    Load observation-level outputs from one experiment.

    These outputs are needed for comparisons such as:
    - ARI
    - cluster profiles
    - latent representation change
    """

    return {
        "labels": np.load(
            run_dir / "full_labels.npy"
        ),
        "q": np.load(
            run_dir / "q_full.npy"
        ),
        "Z": np.load(
            run_dir / "Z_full.npy"
        ),
        "Z_initial": np.load(
            run_dir / "Z_initial.npy"
        ),
        "cluster_centers": np.load(
            run_dir / "cluster_centers.npy"
        ),
        "train_kl": np.load(
            run_dir / "train_kl.npy"
        ),
    }


def load_previous_quality_metrics():
    """
    Load the previously established trainable DEC quality metrics.

    These metrics were computed in the encoder-trainability ablation
    and are reused here without recomputation.
    """
    path = PREVIOUS_ANALYSIS_DIR / "quality_metrics_comparison.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Previous quality metrics not found: {path}"
        )

    df = pd.read_csv(path)

    # Normalize column names because the existing CSV contains
    # formatting artifacts such as * and escaped underscores.
    df.columns = (
        df.columns
        .str.replace("*", "", regex=False)
        .str.replace("\\", "", regex=False)
    )

    required_columns = {
        "condition",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "mean_entropy",
        "normalized_mean_entropy",
        "std_entropy",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {sorted(missing)}"
        )

    trainable = df[df["condition"] == "trainable"].copy()

    if len(trainable) != 1:
        raise ValueError(
            f"Expected exactly one trainable row in {path}, "
            f"found {len(trainable)}."
        )

    return trainable.iloc[0].to_dict()


def load_previous_trainable_kl():
    """
    Load the previously established trainable-DEC
    KL-loss curve.

    These values are NOT recomputed here.
    """

    path = (
        PREVIOUS_ANALYSIS_DIR
        / "training_kl_comparison.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Previous training KL comparison not found: "
            f"{path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "epoch",
        "condition",
        "train_kl",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "training_kl_comparison.csv is missing "
            f"columns: {sorted(missing)}"
        )

    trainable = df[
        df["condition"] == "trainable"
    ].copy()

    if trainable.empty:
        raise ValueError(
            "Could not find the 'trainable' condition "
            "in training_kl_comparison.csv."
        )

    trainable = trainable.sort_values(
        "epoch"
    )

    return trainable["train_kl"].values


def entropy_from_q(q):
    """
    Compute per-sample assignment entropy.
    """

    q_safe = np.clip(
        q,
        1e-12,
        1.0,
    )

    return -np.sum(
        q_safe * np.log(q_safe),
        axis=1,
    )


def calculate_quality_metrics(
    Z,
    labels,
    q,
):
    """
    Calculate clustering quality metrics.

    This function is used ONLY for the new IDEC run.
    """

    n_clusters = len(
        np.unique(labels)
    )

    entropy = entropy_from_q(q)

    return {
        "n_samples": len(Z),
        "n_features": Z.shape[1],
        "n_clusters": n_clusters,
        "silhouette": silhouette_score(
            Z,
            labels,
        ),
        "davies_bouldin": davies_bouldin_score(
            Z,
            labels,
        ),
        "calinski_harabasz": calinski_harabasz_score(
            Z,
            labels,
        ),
        "mean_entropy": np.mean(
            entropy
        ),
        "normalized_mean_entropy": (
            np.mean(entropy)
            / np.log(n_clusters)
        ),
        "std_entropy": np.std(
            entropy
        ),
    }


def calculate_cluster_sizes(
    labels,
    condition,
):
    """
    Calculate cluster counts and percentages.
    """

    unique, counts = np.unique(
        labels,
        return_counts=True,
    )

    n = len(labels)

    rows = []

    for cluster, count in zip(
        unique,
        counts,
    ):
        rows.append(
            {
                "condition": condition,
                "cluster": int(cluster),
                "count": int(count),
                "percentage": (
                    100 * count / n
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_profiles(
    X,
    labels,
    condition,
):
    """
    Calculate cluster means in the original
    standardized seven-feature space.
    """

    feature_names = [
        "Lifestyle",
        "Communication",
        "Responsibility",
        "Openness",
        "Interests",
        "EmotionalStability",
        "SocialEngagement",
    ]

    rows = []

    for cluster in np.unique(labels):

        mask = labels == cluster

        row = {
            "condition": condition,
            "cluster": int(cluster),
            "n": int(mask.sum()),
        }

        for i, feature in enumerate(
            feature_names
        ):
            row[feature] = np.mean(
                X[mask, i]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def match_profiles(
    trainable_profiles,
    idec_profiles,
):
    """
    Match trainable-DEC clusters to IDEC clusters
    using the Hungarian algorithm.

    Matching is based on Euclidean distance between
    seven-dimensional standardized feature profiles.
    """

    feature_names = [
        "Lifestyle",
        "Communication",
        "Responsibility",
        "Openness",
        "Interests",
        "EmotionalStability",
        "SocialEngagement",
    ]

    trainable_clusters = (
        trainable_profiles["cluster"].values
    )

    idec_clusters = (
        idec_profiles["cluster"].values
    )

    trainable_matrix = (
        trainable_profiles[
            feature_names
        ].values
    )

    idec_matrix = (
        idec_profiles[
            feature_names
        ].values
    )

    distance_matrix = np.zeros(
        (
            len(trainable_matrix),
            len(idec_matrix),
        )
    )

    for i in range(
        len(trainable_matrix)
    ):
        for j in range(
            len(idec_matrix)
        ):
            distance_matrix[i, j] = (
                np.linalg.norm(
                    trainable_matrix[i]
                    - idec_matrix[j]
                )
            )

    rows, cols = linear_sum_assignment(
        distance_matrix
    )

    matches = []

    for r, c in zip(rows, cols):

        matches.append(
            {
                "trainable_DEC_cluster": int(
                    trainable_clusters[r]
                ),
                "IDEC_cluster": int(
                    idec_clusters[c]
                ),
                "profile_distance": float(
                    distance_matrix[r, c]
                ),
            }
        )

    return pd.DataFrame(matches)


def calculate_representation_change(
    Z_initial,
    Z_final,
):
    """
    Calculate per-observation Euclidean displacement
    between the initial and final latent representations.
    """

    return np.linalg.norm(
        Z_final - Z_initial,
        axis=1,
    )


def summarize_representation_change(
    displacement,
    condition,
):
    """
    Summarize latent representation displacement.
    """

    return {
        "condition": condition,
        "mean": np.mean(
            displacement
        ),
        "median": np.median(
            displacement
        ),
        "std": np.std(
            displacement
        ),
        "min": np.min(
            displacement
        ),
        "q25": np.percentile(
            displacement,
            25,
        ),
        "q50": np.percentile(
            displacement,
            50,
        ),
        "q75": np.percentile(
            displacement,
            75,
        ),
        "q90": np.percentile(
            displacement,
            90,
        ),
        "q95": np.percentile(
            displacement,
            95,
        ),
        "q99": np.percentile(
            displacement,
            99,
        ),
        "max": np.max(
            displacement
        ),
    }


# Main analysis

def main():

    AGGREGATED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("IDEC vs Trainable DEC Analysis")
    print("=" * 70)

    # Load dataset

    print("\nLoading dataset...")

    X = load_features()

    print(
        f"Dataset shape: {X.shape}"
    )

    # Load trainable DEC reference results

    print(
        "\nLoading trainable DEC observation-level outputs..."
    )

    trainable = load_run(
        TRAINABLE_DIR
    )

    print(
        "Loading previously established "
        "trainable DEC quality metrics..."
    )

    trainable_previous_metrics = (
        load_previous_quality_metrics()
    )

    print(
        "Loading previously established "
        "trainable DEC KL curve..."
    )

    trainable_kl_previous = (
        load_previous_trainable_kl()
    )

    # Load IDEC results

    print(
        "\nLoading IDEC outputs..."
    )

    idec = load_run(
        IDEC_DIR
    )

    # 1. Training behavior

    print("\n" + "=" * 70)
    print("1. TRAINING BEHAVIOR")
    print("=" * 70)

    idec_kl = idec["train_kl"]

    reconstruction = np.load(
        IDEC_DIR
        / "train_reconstruction.npy"
    )

    # KL comparison

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        np.arange(
            1,
            len(trainable_kl_previous) + 1,
        ),
        trainable_kl_previous,
        label="Trainable DEC",
    )

    plt.plot(
        np.arange(
            1,
            len(idec_kl) + 1,
        ),
        idec_kl,
        label="IDEC",
    )

    plt.xlabel("Epoch")
    plt.ylabel("KL loss")
    plt.title(
        "KL Loss: Trainable DEC vs IDEC"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "training_kl_comparison.png",
        dpi=300,
    )

    plt.close()

    # IDEC reconstruction loss

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        np.arange(
            1,
            len(reconstruction) + 1,
        ),
        reconstruction,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction MSE")
    plt.title(
        "IDEC Reconstruction Loss"
    )
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "idec_reconstruction_loss.png",
        dpi=300,
    )

    plt.close()

    print(
        f"Trainable DEC final KL: "
        f"{trainable_kl_previous[-1]:.6f}"
    )

    print(
        f"IDEC final KL: "
        f"{idec_kl[-1]:.6f}"
    )

    print(
        f"IDEC final reconstruction loss: "
        f"{reconstruction[-1]:.6f}"
    )

    # 2. Internal clustering quality

    print("\n" + "=" * 70)
    print("2. INTERNAL CLUSTERING QUALITY")
    print("=" * 70)

    # Existing trainable DEC metrics
    trainable_metrics = {
      "n_samples": len(trainable["Z"]),
      "n_features": trainable["Z"].shape[1],
      "n_clusters": len(np.unique(trainable["labels"])),
      "silhouette": trainable_previous_metrics["silhouette"],
      "davies_bouldin": trainable_previous_metrics["davies_bouldin"],
      "calinski_harabasz": trainable_previous_metrics["calinski_harabasz"],
      "mean_entropy": trainable_previous_metrics["mean_entropy"],
      "normalized_mean_entropy": trainable_previous_metrics[
        "normalized_mean_entropy"
     ],
     "std_entropy": trainable_previous_metrics["std_entropy"],
    }

    # New IDEC metrics
    idec_metrics = (
        calculate_quality_metrics(
            idec["Z"],
            idec["labels"],
            idec["q"],
        )
    )

    quality_df = pd.DataFrame(
        [
            {
                "condition": "trainable_DEC",
                **trainable_metrics,
            },
            {
                "condition": "IDEC",
                **idec_metrics,
            },
        ]
    )

    quality_df.to_csv(
        AGGREGATED_DIR
        / "quality_metrics_comparison.csv",
        index=False,
    )

    print(
        quality_df.to_string(
            index=False
        )
    )

    # 3. Cluster sizes

    print("\n" + "=" * 70)
    print("3. CLUSTER SIZES")
    print("=" * 70)

    trainable_sizes = (
        calculate_cluster_sizes(
            trainable["labels"],
            "trainable_DEC",
        )
    )

    idec_sizes = (
        calculate_cluster_sizes(
            idec["labels"],
            "IDEC",
        )
    )

    sizes_df = pd.concat(
        [
            trainable_sizes,
            idec_sizes,
        ],
        ignore_index=True,
    )

    sizes_df.to_csv(
        AGGREGATED_DIR
        / "cluster_sizes_comparison.csv",
        index=False,
    )

    print(
        sizes_df.to_string(
            index=False
        )
    )

    # 4. Cluster profiles

    print("\n" + "=" * 70)
    print("4. CLUSTER PROFILES")
    print("=" * 70)

    trainable_profiles = (
        calculate_profiles(
            X,
            trainable["labels"],
            "trainable_DEC",
        )
    )

    idec_profiles = (
        calculate_profiles(
            X,
            idec["labels"],
            "IDEC",
        )
    )

    profiles_df = pd.concat(
        [
            trainable_profiles,
            idec_profiles,
        ],
        ignore_index=True,
    )

    profiles_df.to_csv(
        AGGREGATED_DIR
        / "cluster_profiles_comparison.csv",
        index=False,
    )

    print(
        profiles_df.to_string(
            index=False
        )
    )

    # Hungarian profile matching

    profile_matches = match_profiles(
        trainable_profiles,
        idec_profiles,
    )

    profile_matches.to_csv(
        AGGREGATED_DIR
        / "profile_matches.csv",
        index=False,
    )

    print(
        "\nProfile matching:"
    )

    print(
        profile_matches.to_string(
            index=False
        )
    )

    # 5. Partition agreement

    print("\n" + "=" * 70)
    print("5. PARTITION AGREEMENT")
    print("=" * 70)

    ari = adjusted_rand_score(
        trainable["labels"],
        idec["labels"],
    )

    ari_df = pd.DataFrame(
        [
            {
                "comparison": (
                    "trainable_DEC_vs_IDEC"
                ),
                "ARI": ari,
            }
        ]
    )

    ari_df.to_csv(
        AGGREGATED_DIR
        / "partition_ari.csv",
        index=False,
    )

    print(
        f"Adjusted Rand Index: {ari:.6f}"
    )

    # 6. Latent representation change

    print("\n" + "=" * 70)
    print("6. LATENT REPRESENTATION CHANGE")
    print("=" * 70)

    trainable_displacement = (
        calculate_representation_change(
            trainable["Z_initial"],
            trainable["Z"],
        )
    )

    idec_displacement = (
        calculate_representation_change(
            idec["Z_initial"],
            idec["Z"],
        )
    )

    representation_summary = pd.DataFrame(
        [
            summarize_representation_change(
                trainable_displacement,
                "trainable_DEC",
            ),
            summarize_representation_change(
                idec_displacement,
                "IDEC",
            ),
        ]
    )

    representation_summary.to_csv(
        AGGREGATED_DIR
        / "representation_change_summary.csv",
        index=False,
    )

    print(
        representation_summary.to_string(
            index=False
        )
    )

    # Distribution comparison

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        trainable_displacement,
        bins=50,
        alpha=0.6,
        label="Trainable DEC",
    )

    plt.hist(
        idec_displacement,
        bins=50,
        alpha=0.6,
        label="IDEC",
    )

    plt.xlabel(
        "Latent displacement"
    )
    plt.ylabel(
        "Number of observations"
    )
    plt.title(
        "Latent Representation Change"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "representation_change_comparison.png",
        dpi=300,
    )

    plt.close()

    # Final summary

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\nARI (trainable DEC vs IDEC): "
        f"{ari:.6f}"
    )

    print(
        "\nMean latent displacement:"
    )

    print(
        f"  Trainable DEC: "
        f"{np.mean(trainable_displacement):.6f}"
    )

    print(
        f"  IDEC: "
        f"{np.mean(idec_displacement):.6f}"
    )

    print(
        "\nResults saved to:"
    )

    print(
        f"  {AGGREGATED_DIR}"
    )

    print(
        f"  {FIGURES_DIR}"
    )


if __name__ == "__main__":
    main()