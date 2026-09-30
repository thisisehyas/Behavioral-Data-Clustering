import json
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

EXPERIMENT_DIR = Path(
    "experiments/idec_stability/runs"
)

OUTPUT_DIR = Path(
    "analysis/idec_stability"
)

AGGREGATED_DIR = OUTPUT_DIR / "aggregated"
FIGURES_DIR = OUTPUT_DIR / "figures"

SEEDS = [42, 43, 44, 45, 46]

REFERENCE_SEED = 42

FEATURE_NAMES = [
    "Lifestyle",
    "Communication",
    "Responsibility",
    "Openness",
    "Interests",
    "EmotionalStability",
    "SocialEngagement",
]


# Utility functions

def load_run(run_dir):
    """
    Load observation-level and training outputs from one IDEC run.
    """

    required_files = [
        "full_labels.npy",
        "q_full.npy",
        "Z_full.npy",
        "Z_initial.npy",
        "cluster_centers.npy",
        "train_kl.npy",
        "train_reconstruction.npy",
        "meta.json",
    ]

    for filename in required_files:
        path = run_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    with open(run_dir / "meta.json", "r") as f:
        metadata = json.load(f)

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
        "train_reconstruction": np.load(
            run_dir / "train_reconstruction.npy"
        ),
        "metadata": metadata,
    }


def load_all_runs():
    """
    Load all IDEC stability runs.
    """

    runs = {}

    for seed in SEEDS:

        run_dir = (
            EXPERIMENT_DIR
            / f"seed_{seed}"
        )

        if not run_dir.exists():
            raise FileNotFoundError(
                f"Run directory not found: {run_dir}"
            )

        print(
            f"Loading seed {seed}..."
        )

        runs[seed] = load_run(
            run_dir
        )

    return runs


def validate_runs(runs, X):
    """
    Validate that all runs contain compatible outputs.
    """

    expected_n = X.shape[0]

    for seed, run in runs.items():

        labels = run["labels"]
        q = run["q"]
        Z = run["Z"]
        Z_initial = run["Z_initial"]

        if len(labels) != expected_n:
            raise ValueError(
                f"Seed {seed}: label count "
                f"{len(labels)} != dataset size "
                f"{expected_n}."
            )

        if q.shape[0] != expected_n:
            raise ValueError(
                f"Seed {seed}: q_full has "
                f"{q.shape[0]} observations."
            )

        if Z.shape[0] != expected_n:
            raise ValueError(
                f"Seed {seed}: Z_full has "
                f"{Z.shape[0]} observations."
            )

        if Z_initial.shape != Z.shape:
            raise ValueError(
                f"Seed {seed}: Z_initial shape "
                f"{Z_initial.shape} does not match "
                f"Z_full shape {Z.shape}."
            )

        if X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(FEATURE_NAMES)} input features, "
                f"found {X.shape[1]}."
            )

        if q.shape[1] != len(
            np.unique(labels)
        ):
            raise ValueError(
                f"Seed {seed}: q_full has "
                f"{q.shape[1]} clusters, while labels "
                f"contain {len(np.unique(labels))} clusters."
            )


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


# Training behavior

def analyze_training_behavior(runs):
    """
    Aggregate KL and reconstruction losses across seeds.
    """

    rows = []

    for seed, run in runs.items():

        kl = run["train_kl"]
        reconstruction = run[
            "train_reconstruction"
        ]

        if len(kl) != len(reconstruction):
            raise ValueError(
                f"Seed {seed}: KL and reconstruction "
                f"curves have different lengths."
            )

        for epoch in range(len(kl)):

            rows.append(
                {
                    "seed": seed,
                    "epoch": epoch + 1,
                    "train_kl": float(
                        kl[epoch]
                    ),
                    "train_reconstruction": float(
                        reconstruction[epoch]
                    ),
                }
            )

    return pd.DataFrame(rows)


# Internal clustering quality

def calculate_quality_metrics(
    Z,
    labels,
    q,
):
    """
    Calculate internal clustering quality metrics
    for one IDEC run.
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


def analyze_quality(runs):
    """
    Calculate quality metrics independently for every seed.
    """

    rows = []

    for seed, run in runs.items():

        metrics = calculate_quality_metrics(
            run["Z"],
            run["labels"],
            run["q"],
        )

        metrics["seed"] = seed

        rows.append(metrics)

    columns = [
        "seed",
        "n_samples",
        "n_features",
        "n_clusters",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "mean_entropy",
        "normalized_mean_entropy",
        "std_entropy",
    ]

    return pd.DataFrame(
        rows,
        columns=columns,
    )


# Cluster sizes

def calculate_cluster_sizes(
    labels,
    seed,
):
    """
    Calculate cluster counts and percentages
    for one seed.
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
                "seed": seed,
                "cluster": int(cluster),
                "count": int(count),
                "percentage": (
                    100.0 * count / n
                ),
            }
        )

    return pd.DataFrame(rows)


def analyze_cluster_sizes(runs):
    """
    Aggregate cluster sizes across all seeds.
    """

    frames = []

    for seed, run in runs.items():

        frames.append(
            calculate_cluster_sizes(
                run["labels"],
                seed,
            )
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# Cluster profiles

def calculate_profiles(
    X,
    labels,
    seed,
):
    """
    Calculate cluster means in the original
    standardized seven-feature space.
    """

    rows = []

    for cluster in np.unique(labels):

        mask = labels == cluster

        row = {
            "seed": seed,
            "cluster": int(cluster),
            "n": int(mask.sum()),
        }

        for i, feature in enumerate(
            FEATURE_NAMES
        ):

            row[feature] = float(
                np.mean(X[mask, i])
            )

        rows.append(row)

    return pd.DataFrame(rows)


def analyze_cluster_profiles(
    X,
    runs,
):
    """
    Calculate cluster profiles for every seed.
    """

    frames = []

    for seed, run in runs.items():

        frames.append(
            calculate_profiles(
                X,
                run["labels"],
                seed,
            )
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def calculate_profile_distance(
    profile_a,
    profile_b,
):
    """
    Euclidean distance between two
    seven-dimensional cluster profiles.
    """

    vector_a = profile_a[
        FEATURE_NAMES
    ].values.astype(float)

    vector_b = profile_b[
        FEATURE_NAMES
    ].values.astype(float)

    return float(
        np.linalg.norm(
            vector_a - vector_b
        )
    )


def match_profiles(
    reference_profiles,
    target_profiles,
    reference_seed,
    target_seed,
):
    """
    Match target-seed clusters to reference-seed
    clusters using the Hungarian algorithm.

    Matching is based on Euclidean distance between
    seven-dimensional standardized feature profiles.
    """

    reference_clusters = (
        reference_profiles["cluster"].values
    )

    target_clusters = (
        target_profiles["cluster"].values
    )

    reference_matrix = (
        reference_profiles[
            FEATURE_NAMES
        ].values.astype(float)
    )

    target_matrix = (
        target_profiles[
            FEATURE_NAMES
        ].values.astype(float)
    )

    distance_matrix = np.zeros(
        (
            len(reference_matrix),
            len(target_matrix),
        )
    )

    for i in range(
        len(reference_matrix)
    ):

        for j in range(
            len(target_matrix)
        ):

            distance_matrix[i, j] = (
                np.linalg.norm(
                    reference_matrix[i]
                    - target_matrix[j]
                )
            )

    rows, cols = linear_sum_assignment(
        distance_matrix
    )

    matches = []

    for r, c in zip(rows, cols):

        matches.append(
            {
                "reference_seed": reference_seed,
                "reference_cluster": int(
                    reference_clusters[r]
                ),
                "target_seed": target_seed,
                "target_cluster": int(
                    target_clusters[c]
                ),
                "profile_distance": float(
                    distance_matrix[r, c]
                ),
            }
        )

    return pd.DataFrame(matches)


def analyze_profile_stability(
    profiles_df,
):
    """
    Match every non-reference seed to seed 42
    and calculate profile distances.
    """

    reference_profiles = profiles_df[
        profiles_df["seed"]
        == REFERENCE_SEED
    ].copy()

    if reference_profiles.empty:
        raise ValueError(
            f"Reference seed {REFERENCE_SEED} "
            "not found in cluster profiles."
        )

    frames = []

    for seed in SEEDS:

        if seed == REFERENCE_SEED:
            continue

        target_profiles = profiles_df[
            profiles_df["seed"]
            == seed
        ].copy()

        matches = match_profiles(
            reference_profiles,
            target_profiles,
            REFERENCE_SEED,
            seed,
        )

        frames.append(matches)

    if not frames:
        return pd.DataFrame(
            columns=[
                "reference_seed",
                "reference_cluster",
                "target_seed",
                "target_cluster",
                "profile_distance",
            ]
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# Partition stability

def calculate_pairwise_ari(runs):
    """
    Calculate the pairwise ARI matrix across all seeds.
    """

    matrix = np.zeros(
        (
            len(SEEDS),
            len(SEEDS),
        )
    )

    for i, seed_a in enumerate(SEEDS):

        for j, seed_b in enumerate(SEEDS):

            matrix[i, j] = (
                adjusted_rand_score(
                    runs[seed_a]["labels"],
                    runs[seed_b]["labels"],
                )
            )

    return pd.DataFrame(
        matrix,
        index=SEEDS,
        columns=SEEDS,
    )


def summarize_ari(ari_matrix):
    """
    Summarize the off-diagonal pairwise ARI values.
    """

    values = []

    for i in range(
        len(SEEDS)
    ):

        for j in range(i + 1, len(SEEDS)):

            values.append(
                float(
                    ari_matrix.iloc[i, j]
                )
            )

    values = np.array(values)

    return {
        "reference_seed": REFERENCE_SEED,
        "n_seeds": len(SEEDS),
        "n_pairwise_comparisons": len(values),
        "mean_ARI": float(
            np.mean(values)
        ),
        "std_ARI": float(
            np.std(values)
        ),
        "min_ARI": float(
            np.min(values)
        ),
        "max_ARI": float(
            np.max(values)
        ),
        "median_ARI": float(
            np.median(values)
        ),
    }


# Latent representation change

def calculate_representation_change(
    Z_initial,
    Z_final,
):
    """
    Calculate per-observation Euclidean displacement
    between initial and final latent representations.
    """

    return np.linalg.norm(
        Z_final - Z_initial,
        axis=1,
    )


def summarize_representation_change(
    displacement,
    seed,
):
    """
    Summarize latent representation displacement.
    """

    return {
        "seed": seed,
        "mean": float(
            np.mean(displacement)
        ),
        "median": float(
            np.median(displacement)
        ),
        "std": float(
            np.std(displacement)
        ),
        "min": float(
            np.min(displacement)
        ),
        "q25": float(
            np.percentile(
                displacement,
                25,
            )
        ),
        "q50": float(
            np.percentile(
                displacement,
                50,
            )
        ),
        "q75": float(
            np.percentile(
                displacement,
                75,
            )
        ),
        "q90": float(
            np.percentile(
                displacement,
                90,
            )
        ),
        "q95": float(
            np.percentile(
                displacement,
                95,
            )
        ),
        "q99": float(
            np.percentile(
                displacement,
                99,
            )
        ),
        "max": float(
            np.max(displacement)
        ),
    }


def analyze_representation_change(
    runs,
):
    """
    Calculate representation displacement
    independently for every seed.
    """

    rows = []

    displacement_by_seed = {}

    for seed, run in runs.items():

        displacement = (
            calculate_representation_change(
                run["Z_initial"],
                run["Z"],
            )
        )

        displacement_by_seed[seed] = (
            displacement
        )

        rows.append(
            summarize_representation_change(
                displacement,
                seed,
            )
        )

    return (
        pd.DataFrame(rows),
        displacement_by_seed,
    )


# Plotting helpers

def plot_training_kl(
    training_df,
):
    """
    Plot KL-loss curves for all seeds.
    """

    plt.figure(
        figsize=(8, 5)
    )

    for seed in SEEDS:

        subset = training_df[
            training_df["seed"] == seed
        ]

        plt.plot(
            subset["epoch"],
            subset["train_kl"],
            label=f"Seed {seed}",
        )

    plt.xlabel("Epoch")
    plt.ylabel("KL loss")
    plt.title(
        "IDEC KL Loss Across Random Seeds"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "training_kl_curves.png",
        dpi=300,
    )

    plt.close()


def plot_reconstruction_loss(
    training_df,
):
    """
    Plot reconstruction-loss curves for all seeds.
    """

    plt.figure(
        figsize=(8, 5)
    )

    for seed in SEEDS:

        subset = training_df[
            training_df["seed"] == seed
        ]

        plt.plot(
            subset["epoch"],
            subset["train_reconstruction"],
            label=f"Seed {seed}",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction MSE")
    plt.title(
        "IDEC Reconstruction Loss Across Random Seeds"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "reconstruction_loss_curves.png",
        dpi=300,
    )

    plt.close()


def plot_quality_metrics(
    quality_df,
):
    """
    Plot the main clustering quality metrics
    across seeds.
    """

    metrics = [
        (
            "silhouette",
            "Silhouette Score",
        ),
        (
            "davies_bouldin",
            "Davies–Bouldin Index",
        ),
        (
            "calinski_harabasz",
            "Calinski–Harabasz Score",
        ),
        (
            "normalized_mean_entropy",
            "Normalized Mean Entropy",
        ),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8),
    )

    axes = axes.flatten()

    for ax, (column, title) in zip(
        axes,
        metrics,
    ):

        ax.plot(
            quality_df["seed"],
            quality_df[column],
            marker="o",
        )

        ax.set_xlabel("Seed")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(SEEDS)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "quality_metrics_across_seeds.png",
        dpi=300,
    )

    plt.close()


def plot_cluster_sizes(
    sizes_df,
):
    """
    Plot cluster percentages across seeds.
    """

    plt.figure(
        figsize=(9, 5)
    )

    for cluster in sorted(
        sizes_df["cluster"].unique()
    ):

        subset = sizes_df[
            sizes_df["cluster"] == cluster
        ]

        plt.plot(
            subset["seed"],
            subset["percentage"],
            marker="o",
            label=f"Cluster {cluster}",
        )

    plt.xlabel("Seed")
    plt.ylabel("Cluster size (%)")
    plt.title(
        "IDEC Cluster Sizes Across Random Seeds"
    )
    plt.xticks(SEEDS)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "cluster_sizes_across_seeds.png",
        dpi=300,
    )

    plt.close()


def plot_ari_heatmap(
    ari_matrix,
):
    """
    Plot pairwise ARI heatmap.
    """

    plt.figure(
        figsize=(7, 6)
    )

    image = plt.imshow(
        ari_matrix.values,
        vmin=0,
        vmax=1,
        aspect="equal",
    )

    plt.colorbar(
        image,
        label="Adjusted Rand Index",
    )

    plt.xticks(
        range(len(SEEDS)),
        SEEDS,
    )

    plt.yticks(
        range(len(SEEDS)),
        SEEDS,
    )

    for i in range(len(SEEDS)):

        for j in range(len(SEEDS)):

            plt.text(
                j,
                i,
                f"{ari_matrix.iloc[i, j]:.3f}",
                ha="center",
                va="center",
            )

    plt.xlabel("Seed")
    plt.ylabel("Seed")
    plt.title(
        "Pairwise ARI Across IDEC Seeds"
    )
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "ari_heatmap.png",
        dpi=300,
    )

    plt.close()


def plot_cluster_profiles(
    profiles_df,
):
    """
    Plot seven-feature cluster profiles for
    all seeds.

    Cluster IDs are shown directly because the
    plot is descriptive; cross-seed profile matching
    is handled separately using the Hungarian algorithm.
    """

    n_clusters = profiles_df[
        "cluster"
    ].nunique()

    fig, axes = plt.subplots(
        n_clusters,
        1,
        figsize=(10, 3 * n_clusters),
        sharex=True,
    )

    if n_clusters == 1:
        axes = [axes]

    x = np.arange(
        len(FEATURE_NAMES)
    )

    for ax, cluster in zip(
        axes,
        sorted(
            profiles_df["cluster"].unique()
        ),
    ):

        for seed in SEEDS:

            subset = profiles_df[
                (
                    profiles_df["seed"]
                    == seed
                )
                &
                (
                    profiles_df["cluster"]
                    == cluster
                )
            ]

            if subset.empty:
                continue

            values = subset[
                FEATURE_NAMES
            ].iloc[0].values

            ax.plot(
                x,
                values,
                marker="o",
                label=f"Seed {seed}",
            )

        ax.axhline(
            0,
            linewidth=0.8,
        )

        ax.set_ylabel(
            f"Cluster {cluster}"
        )

        ax.legend(
            ncol=3,
            fontsize=8,
        )

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        FEATURE_NAMES,
        rotation=30,
        ha="right",
    )

    fig.suptitle(
        "IDEC Cluster Profiles Across Random Seeds"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "cluster_profiles_across_seeds.png",
        dpi=300,
    )

    plt.close()


def plot_representation_displacement(
    displacement_by_seed,
):
    """
    Plot latent displacement distributions across seeds.
    """

    data = [
        displacement_by_seed[seed]
        for seed in SEEDS
    ]

    plt.figure(
        figsize=(9, 5)
    )

    plt.boxplot(
        data,
        labels=[
            str(seed)
            for seed in SEEDS
        ],
        showfliers=False,
    )

    plt.xlabel("Seed")
    plt.ylabel(
        "Latent displacement"
    )
    plt.title(
        "Latent Representation Displacement Across IDEC Seeds"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "representation_displacement.png",
        dpi=300,
    )

    plt.close()


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
    print("IDEC STABILITY ANALYSIS")
    print("=" * 70)

    # Load dataset

    print("\nLoading dataset...")

    X = load_features()

    print(
        f"Dataset shape: {X.shape}"
    )

    # Load runs

    print(
        "\nLoading IDEC stability runs..."
    )

    runs = load_all_runs()

    validate_runs(
        runs,
        X,
    )

    print(
        f"\nLoaded {len(runs)} runs successfully."
    )

    # 1. Training behavior

    print("\n" + "=" * 70)
    print("1. TRAINING BEHAVIOR")
    print("=" * 70)

    training_df = (
        analyze_training_behavior(
            runs
        )
    )

    training_df.to_csv(
        AGGREGATED_DIR
        / "training_metrics.csv",
        index=False,
    )

    plot_training_kl(
        training_df
    )

    plot_reconstruction_loss(
        training_df
    )

    print(
        "\nFinal training values:"
    )

    for seed in SEEDS:

        subset = training_df[
            training_df["seed"] == seed
        ]

        final_row = subset.iloc[-1]

        print(
            f"Seed {seed}: "
            f"KL={final_row['train_kl']:.6f}, "
            f"Recon={final_row['train_reconstruction']:.6f}"
        )

    # 2. Internal clustering quality

    print("\n" + "=" * 70)
    print("2. INTERNAL CLUSTERING QUALITY")
    print("=" * 70)

    quality_df = analyze_quality(
        runs
    )

    quality_df.to_csv(
        AGGREGATED_DIR
        / "quality_metrics.csv",
        index=False,
    )

    plot_quality_metrics(
        quality_df
    )

    print(
        "\nQuality metrics:"
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

    sizes_df = analyze_cluster_sizes(
        runs
    )

    sizes_df.to_csv(
        AGGREGATED_DIR
        / "cluster_sizes.csv",
        index=False,
    )

    plot_cluster_sizes(
        sizes_df
    )

    print(
        "\nCluster sizes:"
    )

    print(
        sizes_df.to_string(
            index=False
        )
    )

    # 4. Partition stability

    print("\n" + "=" * 70)
    print("4. PARTITION STABILITY")
    print("=" * 70)

    ari_matrix = calculate_pairwise_ari(
        runs
    )

    ari_matrix.to_csv(
        AGGREGATED_DIR
        / "pairwise_ari.csv"
    )

    ari_summary = summarize_ari(
        ari_matrix
    )

    with open(
        AGGREGATED_DIR
        / "ari_summary.json",
        "w",
    ) as f:

        json.dump(
            ari_summary,
            f,
            indent=2,
        )

    plot_ari_heatmap(
        ari_matrix
    )

    print(
        "\nPairwise ARI:"
    )

    print(
        ari_matrix.to_string()
    )

    print(
        "\nARI summary:"
    )

    for key, value in ari_summary.items():

        print(
            f"  {key}: {value}"
        )

    # 5. Cluster profile stability

    print("\n" + "=" * 70)
    print("5. CLUSTER PROFILE STABILITY")
    print("=" * 70)

    profiles_df = (
        analyze_cluster_profiles(
            X,
            runs,
        )
    )

    profiles_df.to_csv(
        AGGREGATED_DIR
        / "cluster_profiles.csv",
        index=False,
    )

    profile_stability_df = (
        analyze_profile_stability(
            profiles_df
        )
    )

    profile_stability_df.to_csv(
        AGGREGATED_DIR
        / "profile_stability.csv",
        index=False,
    )

    plot_cluster_profiles(
        profiles_df
    )

    print(
        "\nCluster profiles:"
    )

    print(
        profiles_df.to_string(
            index=False
        )
    )

    print(
        "\nProfile stability relative to "
        f"reference seed {REFERENCE_SEED}:"
    )

    print(
        profile_stability_df.to_string(
            index=False
        )
    )

    # 6. Latent representation stability

    print("\n" + "=" * 70)
    print("6. LATENT REPRESENTATION STABILITY")
    print("=" * 70)

    representation_df, displacement_by_seed = (
        analyze_representation_change(
            runs
        )
    )

    representation_df.to_csv(
        AGGREGATED_DIR
        / "representation_change.csv",
        index=False,
    )

    plot_representation_displacement(
        displacement_by_seed
    )

    print(
        "\nLatent representation displacement:"
    )

    print(
        representation_df.to_string(
            index=False
        )
    )

    # Final summary

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        "\nKey stability summary:"
    )

    print(
        f"  Mean pairwise ARI: "
        f"{ari_summary['mean_ARI']:.6f}"
    )

    print(
        f"  Std pairwise ARI: "
        f"{ari_summary['std_ARI']:.6f}"
    )

    print(
        f"  Min pairwise ARI: "
        f"{ari_summary['min_ARI']:.6f}"
    )

    print(
        f"  Max pairwise ARI: "
        f"{ari_summary['max_ARI']:.6f}"
    )

    print(
        "\nFinal reconstruction loss:"
    )

    print(
        f"  Mean: "
        f"{training_df.groupby('seed')['train_reconstruction'].last().mean():.6f}"
    )

    print(
        f"  Std: "
        f"{training_df.groupby('seed')['train_reconstruction'].last().std():.6f}"
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