from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


# Configuration

CONDITIONS = ["frozen", "trainable"]

FEATURE_NAMES = [
    "Lifestyle",
    "Communication",
    "Responsibility",
    "Openness",
    "Interests",
    "EmotionalStability",
    "SocialEngagement",
]

SMALL_CLUSTER_THRESHOLD = 1.0  # percentage


# Data loading

def load_features(project_root: Path) -> np.ndarray:
    """
    Load the 7-dimensional standardized feature matrix.
    """

    feature_path = (
        project_root
        / "embeddings"
        / "X_scaled.npy"
    )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing X_scaled.npy: {feature_path}"
        )

    X = np.load(feature_path)

    if X.ndim != 2:
        raise ValueError(
            f"X_scaled must be 2-dimensional, "
            f"but got shape {X.shape}."
        )

    if X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features, "
            f"but X_scaled contains {X.shape[1]}."
        )

    print(
        f"Loaded X_scaled: "
        f"{X.shape[0]:,} samples × {X.shape[1]} features"
    )

    return X


def load_labels(
    runs_root: Path,
    conditions: list[str],
) -> dict[str, np.ndarray]:
    """
    Load full-dataset cluster labels for each condition.
    """

    labels_by_condition = {}

    for condition in conditions:

        label_path = (
            runs_root
            / condition
            / "full_labels.npy"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing labels for condition "
                f"'{condition}': {label_path}"
            )

        labels = np.load(label_path)

        if labels.ndim != 1:
            raise ValueError(
                f"Labels for {condition} must be "
                f"1-dimensional, but got shape {labels.shape}."
            )

        labels_by_condition[condition] = labels

        print(
            f"Loaded {condition}: "
            f"{len(labels):,} labels, "
            f"{len(np.unique(labels))} clusters"
        )

    return labels_by_condition


# Validation

def validate_data(
    X: np.ndarray,
    labels_by_condition: dict[str, np.ndarray],
) -> None:
    """
    Verify that X_scaled and all label arrays contain
    the same number of observations and exactly 5 clusters.
    """

    n_samples = X.shape[0]

    for condition, labels in labels_by_condition.items():

        if len(labels) != n_samples:
            raise ValueError(
                f"Sample mismatch for {condition}: "
                f"X_scaled has {n_samples:,}, "
                f"but labels have {len(labels):,}."
            )

        unique_labels = np.unique(labels)

        if len(unique_labels) != 5:
            raise ValueError(
                f"{condition} expected 5 clusters, "
                f"but found {len(unique_labels)}: "
                f"{unique_labels}"
            )

    print(
        f"\nValidation successful: "
        f"all conditions contain {n_samples:,} samples "
        f"and 5 clusters."
    )


# Cluster sizes

def compute_cluster_sizes(
    labels_by_condition: dict[str, np.ndarray],
) -> dict:
    """
    Compute cluster counts and percentages.
    """

    results = {}

    for condition, labels in labels_by_condition.items():

        total_samples = len(labels)

        unique_labels, counts = np.unique(
            labels,
            return_counts=True,
        )

        condition_results = {}

        for cluster_id, count in zip(
            unique_labels,
            counts,
        ):

            percentage = (
                count / total_samples
            ) * 100.0

            condition_results[int(cluster_id)] = {
                "count": int(count),
                "percentage": float(percentage),
            }

        results[condition] = condition_results

    return results


def compute_size_summary(
    cluster_sizes: dict,
) -> dict:
    """
    Compute summary statistics for cluster-size distributions.
    """

    summary = {}

    for condition, clusters in cluster_sizes.items():

        counts = np.array(
            [
                result["count"]
                for result in clusters.values()
            ],
            dtype=np.float64,
        )

        percentages = np.array(
            [
                result["percentage"]
                for result in clusters.values()
            ],
            dtype=np.float64,
        )

        summary[condition] = {
            "min_cluster_count": int(np.min(counts)),
            "max_cluster_count": int(np.max(counts)),
            "min_cluster_percentage": float(
                np.min(percentages)
            ),
            "max_cluster_percentage": float(
                np.max(percentages)
            ),
            "max_min_size_ratio": float(
                np.max(counts) / np.min(counts)
            ),
            "num_small_clusters": int(
                np.sum(
                    percentages < SMALL_CLUSTER_THRESHOLD
                )
            ),
        }

    return summary


def print_cluster_sizes(
    cluster_sizes: dict,
    summary: dict,
) -> None:
    """
    Print cluster-size distributions and summaries.
    """

    print("\n" + "=" * 80)
    print("CLUSTER SIZE DISTRIBUTIONS")
    print("=" * 80)

    for condition in CONDITIONS:

        print(f"\n{condition.upper()}")
        print("-" * 50)

        for cluster_id in sorted(
            cluster_sizes[condition]
        ):

            result = cluster_sizes[
                condition
            ][cluster_id]

            print(
                f"  Cluster {cluster_id}: "
                f"{result['count']:>7,} "
                f"({result['percentage']:>6.2f}%)"
            )

    print("\n" + "=" * 80)
    print("CLUSTER SIZE SUMMARY")
    print("=" * 80)

    print(
        f"{'Condition':>12} | "
        f"{'Min':>10} | "
        f"{'Max':>10} | "
        f"{'Min %':>8} | "
        f"{'Max %':>8} | "
        f"{'Max/Min':>9} | "
        f"{'Small':>7}"
    )

    print("-" * 80)

    for condition in CONDITIONS:

        stats = summary[condition]

        print(
            f"{condition:>12} | "
            f"{stats['min_cluster_count']:>10,} | "
            f"{stats['max_cluster_count']:>10,} | "
            f"{stats['min_cluster_percentage']:>7.2f}% | "
            f"{stats['max_cluster_percentage']:>7.2f}% | "
            f"{stats['max_min_size_ratio']:>9.2f} | "
            f"{stats['num_small_clusters']:>7}"
        )


# Cluster profiles

def compute_cluster_profiles(
    X: np.ndarray,
    labels_by_condition: dict[str, np.ndarray],
) -> dict:
    """
    Compute mean standardized feature values for every cluster.

    X_scaled is used because these seven features are the
    interpretable behavioral dimensions of the dataset.
    """

    profiles = {}

    for condition, labels in labels_by_condition.items():

        condition_profiles = {}

        for cluster_id in sorted(
            np.unique(labels)
        ):

            mask = labels == cluster_id
            cluster_X = X[mask]

            mean_profile = np.mean(
                cluster_X,
                axis=0,
            )

            condition_profiles[int(cluster_id)] = {
                "count": int(np.sum(mask)),
                "percentage": float(
                    np.mean(mask) * 100.0
                ),
                "profile": mean_profile.tolist(),
            }

        profiles[condition] = condition_profiles

    return profiles


def print_profiles(
    profiles: dict,
) -> None:
    """
    Print all cluster profiles.
    """

    print("\n" + "=" * 80)
    print("SEVEN-FEATURE CLUSTER PROFILES")
    print("=" * 80)

    for condition in CONDITIONS:

        print(f"\n{condition.upper()}")
        print("-" * 80)

        for cluster_id in sorted(
            profiles[condition]
        ):

            result = profiles[
                condition
            ][cluster_id]

            print(
                f"\nCluster {cluster_id} "
                f"(N={result['count']:,}, "
                f"{result['percentage']:.2f}%):"
            )

            for feature_name, value in zip(
                FEATURE_NAMES,
                result["profile"],
            ):

                print(
                    f"  {feature_name:<22} "
                    f"{value:>8.3f}"
                )


# Profile matching

def compute_profile_distance_matrix(
    reference_profiles: dict,
    target_profiles: dict,
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Compute Euclidean distances between cluster profiles.
    """

    reference_clusters = sorted(
        reference_profiles.keys()
    )

    target_clusters = sorted(
        target_profiles.keys()
    )

    reference_matrix = np.array(
        [
            reference_profiles[c]["profile"]
            for c in reference_clusters
        ]
    )

    target_matrix = np.array(
        [
            target_profiles[c]["profile"]
            for c in target_clusters
        ]
    )

    distance_matrix = np.zeros(
        (
            len(reference_clusters),
            len(target_clusters),
        ),
        dtype=np.float64,
    )

    for i, reference_profile in enumerate(
        reference_matrix
    ):

        for j, target_profile in enumerate(
            target_matrix
        ):

            distance_matrix[i, j] = np.linalg.norm(
                reference_profile - target_profile
            )

    return (
        distance_matrix,
        reference_clusters,
        target_clusters,
    )


def match_profiles(
    reference_profiles: dict,
    target_profiles: dict,
) -> dict:
    """
    Match frozen and trainable clusters using minimum
    Euclidean distance between their seven-feature profiles.

    This is descriptive only; it does not establish
    observation-level correspondence.
    """

    (
        distance_matrix,
        reference_clusters,
        target_clusters,
    ) = compute_profile_distance_matrix(
        reference_profiles,
        target_profiles,
    )

    row_indices, col_indices = linear_sum_assignment(
        distance_matrix
    )

    matches = {}

    for row, col in zip(
        row_indices,
        col_indices,
    ):

        reference_cluster = reference_clusters[row]
        target_cluster = target_clusters[col]

        matches[reference_cluster] = {
            "target_cluster": int(target_cluster),
            "profile_distance": float(
                distance_matrix[row, col]
            ),
            "reference_count": int(
                reference_profiles[
                    reference_cluster
                ]["count"]
            ),
            "target_count": int(
                target_profiles[
                    target_cluster
                ]["count"]
            ),
        }

    return matches


def compute_profile_differences(
    reference_profiles: dict,
    target_profiles: dict,
    matches: dict,
) -> dict:
    """
    Compute feature-level differences for matched profiles.

    Difference = trainable profile - frozen profile.
    """

    results = {}

    for reference_cluster, match in matches.items():

        target_cluster = match["target_cluster"]

        reference_profile = np.array(
            reference_profiles[
                reference_cluster
            ]["profile"]
        )

        target_profile = np.array(
            target_profiles[
                target_cluster
            ]["profile"]
        )

        differences = (
            target_profile - reference_profile
        )

        results[int(reference_cluster)] = {
            "target_cluster": int(target_cluster),
            "profile_distance": float(
                match["profile_distance"]
            ),
            "reference_profile": (
                reference_profile.tolist()
            ),
            "target_profile": (
                target_profile.tolist()
            ),
            "feature_differences": {
                feature_name: float(difference)
                for feature_name, difference in zip(
                    FEATURE_NAMES,
                    differences,
                )
            },
        }

    return results


def print_profile_matches(
    matches: dict,
    differences: dict,
) -> None:
    """
    Print frozen → trainable profile correspondence.
    """

    print("\n" + "=" * 80)
    print("PROFILE CORRESPONDENCE: FROZEN → TRAINABLE")
    print("=" * 80)

    print(
        f"{'Frozen':>10} | "
        f"{'Trainable':>12} | "
        f"{'Distance':>10} | "
        f"{'Frozen N':>10} | "
        f"{'Trainable N':>12}"
    )

    print("-" * 75)

    for reference_cluster in sorted(matches):

        match = matches[
            reference_cluster
        ]

        print(
            f"{reference_cluster:>10} | "
            f"{match['target_cluster']:>12} | "
            f"{match['profile_distance']:>10.4f} | "
            f"{match['reference_count']:>10,} | "
            f"{match['target_count']:>12,}"
        )

    print("\nFeature-level profile changes")
    print("-" * 80)

    for reference_cluster in sorted(differences):

        result = differences[
            reference_cluster
        ]

        print(
            f"\nFrozen C{reference_cluster} "
            f"→ Trainable C{result['target_cluster']} "
            f"(distance={result['profile_distance']:.4f})"
        )

        for feature_name, difference in (
            result["feature_differences"].items()
        ):

            print(
                f"  {feature_name:<22} "
                f"{difference:+.4f}"
            )


# Plot cluster sizes

def plot_cluster_sizes(
    cluster_sizes: dict,
    output_path: Path,
) -> None:
    """
    Compare cluster sizes between frozen and trainable
    conditions.

    Cluster IDs are displayed separately because cluster
    labels are arbitrary.
    """

    x = np.arange(5)
    width = 0.35

    frozen_counts = [
        cluster_sizes["frozen"][c]["count"]
        for c in range(5)
    ]

    trainable_counts = [
        cluster_sizes["trainable"][c]["count"]
        for c in range(5)
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width / 2,
        frozen_counts,
        width,
        label="Frozen",
    )

    plt.bar(
        x + width / 2,
        trainable_counts,
        width,
        label="Trainable",
    )

    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Samples")
    plt.title(
        "Cluster Size Comparison: Frozen vs Trainable Encoder"
    )

    plt.xticks(
        x,
        [f"C{i}" for i in range(5)],
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved cluster-size figure to: {output_path}"
    )


# Plot cluster profiles

def plot_profiles(
    profiles: dict,
    output_path: Path,
) -> None:
    """
    Plot seven-feature cluster profiles for both conditions.

    Cluster IDs are displayed separately because they are
    arbitrary across conditions.
    """

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 9),
    )

    x = np.arange(
        len(FEATURE_NAMES)
    )

    for ax, condition in zip(
        axes,
        CONDITIONS,
    ):

        for cluster_id in sorted(
            profiles[condition]
        ):

            profile = profiles[
                condition
            ][cluster_id]["profile"]

            ax.plot(
                x,
                profile,
                marker="o",
                label=f"Cluster {cluster_id}",
            )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1,
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            FEATURE_NAMES,
            rotation=30,
            ha="right",
        )

        ax.set_ylabel(
            "Standardized cluster mean"
        )

        ax.set_title(
            f"{condition.capitalize()} Encoder"
        )

        ax.legend()

    fig.supxlabel("Feature")

    fig.suptitle(
        "Seven-Feature Cluster Profiles",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved profile figure to: {output_path}"
    )


# Main

def main():

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    runs_root = (
        project_root
        / "experiments"
        / "encoder_ablation"
        / "runs"
    )

    analysis_root = (
        project_root
        / "analysis"
        / "encoder_ablation"
    )

    aggregated_dir = (
        analysis_root
        / "aggregated"
    )

    figures_dir = (
        analysis_root
        / "figures"
    )

    aggregated_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "DEC ENCODER TRAINABILITY ABLATION — "
        "CLUSTER STRUCTURE ANALYSIS"
    )
    print("=" * 80)

    # Load

    X = load_features(
        project_root
    )

    labels_by_condition = load_labels(
        runs_root=runs_root,
        conditions=CONDITIONS,
    )

    # Validate

    validate_data(
        X=X,
        labels_by_condition=labels_by_condition,
    )

    # Cluster sizes

    cluster_sizes = compute_cluster_sizes(
        labels_by_condition
    )

    size_summary = compute_size_summary(
        cluster_sizes
    )

    print_cluster_sizes(
        cluster_sizes=cluster_sizes,
        summary=size_summary,
    )

    # Cluster profiles

    profiles = compute_cluster_profiles(
        X=X,
        labels_by_condition=labels_by_condition,
    )

    print_profiles(
        profiles=profiles,
    )

    # Frozen → Trainable profile matching

    matches = match_profiles(
        reference_profiles=profiles["frozen"],
        target_profiles=profiles["trainable"],
    )

    profile_differences = (
        compute_profile_differences(
            reference_profiles=profiles["frozen"],
            target_profiles=profiles["trainable"],
            matches=matches,
        )
    )

    print_profile_matches(
        matches=matches,
        differences=profile_differences,
    )

    # Save results

    results = {
        "experiment": "encoder_ablation",
        "conditions": CONDITIONS,
        "feature_names": FEATURE_NAMES,

        "cluster_size_summary": size_summary,

        "cluster_sizes": {
            condition: {
                str(cluster_id): values
                for cluster_id, values
                in condition_results.items()
            }
            for condition, condition_results
            in cluster_sizes.items()
        },

        "profiles": {
            condition: {
                str(cluster_id): values
                for cluster_id, values
                in condition_profiles.items()
            }
            for condition, condition_profiles
            in profiles.items()
        },

        "profile_correspondence": {
            str(reference_cluster): values
            for reference_cluster, values
            in profile_differences.items()
        },
    }

    json_path = (
        aggregated_dir
        / "cluster_structure_comparison.json"
    )

    with open(
        json_path,
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        f"\nSaved JSON results to: {json_path}"
    )

    # Save profile CSV

    csv_path = (
        aggregated_dir
        / "cluster_profiles_comparison.csv"
    )

    fieldnames = [
        "condition",
        "cluster",
        "count",
        "percentage",
    ] + FEATURE_NAMES

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for condition in CONDITIONS:

            for cluster_id, result in (
                profiles[condition].items()
            ):

                row = {
                    "condition": condition,
                    "cluster": cluster_id,
                    "count": result["count"],
                    "percentage": result["percentage"],
                }

                for feature_name, value in zip(
                    FEATURE_NAMES,
                    result["profile"],
                ):

                    row[feature_name] = value

                writer.writerow(row)

    print(
        f"Saved profile CSV to: {csv_path}"
    )

    # Figures

    plot_cluster_sizes(
        cluster_sizes=cluster_sizes,
        output_path=(
            figures_dir
            / "cluster_sizes_comparison.png"
        ),
    )

    plot_profiles(
        profiles=profiles,
        output_path=(
            figures_dir
            / "cluster_profiles_comparison.png"
        ),
    )

    # Complete

    print("\n" + "=" * 80)
    print(
        "CLUSTER STRUCTURE ANALYSIS COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()