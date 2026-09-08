from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


# Configuration

K_VALUES = [2, 3, 4, 5, 6]

FEATURE_NAMES = [
    "Lifestyle",
    "Communication",
    "Responsibility",
    "Openness",
    "Interests",
    "EmotionalStability",
    "SocialEngagement",
]


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
    k_values: list[int],
) -> dict[int, np.ndarray]:
    """
    Load full-dataset cluster labels for every k.
    """

    labels_by_k = {}

    for k in k_values:

        label_path = (
            runs_root
            / f"k_{k}"
            / "full_labels.npy"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing labels for k={k}: "
                f"{label_path}"
            )

        labels = np.load(label_path)

        if labels.ndim != 1:
            raise ValueError(
                f"Labels for k={k} must be 1-dimensional, "
                f"but got shape {labels.shape}."
            )

        unique_labels = np.unique(labels)

        if len(unique_labels) != k:
            raise ValueError(
                f"Expected {k} clusters for k={k}, "
                f"but found {len(unique_labels)}: "
                f"{unique_labels}"
            )

        labels_by_k[k] = labels

        print(
            f"Loaded k={k}: "
            f"{len(labels):,} labels, "
            f"{len(unique_labels)} clusters"
        )

    return labels_by_k


# Validation

def validate_data(
    X: np.ndarray,
    labels_by_k: dict[int, np.ndarray],
) -> None:
    """
    Verify that X_scaled and all label arrays contain
    the same number of samples.
    """

    n_samples = X.shape[0]

    for k, labels in labels_by_k.items():

        if len(labels) != n_samples:
            raise ValueError(
                f"Sample mismatch for k={k}: "
                f"X_scaled has {n_samples:,}, "
                f"but labels have {len(labels):,}."
            )

    print(
        f"\nValidation successful: "
        f"all runs contain {n_samples:,} samples."
    )


# Cluster profile computation

def compute_cluster_profiles(
    X: np.ndarray,
    labels_by_k: dict[int, np.ndarray],
    k_values: list[int],
) -> dict[int, dict[int, dict]]:
    """
    Compute the mean 7-feature profile for every cluster
    at every k.

    X_scaled is standardized, so cluster means indicate
    deviations from the overall feature mean.
    """

    profiles = {}

    for k in k_values:

        labels = labels_by_k[k]
        unique_labels = sorted(np.unique(labels))

        k_profiles = {}

        for cluster_id in unique_labels:

            mask = labels == cluster_id
            cluster_X = X[mask]

            mean_profile = np.mean(
                cluster_X,
                axis=0,
            )

            k_profiles[int(cluster_id)] = {
                "count": int(np.sum(mask)),
                "percentage": float(
                    np.mean(mask) * 100.0
                ),
                "profile": mean_profile.tolist(),
            }

        profiles[k] = k_profiles

    return profiles


# Printing profiles

def print_profiles(
    profiles: dict[int, dict[int, dict]],
    k_values: list[int],
) -> None:
    """
    Print all cluster profiles.
    """

    print("\n" + "=" * 80)
    print("CLUSTER PROFILES ACROSS k")
    print("=" * 80)

    for k in k_values:

        print(f"\nk = {k}")
        print("-" * 80)

        for cluster_id in sorted(profiles[k].keys()):

            result = profiles[k][cluster_id]

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

def compute_distance_matrix(
    reference_profiles: dict[int, dict],
    target_profiles: dict[int, dict],
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Compute Euclidean distances between cluster profiles.

    This is used only for descriptive comparison between
    two different k values.
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


def match_clusters(
    reference_profiles: dict[int, dict],
    target_profiles: dict[int, dict],
) -> dict:
    """
    Find the minimum-distance one-to-one correspondence
    between clusters using the Hungarian algorithm.

    Note:
    This comparison is descriptive because different k
    values may contain different numbers of clusters.
    """

    (
        distance_matrix,
        reference_clusters,
        target_clusters,
    ) = compute_distance_matrix(
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


# Profile differences

def compute_profile_differences(
    reference_profiles: dict[int, dict],
    target_profiles: dict[int, dict],
    matches: dict,
) -> dict:
    """
    Compute feature-level differences between matched
    profiles.

    Difference = target profile - reference profile.
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

        results[reference_cluster] = {
            "target_cluster": target_cluster,
            "profile_distance": match[
                "profile_distance"
            ],
            "feature_differences": {
                feature_name: float(difference)
                for feature_name, difference in zip(
                    FEATURE_NAMES,
                    differences,
                )
            },
            "reference_profile": (
                reference_profile.tolist()
            ),
            "target_profile": (
                target_profile.tolist()
            ),
        }

    return results


# Printing correspondence

def print_matches(
    reference_k: int,
    target_k: int,
    matches: dict,
) -> None:
    """
    Print profile correspondence between two k values.
    """

    print("\n" + "=" * 80)
    print(
        f"PROFILE CORRESPONDENCE: "
        f"k={reference_k} → k={target_k}"
    )
    print("=" * 80)

    print(
        f"{'Reference':>12} | "
        f"{'Target':>8} | "
        f"{'Distance':>10} | "
        f"{'Ref N':>10} | "
        f"{'Target N':>10}"
    )

    print("-" * 65)

    for reference_cluster in sorted(
        matches.keys()
    ):

        match = matches[reference_cluster]

        print(
            f"{reference_cluster:>12} | "
            f"{match['target_cluster']:>8} | "
            f"{match['profile_distance']:>10.4f} | "
            f"{match['reference_count']:>10,} | "
            f"{match['target_count']:>10,}"
        )


def print_profile_differences(
    reference_k: int,
    target_k: int,
    differences: dict,
) -> None:
    """
    Print feature-level differences between matched
    profiles.
    """

    print("\n" + "=" * 80)
    print(
        f"PROFILE DIFFERENCES: "
        f"k={reference_k} → k={target_k}"
    )
    print("=" * 80)

    for reference_cluster in sorted(
        differences.keys()
    ):

        result = differences[reference_cluster]

        print(
            f"\nReference C{reference_cluster} "
            f"→ k={target_k} "
            f"C{result['target_cluster']}"
        )

        print(
            f"  Euclidean profile distance: "
            f"{result['profile_distance']:.4f}"
        )

        for feature_name, difference in (
            result["feature_differences"].items()
        ):

            print(
                f"  {feature_name:<22} "
                f"{difference:+.4f}"
            )


# Save results

def save_json(
    data: dict,
    output_path: Path,
) -> None:
    """
    Save results as JSON.
    """

    with open(output_path, "w") as f:
        json.dump(
            data,
            f,
            indent=2,
        )

    print(
        f"Saved JSON results to: {output_path}"
    )


def save_profile_csv(
    profiles: dict[int, dict[int, dict]],
    output_path: Path,
) -> None:
    """
    Save all cluster profiles to CSV.
    """

    fieldnames = [
        "k",
        "cluster",
        "count",
        "percentage",
    ] + FEATURE_NAMES

    with open(
        output_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for k, k_profiles in profiles.items():

            for cluster_id, result in (
                k_profiles.items()
            ):

                row = {
                    "k": k,
                    "cluster": cluster_id,
                    "count": result["count"],
                    "percentage": result[
                        "percentage"
                    ],
                }

                for feature_name, value in zip(
                    FEATURE_NAMES,
                    result["profile"],
                ):
                    row[feature_name] = value

                writer.writerow(row)

    print(
        f"Saved profile CSV to: {output_path}"
    )


# Plot profiles for each k

def plot_profiles_by_k(
    profiles: dict[int, dict[int, dict]],
    k_values: list[int],
    output_path: Path,
) -> None:
    """
    Create one figure containing a profile plot for
    each k value.

    Each subplot shows cluster means across the seven
    interpretable standardized features.
    """

    fig, axes = plt.subplots(
        nrows=len(k_values),
        ncols=1,
        figsize=(12, 4 * len(k_values)),
    )

    x = np.arange(len(FEATURE_NAMES))

    for ax, k in zip(axes, k_values):

        k_profiles = profiles[k]

        for cluster_id in sorted(
            k_profiles.keys()
        ):

            profile = k_profiles[
                cluster_id
            ]["profile"]

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
            f"DEC Cluster Profiles — k={k}"
        )

        ax.legend()

    fig.supxlabel("Feature")

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


# Plot k=5 vs k=6 profiles

def plot_k5_k6_profiles(
    profiles: dict[int, dict[int, dict]],
    output_path: Path,
) -> None:
    """
    Compare k=5 and k=6 profiles side-by-side.

    Cluster IDs are shown separately because cluster IDs
    are not inherently comparable across k.
    """

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 8),
    )

    x = np.arange(len(FEATURE_NAMES))

    for ax, k in zip(
        axes,
        [5, 6],
    ):

        for cluster_id in sorted(
            profiles[k].keys()
        ):

            profile = profiles[k][
                cluster_id
            ]["profile"]

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
            f"DEC Cluster Profiles — k={k}"
        )

        ax.legend()

    fig.supxlabel("Feature")

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved k=5 vs k=6 profile figure to: "
        f"{output_path}"
    )


# Main

def main():

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    runs_root = (
        project_root
        / "experiments"
        / "final_k_sensitivity"
        / "runs"
    )

    analysis_root = (
        project_root
        / "analysis"
        / "final_k_sensitivity"
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
        "FINAL DEC k-SENSITIVITY — "
        "CLUSTER PROFILE ANALYSIS"
    )
    print("=" * 80)

    # Load data

    X = load_features(
        project_root
    )

    labels_by_k = load_labels(
        runs_root=runs_root,
        k_values=K_VALUES,
    )

    validate_data(
        X=X,
        labels_by_k=labels_by_k,
    )

    # Compute profiles

    print(
        "\nComputing cluster profiles..."
    )

    profiles = compute_cluster_profiles(
        X=X,
        labels_by_k=labels_by_k,
        k_values=K_VALUES,
    )

    print_profiles(
        profiles=profiles,
        k_values=K_VALUES,
    )

    # Compare k=5 against k=6

    print("\n")
    print("=" * 80)
    print("COMPARING k=5 AND k=6")
    print("=" * 80)

    matches_5_to_6 = match_clusters(
        reference_profiles=profiles[5],
        target_profiles=profiles[6],
    )

    differences_5_to_6 = (
        compute_profile_differences(
            reference_profiles=profiles[5],
            target_profiles=profiles[6],
            matches=matches_5_to_6,
        )
    )

    print_matches(
        reference_k=5,
        target_k=6,
        matches=matches_5_to_6,
    )

    print_profile_differences(
        reference_k=5,
        target_k=6,
        differences=differences_5_to_6,
    )

    # Compare k=4 against k=5

    print("\n")
    print("=" * 80)
    print("COMPARING k=4 AND k=5")
    print("=" * 80)

    matches_4_to_5 = match_clusters(
        reference_profiles=profiles[4],
        target_profiles=profiles[5],
    )

    differences_4_to_5 = (
        compute_profile_differences(
            reference_profiles=profiles[4],
            target_profiles=profiles[5],
            matches=matches_4_to_5,
        )
    )

    print_matches(
        reference_k=4,
        target_k=5,
        matches=matches_4_to_5,
    )

    print_profile_differences(
        reference_k=4,
        target_k=5,
        differences=differences_4_to_5,
    )

    # Save results

    results = {
        "experiment": "final_k_sensitivity",
        "k_values": K_VALUES,
        "feature_names": FEATURE_NAMES,

        "profiles": {
            str(k): {
                str(cluster_id): values
                for cluster_id, values
                in k_profiles.items()
            }
            for k, k_profiles
            in profiles.items()
        },

        "comparisons": {
            "k_4_to_k_5": {
                str(reference_cluster): values
                for reference_cluster, values
                in differences_4_to_5.items()
            },

            "k_5_to_k_6": {
                str(reference_cluster): values
                for reference_cluster, values
                in differences_5_to_6.items()
            },
        },
    }

    save_json(
        data=results,
        output_path=(
            aggregated_dir
            / "cluster_profiles_by_k.json"
        ),
    )

    save_profile_csv(
        profiles=profiles,
        output_path=(
            aggregated_dir
            / "cluster_profiles_by_k.csv"
        ),
    )

    # Plot all k values

    plot_profiles_by_k(
        profiles=profiles,
        k_values=K_VALUES,
        output_path=(
            figures_dir
            / "cluster_profiles_by_k.png"
        ),
    )

    # Plot k=5 vs k=6

    plot_k5_k6_profiles(
        profiles=profiles,
        output_path=(
            figures_dir
            / "cluster_profiles_k5_vs_k6.png"
        ),
    )

    # Complete

    print("\n" + "=" * 80)
    print(
        "CLUSTER PROFILE ANALYSIS COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
