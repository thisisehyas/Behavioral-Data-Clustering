from pathlib import Path

import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


# Configuration

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


# Data loading

def load_features(project_root: Path) -> np.ndarray:
    """
    Load the 7-dimensional standardized feature matrix.
    """

    feature_path = project_root / "embeddings" / "X_scaled.npy"

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
        f"{X.shape[0]} samples × {X.shape[1]} features"
    )

    return X


def load_labels(
    runs_root: Path,
    seeds: list[int],
) -> dict[int, np.ndarray]:
    """
    Load full-dataset cluster labels for every seed.
    """

    labels_by_seed = {}

    for seed in seeds:
        label_path = (
            runs_root
            / f"seed_{seed}"
            / "full_labels.npy"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing labels for seed {seed}: "
                f"{label_path}"
            )

        labels = np.load(label_path)

        if labels.ndim != 1:
            raise ValueError(
                f"Labels for seed {seed} must be 1-dimensional, "
                f"but got shape {labels.shape}."
            )

        labels_by_seed[seed] = labels

        print(
            f"Loaded seed={seed}: "
            f"{len(labels)} labels, "
            f"{len(np.unique(labels))} clusters"
        )

    return labels_by_seed


def validate_data(
    X: np.ndarray,
    labels_by_seed: dict[int, np.ndarray],
) -> None:
    """
    Verify that X_scaled and all label arrays contain the
    same number of samples.
    """

    n_samples = X.shape[0]

    for seed, labels in labels_by_seed.items():
        if len(labels) != n_samples:
            raise ValueError(
                f"Sample mismatch for seed {seed}: "
                f"X_scaled has {n_samples}, "
                f"but labels have {len(labels)}."
            )

    print(
        f"\nValidation successful: "
        f"all runs contain {n_samples} samples."
    )


# Cluster profile computation

def compute_cluster_profiles(
    X: np.ndarray,
    labels_by_seed: dict[int, np.ndarray],
    seeds: list[int],
) -> dict[int, dict[int, dict]]:
    """
    Compute the mean 7-feature profile for every cluster
    in every seed.

    Because X_scaled is standardized, the cluster means can
    be interpreted as standardized deviations from the
    overall feature mean.
    """

    profiles = {}

    for seed in seeds:
        labels = labels_by_seed[seed]

        unique_labels = sorted(np.unique(labels))

        seed_profiles = {}

        for cluster_id in unique_labels:
            mask = labels == cluster_id

            cluster_X = X[mask]

            mean_profile = np.mean(
                cluster_X,
                axis=0,
            )

            seed_profiles[int(cluster_id)] = {
                "count": int(np.sum(mask)),
                "percentage": float(
                    np.mean(mask) * 100.0
                ),
                "profile": mean_profile.tolist(),
            }

        profiles[seed] = seed_profiles

    return profiles


# Profile matching

def compute_distance_matrix(
    reference_profiles: dict[int, dict],
    target_profiles: dict[int, dict],
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Compute Euclidean distances between all reference and
    target cluster profiles.
    """

    reference_clusters = sorted(reference_profiles.keys())
    target_clusters = sorted(target_profiles.keys())

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

    for i, reference_profile in enumerate(reference_matrix):
        for j, target_profile in enumerate(target_matrix):
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
    Find the optimal one-to-one correspondence between
    reference and target clusters using the Hungarian algorithm.
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
            "reference_percentage": float(
                reference_profiles[
                    reference_cluster
                ]["percentage"]
            ),
            "target_percentage": float(
                target_profiles[
                    target_cluster
                ]["percentage"]
            ),
        }

    return matches


# Profile similarity

def compute_profile_differences(
    reference_profiles: dict[int, dict],
    target_profiles: dict[int, dict],
    matches: dict,
) -> dict:
    """
    Compute feature-level differences between matched
    cluster profiles.
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
            "reference_profile": reference_profile.tolist(),
            "target_profile": target_profile.tolist(),
        }

    return results


# Printing results

def print_profiles(
    profiles: dict[int, dict[int, dict]],
    seeds: list[int],
) -> None:
    """
    Print all cluster profiles.
    """

    print("\n" + "=" * 80)
    print("CLUSTER PROFILES")
    print("=" * 80)

    for seed in seeds:
        print(f"\nSeed {seed}")
        print("-" * 80)

        for cluster_id in sorted(profiles[seed].keys()):
            result = profiles[seed][cluster_id]

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
                    f"  {feature_name:<20} "
                    f"{value:>7.3f}"
                )


def print_matches(
    reference_seed: int,
    target_seed: int,
    matches: dict,
) -> None:
    """
    Print cluster correspondence between the reference
    seed and a target seed.
    """

    print("\n" + "=" * 80)
    print(
        f"CLUSTER CORRESPONDENCE: "
        f"SEED {reference_seed} → SEED {target_seed}"
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

    for reference_cluster in sorted(matches.keys()):
        match = matches[reference_cluster]

        print(
            f"{reference_cluster:>12} | "
            f"{match['target_cluster']:>8} | "
            f"{match['profile_distance']:>10.4f} | "
            f"{match['reference_count']:>10,} | "
            f"{match['target_count']:>10,}"
        )


def print_profile_differences(
    reference_seed: int,
    target_seed: int,
    differences: dict,
) -> None:
    """
    Print feature-level differences for matched clusters.
    """

    print("\n" + "=" * 80)
    print(
        f"PROFILE DIFFERENCES: "
        f"SEED {reference_seed} → SEED {target_seed}"
    )
    print("=" * 80)

    for reference_cluster in sorted(
        differences.keys()
    ):
        result = differences[reference_cluster]

        print(
            f"\nReference C{reference_cluster} "
            f"→ Seed {target_seed} "
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
                f"  {feature_name:<20} "
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


def save_correspondence_csv(
    correspondence: dict,
    output_path: Path,
) -> None:
    """
    Save cluster correspondence results as CSV.
    """

    fieldnames = [
        "target_seed",
        "reference_cluster",
        "target_cluster",
        "profile_distance",
        "reference_count",
        "target_count",
        "reference_percentage",
        "target_percentage",
    ]

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

        for target_seed, seed_matches in (
            correspondence.items()
        ):
            for reference_cluster, match in (
                seed_matches.items()
            ):
                writer.writerow(
                    {
                        "target_seed": target_seed,
                        "reference_cluster": (
                            reference_cluster
                        ),
                        "target_cluster": (
                            match["target_cluster"]
                        ),
                        "profile_distance": (
                            match["profile_distance"]
                        ),
                        "reference_count": (
                            match["reference_count"]
                        ),
                        "target_count": (
                            match["target_count"]
                        ),
                        "reference_percentage": (
                            match["reference_percentage"]
                        ),
                        "target_percentage": (
                            match["target_percentage"]
                        ),
                    }
                )

    print(
        f"Saved correspondence CSV to: {output_path}"
    )


# Plot profile comparison

def plot_reference_profiles(
    profiles: dict[int, dict[int, dict]],
    reference_seed: int,
    output_path: Path,
) -> None:
    """
    Plot the seven-dimensional cluster profiles for the
    reference seed.

    This provides a visual reference for the profiles that
    will be matched across other seeds.
    """

    reference_profiles = profiles[
        reference_seed
    ]

    cluster_ids = sorted(
        reference_profiles.keys()
    )

    profile_matrix = np.array(
        [
            reference_profiles[c]["profile"]
            for c in cluster_ids
        ]
    )

    plt.figure(figsize=(11, 6))

    x = np.arange(len(FEATURE_NAMES))

    for i, cluster_id in enumerate(cluster_ids):
        plt.plot(
            x,
            profile_matrix[i],
            marker="o",
            label=f"Cluster {cluster_id}",
        )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.xticks(
        x,
        FEATURE_NAMES,
        rotation=30,
        ha="right",
    )

    plt.ylabel("Standardized cluster mean")
    plt.xlabel("Feature")
    plt.title(
        f"Reference Cluster Profiles "
        f"(Seed {reference_seed})"
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
        f"Saved reference profile figure to: "
        f"{output_path}"
    )


# Main

def main():
    project_root = Path(__file__).resolve().parents[2]

    runs_root = (
        project_root
        / "experiments"
        / "final_stability"
        / "runs"
    )

    analysis_root = (
        project_root
        / "analysis"
        / "final_stability"
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
    print("FINAL DEC STABILITY — CLUSTER PROFILE CORRESPONDENCE")
    print("=" * 80)

    print(
        f"\nReference seed: {REFERENCE_SEED}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X = load_features(
        project_root
    )

    labels_by_seed = load_labels(
        runs_root=runs_root,
        seeds=SEEDS,
    )

    validate_data(
        X=X,
        labels_by_seed=labels_by_seed,
    )

    # --------------------------------------------------------
    # Compute profiles
    # --------------------------------------------------------

    print("\nComputing cluster profiles...")

    profiles = compute_cluster_profiles(
        X=X,
        labels_by_seed=labels_by_seed,
        seeds=SEEDS,
    )

    print_profiles(
        profiles=profiles,
        seeds=SEEDS,
    )

    # --------------------------------------------------------
    # Match every seed to seed 42
    # --------------------------------------------------------

    reference_profiles = profiles[
        REFERENCE_SEED
    ]

    correspondence = {}
    profile_differences = {}

    for seed in SEEDS:
        if seed == REFERENCE_SEED:
            continue

        matches = match_clusters(
            reference_profiles=reference_profiles,
            target_profiles=profiles[seed],
        )

        correspondence[seed] = matches

        differences = compute_profile_differences(
            reference_profiles=reference_profiles,
            target_profiles=profiles[seed],
            matches=matches,
        )

        profile_differences[seed] = differences

        print_matches(
            reference_seed=REFERENCE_SEED,
            target_seed=seed,
            matches=matches,
        )

        print_profile_differences(
            reference_seed=REFERENCE_SEED,
            target_seed=seed,
            differences=differences,
        )

    # --------------------------------------------------------
    # Save complete results
    # --------------------------------------------------------

    results = {
        "experiment": "final_stability",
        "reference_seed": REFERENCE_SEED,
        "seeds": SEEDS,
        "feature_names": FEATURE_NAMES,
        "profiles": {
            str(seed): {
                str(cluster_id): values
                for cluster_id, values
                in seed_profiles.items()
            }
            for seed, seed_profiles
            in profiles.items()
        },
        "correspondence": {
            str(seed): {
                str(reference_cluster): values
                for reference_cluster, values
                in seed_matches.items()
            }
            for seed, seed_matches
            in correspondence.items()
        },
        "profile_differences": {
            str(seed): {
                str(reference_cluster): values
                for reference_cluster, values
                in seed_differences.items()
            }
            for seed, seed_differences
            in profile_differences.items()
        },
    }

    save_json(
        data=results,
        output_path=(
            aggregated_dir
            / "cluster_profile_correspondence.json"
        ),
    )

    save_correspondence_csv(
        correspondence=correspondence,
        output_path=(
            aggregated_dir
            / "cluster_profile_correspondence.csv"
        ),
    )

    # --------------------------------------------------------
    # Plot reference profiles
    # --------------------------------------------------------

    plot_reference_profiles(
        profiles=profiles,
        reference_seed=REFERENCE_SEED,
        output_path=(
            figures_dir
            / "reference_cluster_profiles.png"
        ),
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("CLUSTER PROFILE CORRESPONDENCE ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
