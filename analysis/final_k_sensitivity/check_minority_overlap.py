from pathlib import Path
import json

import numpy as np


# Configuration

K5 = 5
K6 = 6

# The smallest cluster IDs from the profile analysis
K5_MINORITY_CLUSTER = 0
K6_MINORITY_CLUSTER = 5


# Data loading

def load_labels(project_root: Path, k: int) -> np.ndarray:
    """
    Load full-dataset DEC cluster labels for a given k.
    """

    path = (
        project_root
        / "experiments"
        / "final_k_sensitivity"
        / "runs"
        / f"k_{k}"
        / "full_labels.npy"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing labels for k={k}: {path}"
        )

    labels = np.load(path)

    if labels.ndim != 1:
        raise ValueError(
            f"Labels for k={k} must be 1-dimensional, "
            f"but got shape {labels.shape}."
        )

    print(
        f"Loaded k={k} labels: "
        f"{len(labels):,} observations"
    )

    return labels


# Validation

def validate_labels(
    labels_k5: np.ndarray,
    labels_k6: np.ndarray,
) -> None:
    """
    Verify that both label arrays refer to the same
    observations in the same row order.
    """

    if len(labels_k5) != len(labels_k6):
        raise ValueError(
            "Sample mismatch between k=5 and k=6."
        )

    if len(labels_k5) == 0:
        raise ValueError(
            "Label arrays are empty."
        )

    print(
        f"\nValidation successful: "
        f"{len(labels_k5):,} observations in both runs."
    )


# Minority-cluster overlap

def compute_overlap(
    labels_k5: np.ndarray,
    labels_k6: np.ndarray,
    k5_cluster: int,
    k6_cluster: int,
) -> dict:
    """
    Compare the observation membership of the smallest
    clusters at k=5 and k=6.
    """

    mask_k5 = labels_k5 == k5_cluster
    mask_k6 = labels_k6 == k6_cluster

    n_k5 = int(np.sum(mask_k5))
    n_k6 = int(np.sum(mask_k6))

    intersection = int(
        np.sum(mask_k5 & mask_k6)
    )

    union = int(
        np.sum(mask_k5 | mask_k6)
    )

    # Conditional overlap:
    # What percentage of the k=5 minority is also in k=6?
    k5_recovered = (
        intersection / n_k5
        if n_k5 > 0
        else 0.0
    )

    # Conditional overlap:
    # What percentage of the k=6 minority is also in k=5?
    k6_recovered = (
        intersection / n_k6
        if n_k6 > 0
        else 0.0
    )

    # Jaccard similarity:
    # intersection / union
    jaccard = (
        intersection / union
        if union > 0
        else 0.0
    )

    # Binary ARI:
    # minority vs. rest
    from sklearn.metrics import adjusted_rand_score

    binary_k5 = mask_k5.astype(int)
    binary_k6 = mask_k6.astype(int)

    binary_ari = adjusted_rand_score(
        binary_k5,
        binary_k6,
    )

    return {
        "k5_cluster": k5_cluster,
        "k6_cluster": k6_cluster,
        "k5_size": n_k5,
        "k6_size": n_k6,
        "intersection": intersection,
        "union": union,
        "k5_recovered_by_k6_percent": (
            k5_recovered * 100.0
        ),
        "k6_recovered_by_k5_percent": (
            k6_recovered * 100.0
        ),
        "jaccard_similarity": jaccard,
        "binary_minority_vs_rest_ARI": binary_ari,
    }


# Print results

def print_results(results: dict) -> None:
    """
    Print a concise interpretation-oriented summary.
    """

    print("\n" + "=" * 80)
    print("k=5 vs k=6 MINORITY-CLUSTER OVERLAP")
    print("=" * 80)

    print(
        f"\nk=5 minority cluster: "
        f"C{results['k5_cluster']}"
    )

    print(
        f"k=6 minority cluster: "
        f"C{results['k6_cluster']}"
    )

    print(
        f"\nCluster sizes:"
        f"\n  k=5: {results['k5_size']:,}"
        f"\n  k=6: {results['k6_size']:,}"
    )

    print(
        f"\nShared observations: "
        f"{results['intersection']:,}"
    )

    print(
        f"Union of observations: "
        f"{results['union']:,}"
    )

    print(
        f"\nPercentage of k=5 minority recovered "
        f"at k=6: "
        f"{results['k5_recovered_by_k6_percent']:.2f}%"
    )

    print(
        f"Percentage of k=6 minority recovered "
        f"at k=5: "
        f"{results['k6_recovered_by_k5_percent']:.2f}%"
    )

    print(
        f"\nJaccard similarity: "
        f"{results['jaccard_similarity']:.4f}"
    )

    print(
        f"Binary minority-vs-rest ARI: "
        f"{results['binary_minority_vs_rest_ARI']:.4f}"
    )

    print("\n" + "-" * 80)
    print("INTERPRETATION")
    print("-" * 80)

    recovery_5 = (
        results["k5_recovered_by_k6_percent"]
    )

    recovery_6 = (
        results["k6_recovered_by_k5_percent"]
    )

    jaccard = results["jaccard_similarity"]

    if (
        recovery_5 >= 95.0
        and recovery_6 >= 95.0
        and jaccard >= 0.90
    ):
        print(
            "The minority subgroup is highly consistent "
            "between k=5 and k=6."
        )

        print(
            "The observation-level membership is strongly "
            "preserved across the two neighboring k values."
        )

    elif (
        recovery_5 >= 80.0
        and recovery_6 >= 80.0
        and jaccard >= 0.70
    ):
        print(
            "The minority subgroup shows strong but "
            "not complete membership consistency "
            "between k=5 and k=6."
        )

    else:
        print(
            "The minority subgroup shows limited "
            "observation-level overlap between k=5 "
            "and k=6."
        )

    print(
        "\nNote: This comparison tests observation-level "
        "membership, complementing the earlier profile-level "
        "similarity analysis."
    )


# Save results

def save_results(
    results: dict,
    output_path: Path,
) -> None:
    """
    Save overlap results as JSON.
    """

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        f"\nSaved results to: {output_path}"
    )


# Main

def main():

    project_root = (
        Path(__file__).resolve().parents[2]
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

    aggregated_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "FINAL DEC k-SENSITIVITY — "
        "MINORITY OVERLAP CHECK"
    )
    print("=" * 80)

    # Load labels

    labels_k5 = load_labels(
        project_root,
        K5,
    )

    labels_k6 = load_labels(
        project_root,
        K6,
    )

    # Validate

    validate_labels(
        labels_k5,
        labels_k6,
    )

    # Compute overlap

    results = compute_overlap(
        labels_k5=labels_k5,
        labels_k6=labels_k6,
        k5_cluster=K5_MINORITY_CLUSTER,
        k6_cluster=K6_MINORITY_CLUSTER,
    )

    # Print

    print_results(results)

    # Save

    output_path = (
        aggregated_dir
        / "minority_cluster_overlap_k5_k6.json"
    )

    save_results(
        results,
        output_path,
    )

    print("\n" + "=" * 80)
    print("MINORITY OVERLAP CHECK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()