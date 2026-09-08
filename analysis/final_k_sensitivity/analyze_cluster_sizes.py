from pathlib import Path

import csv
import json

import matplotlib.pyplot as plt
import numpy as np


# Configuration

K_VALUES = [2, 3, 4, 5, 6]

SMALL_CLUSTER_THRESHOLD = 1.0  # percentage


# Load labels

def load_labels(
    runs_root: Path,
    k_values: list[int],
) -> dict[int, np.ndarray]:
    """
    Load full-dataset cluster labels for each k.
    """

    labels_by_k = {}

    for k in k_values:
        run_dir = runs_root / f"k_{k}"
        label_path = run_dir / "full_labels.npy"

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing full_labels.npy for k={k}: {label_path}"
            )

        labels = np.load(label_path)

        if labels.ndim != 1:
            raise ValueError(
                f"Labels for k={k} must be 1-dimensional, "
                f"but got shape {labels.shape}."
            )

        labels_by_k[k] = labels

        print(
            f"Loaded k={k}: "
            f"{len(labels):,} samples, "
            f"{len(np.unique(labels))} clusters"
        )

    return labels_by_k


# Validation

def validate_labels(
    labels_by_k: dict[int, np.ndarray],
    k_values: list[int],
) -> None:
    """
    Verify that all runs contain the same number of samples
    and the expected number of clusters.
    """

    lengths = {
        k: len(labels)
        for k, labels in labels_by_k.items()
    }

    unique_lengths = set(lengths.values())

    if len(unique_lengths) != 1:
        raise ValueError(
            f"Inconsistent number of samples across runs: {lengths}"
        )

    for k in k_values:
        actual_clusters = len(
            np.unique(labels_by_k[k])
        )

        if actual_clusters != k:
            raise ValueError(
                f"k={k} expected {k} clusters, "
                f"but found {actual_clusters}."
            )

    total_samples = next(iter(unique_lengths))

    print(
        f"\nAll runs contain "
        f"{total_samples:,} samples."
    )

    print(
        "All runs contain the expected number of clusters."
    )


# Compute cluster sizes

def compute_cluster_sizes(
    labels_by_k: dict[int, np.ndarray],
    k_values: list[int],
) -> dict[int, dict[int, dict[str, float]]]:
    """
    Compute cluster counts and percentages for each k.

    Returns:
        {
            k: {
                cluster_id: {
                    "count": ...,
                    "percentage": ...
                }
            }
        }
    """

    results = {}

    for k in k_values:
        labels = labels_by_k[k]
        total_samples = len(labels)

        unique_labels, counts = np.unique(
            labels,
            return_counts=True,
        )

        k_results = {}

        for cluster_id, count in zip(
            unique_labels,
            counts,
        ):
            percentage = (
                count / total_samples
            ) * 100.0

            k_results[int(cluster_id)] = {
                "count": int(count),
                "percentage": float(percentage),
            }

        results[k] = k_results

    return results


# Compute summary statistics

def compute_summary_statistics(
    cluster_sizes: dict[int, dict[int, dict[str, float]]],
    k_values: list[int],
) -> dict:
    """
    Compute descriptive statistics for cluster sizes within each k.

    Statistics:
        - minimum cluster size
        - maximum cluster size
        - minimum percentage
        - maximum percentage
        - maximum/minimum size ratio
        - number of clusters below the small-cluster threshold
    """

    summary = {}

    for k in k_values:

        counts = np.array(
            [
                result["count"]
                for result in cluster_sizes[k].values()
            ],
            dtype=np.float64,
        )

        percentages = np.array(
            [
                result["percentage"]
                for result in cluster_sizes[k].values()
            ],
            dtype=np.float64,
        )

        min_count = int(np.min(counts))
        max_count = int(np.max(counts))

        min_percentage = float(
            np.min(percentages)
        )

        max_percentage = float(
            np.max(percentages)
        )

        max_min_ratio = (
            max_count / min_count
        )

        small_clusters = int(
            np.sum(
                percentages
                < SMALL_CLUSTER_THRESHOLD
            )
        )

        summary[str(k)] = {
            "min_cluster_count": min_count,
            "max_cluster_count": max_count,
            "min_cluster_percentage": min_percentage,
            "max_cluster_percentage": max_percentage,
            "max_min_size_ratio": float(
                max_min_ratio
            ),
            "num_small_clusters": small_clusters,
            "small_cluster_threshold_percentage": (
                SMALL_CLUSTER_THRESHOLD
            ),
        }

    return summary


# Print cluster-size distributions

def print_cluster_sizes(
    cluster_sizes: dict[int, dict[int, dict[str, float]]],
    k_values: list[int],
) -> None:
    """
    Print cluster sizes and percentages for each k.
    """

    print("\n" + "=" * 80)
    print("CLUSTER SIZE DISTRIBUTIONS ACROSS K")
    print("=" * 80)

    for k in k_values:

        print(f"\nk={k}")
        print("-" * 40)

        for cluster_id in sorted(
            cluster_sizes[k]
        ):
            result = cluster_sizes[k][cluster_id]

            print(
                f"  Cluster {cluster_id}: "
                f"{result['count']:>7,} "
                f"({result['percentage']:>6.2f}%)"
            )


# Print summary

def print_summary(
    summary: dict,
    k_values: list[int],
) -> None:
    """
    Print summary statistics across k.
    """

    print("\n" + "=" * 80)
    print("CLUSTER SIZE SUMMARY ACROSS K")
    print("=" * 80)

    header = (
        f"{'k':>4}"
        f" | {'Min':>10}"
        f" | {'Max':>10}"
        f" | {'Min %':>8}"
        f" | {'Max %':>8}"
        f" | {'Max/Min':>9}"
        f" | {'Small':>7}"
    )

    print(header)
    print("-" * len(header))

    for k in k_values:

        stats = summary[str(k)]

        print(
            f"{k:>4}"
            f" | {stats['min_cluster_count']:>10,}"
            f" | {stats['max_cluster_count']:>10,}"
            f" | {stats['min_cluster_percentage']:>7.2f}%"
            f" | {stats['max_cluster_percentage']:>7.2f}%"
            f" | {stats['max_min_size_ratio']:>9.2f}"
            f" | {stats['num_small_clusters']:>7}"
        )


# Save JSON

def save_json(
    data: dict,
    output_path: Path,
) -> None:
    """
    Save results as JSON.
    """

    with open(
        output_path,
        "w",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
        )

    print(
        f"Saved JSON results to: {output_path}"
    )


# Save CSV

def save_csv(
    cluster_sizes: dict[int, dict[int, dict[str, float]]],
    summary: dict,
    k_values: list[int],
    output_path: Path,
) -> None:
    """
    Save one row per k with cluster counts,
    percentages, and summary statistics.
    """

    max_k = max(k_values)

    fieldnames = ["k"]

    for cluster_id in range(max_k):
        fieldnames.extend(
            [
                f"cluster_{cluster_id}_count",
                f"cluster_{cluster_id}_percentage",
            ]
        )

    fieldnames.extend(
        [
            "min_cluster_count",
            "max_cluster_count",
            "min_cluster_percentage",
            "max_cluster_percentage",
            "max_min_size_ratio",
            "num_small_clusters",
        ]
    )

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

        for k in k_values:

            row = {"k": k}

            for cluster_id in range(max_k):

                if cluster_id in cluster_sizes[k]:

                    result = cluster_sizes[k][cluster_id]

                    row[
                        f"cluster_{cluster_id}_count"
                    ] = result["count"]

                    row[
                        f"cluster_{cluster_id}_percentage"
                    ] = result["percentage"]

                else:

                    row[
                        f"cluster_{cluster_id}_count"
                    ] = ""

                    row[
                        f"cluster_{cluster_id}_percentage"
                    ] = ""

            stats = summary[str(k)]

            row["min_cluster_count"] = (
                stats["min_cluster_count"]
            )

            row["max_cluster_count"] = (
                stats["max_cluster_count"]
            )

            row["min_cluster_percentage"] = (
                stats["min_cluster_percentage"]
            )

            row["max_cluster_percentage"] = (
                stats["max_cluster_percentage"]
            )

            row["max_min_size_ratio"] = (
                stats["max_min_size_ratio"]
            )

            row["num_small_clusters"] = (
                stats["num_small_clusters"]
            )

            writer.writerow(row)

    print(
        f"Saved CSV results to: {output_path}"
    )


# Plot cluster sizes

def plot_cluster_sizes(
    cluster_sizes: dict[int, dict[int, dict[str, float]]],
    k_values: list[int],
    output_path: Path,
) -> None:
    """
    Create a grouped bar chart showing cluster sizes
    across different k values.

    Since the number of clusters differs across k,
    each cluster ID is shown where it exists.
    """

    max_k = max(k_values)

    x = np.arange(len(k_values))

    width = 0.8 / max_k

    plt.figure(figsize=(11, 6))

    for cluster_id in range(max_k):

        counts = []

        for k in k_values:

            if cluster_id in cluster_sizes[k]:
                counts.append(
                    cluster_sizes[k][cluster_id]["count"]
                )
            else:
                counts.append(0)

        offset = (
            cluster_id
            - (max_k - 1) / 2
        ) * width

        bars = plt.bar(
            x + offset,
            counts,
            width,
            label=f"Cluster {cluster_id}",
        )

        for bar, count in zip(
            bars,
            counts,
        ):
            if count > 0:
                plt.text(
                    bar.get_x()
                    + bar.get_width() / 2,
                    bar.get_height(),
                    f"{count:,}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Number of Samples")
    plt.title(
        "Cluster Size Distributions Across DEC k-Sensitivity Runs"
    )

    plt.xticks(
        x,
        k_values,
    )

    plt.legend(
        title="Cluster",
    )

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


# Main

def main():

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    RUNS_ROOT = (
        PROJECT_ROOT
        / "experiments"
        / "final_k_sensitivity"
        / "runs"
    )

    ANALYSIS_ROOT = (
        PROJECT_ROOT
        / "analysis"
        / "final_k_sensitivity"
    )

    AGGREGATED_DIR = (
        ANALYSIS_ROOT
        / "aggregated"
    )

    FIGURES_DIR = (
        ANALYSIS_ROOT
        / "figures"
    )

    # Prepare output directories

    AGGREGATED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("FINAL DEC K-SENSITIVITY — CLUSTER SIZE ANALYSIS")
    print("=" * 80)

    print(
        f"\nRuns root: {RUNS_ROOT}"
    )

    print(
        f"Output directory: {ANALYSIS_ROOT}"
    )

    print(
        f"k values: {K_VALUES}"
    )

    # Load labels

    labels_by_k = load_labels(
        runs_root=RUNS_ROOT,
        k_values=K_VALUES,
    )

    # Validate labels

    validate_labels(
        labels_by_k=labels_by_k,
        k_values=K_VALUES,
    )

    # Compute cluster sizes

    cluster_sizes = compute_cluster_sizes(
        labels_by_k=labels_by_k,
        k_values=K_VALUES,
    )

    # Print distributions

    print_cluster_sizes(
        cluster_sizes=cluster_sizes,
        k_values=K_VALUES,
    )

    # Compute summary statistics

    summary = compute_summary_statistics(
        cluster_sizes=cluster_sizes,
        k_values=K_VALUES,
    )

    # Print summary

    print_summary(
        summary=summary,
        k_values=K_VALUES,
    )

    # Prepare complete results

    results = {
        "experiment": "final_k_sensitivity",
        "k_values": K_VALUES,
        "small_cluster_threshold_percentage": (
            SMALL_CLUSTER_THRESHOLD
        ),
        "cluster_sizes": {
            str(k): {
                str(cluster_id): values
                for cluster_id, values
                in cluster_results.items()
            }
            for k, cluster_results
            in cluster_sizes.items()
        },
        "summary": summary,
    }

    # Save JSON

    save_json(
        data=results,
        output_path=(
            AGGREGATED_DIR
            / "cluster_sizes_by_k.json"
        ),
    )

    # Save CSV

    save_csv(
        cluster_sizes=cluster_sizes,
        summary=summary,
        k_values=K_VALUES,
        output_path=(
            AGGREGATED_DIR
            / "cluster_sizes_by_k.csv"
        ),
    )

    # Plot

    plot_cluster_sizes(
        cluster_sizes=cluster_sizes,
        k_values=K_VALUES,
        output_path=(
            FIGURES_DIR
            / "cluster_sizes_by_k.png"
        ),
    )

    print("\n" + "=" * 80)
    print("CLUSTER SIZE ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()