from pathlib import Path

import csv
import json

import matplotlib.pyplot as plt
import numpy as np


def load_labels(
    runs_root: Path,
    seeds: list[int],
) -> dict[int, np.ndarray]:
    """
    Load full-dataset cluster labels for each random seed.
    """

    labels_by_seed = {}

    for seed in seeds:
        run_dir = runs_root / f"seed_{seed}"
        label_path = run_dir / "full_labels.npy"

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing full_labels.npy for seed {seed}: {label_path}"
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
            f"{len(labels)} samples, "
            f"{len(np.unique(labels))} clusters"
        )

    return labels_by_seed


def validate_labels(
    labels_by_seed: dict[int, np.ndarray],
) -> None:
    """
    Verify that all runs contain labels for the same number of samples
    and the expected number of clusters.
    """

    lengths = {
        seed: len(labels)
        for seed, labels in labels_by_seed.items()
    }

    unique_lengths = set(lengths.values())

    if len(unique_lengths) != 1:
        raise ValueError(
            f"Inconsistent number of samples across runs: {lengths}"
        )

    num_clusters = {
        seed: len(np.unique(labels))
        for seed, labels in labels_by_seed.items()
    }

    if len(set(num_clusters.values())) != 1:
        raise ValueError(
            f"Inconsistent number of clusters across runs: "
            f"{num_clusters}"
        )

    print(
        f"\nAll runs contain "
        f"{next(iter(unique_lengths))} samples."
    )

    print(
        f"All runs contain "
        f"{next(iter(num_clusters.values()))} clusters."
    )


def compute_cluster_sizes(
    labels_by_seed: dict[int, np.ndarray],
    seeds: list[int],
) -> dict[int, dict[int, dict[str, float]]]:
    """
    Compute cluster counts and percentages for each seed.

    Returns:
        {
            seed: {
                cluster_id: {
                    "count": ...,
                    "percentage": ...
                }
            }
        }
    """

    results = {}

    for seed in seeds:
        labels = labels_by_seed[seed]
        total_samples = len(labels)

        unique_labels, counts = np.unique(
            labels,
            return_counts=True,
        )

        seed_results = {}

        for cluster_id, count in zip(
            unique_labels,
            counts,
        ):
            percentage = (
                count / total_samples
            ) * 100.0

            seed_results[int(cluster_id)] = {
                "count": int(count),
                "percentage": float(percentage),
            }

        results[seed] = seed_results

    return results


def print_cluster_sizes(
    cluster_sizes: dict[int, dict[int, dict[str, float]]],
    seeds: list[int],
) -> None:
    """
    Print cluster sizes and percentages in a readable table.
    """

    print("\n" + "=" * 80)
    print("CLUSTER SIZE DISTRIBUTIONS")
    print("=" * 80)

    all_clusters = sorted(
        {
            cluster_id
            for seed_results in cluster_sizes.values()
            for cluster_id in seed_results
        }
    )

    header = (
        f"{'Seed':>8}"
        + "".join(
            f" | C{cluster_id} Count | C{cluster_id} %"
            for cluster_id in all_clusters
        )
    )

    print(header)
    print("-" * len(header))

    for seed in seeds:
        row = f"{seed:>8}"

        for cluster_id in all_clusters:
            result = cluster_sizes[seed][cluster_id]

            row += (
                f" | {result['count']:>13,}"
                f" | {result['percentage']:>6.2f}%"
            )

        print(row)


def compute_summary_statistics(
    cluster_sizes: dict[int, dict[int, dict[str, float]]],
    seeds: list[int],
) -> dict:
    """
    Compute mean, standard deviation, minimum, and maximum
    cluster size across seeds.

    Note:
    Cluster IDs are kept as stored in each run. This analysis is
    intentionally descriptive; it does not attempt to match clusters
    across seeds because cluster labels are arbitrary.
    """

    all_clusters = sorted(
        {
            cluster_id
            for seed_results in cluster_sizes.values()
            for cluster_id in seed_results
        }
    )

    summary = {}

    for cluster_id in all_clusters:
        counts = np.array(
            [
                cluster_sizes[seed][cluster_id]["count"]
                for seed in seeds
            ],
            dtype=np.float64,
        )

        percentages = np.array(
            [
                cluster_sizes[seed][cluster_id]["percentage"]
                for seed in seeds
            ],
            dtype=np.float64,
        )

        summary[str(cluster_id)] = {
            "count": {
                "mean": float(np.mean(counts)),
                "std": float(np.std(counts)),
                "min": int(np.min(counts)),
                "max": int(np.max(counts)),
            },
            "percentage": {
                "mean": float(np.mean(percentages)),
                "std": float(np.std(percentages)),
                "min": float(np.min(percentages)),
                "max": float(np.max(percentages)),
            },
        }

    return summary


def save_json(
    data: dict,
    output_path: Path,
) -> None:
    """
    Save results as JSON.
    """

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved JSON results to: {output_path}")


def save_csv(
    cluster_sizes: dict[int, dict[int, dict[str, float]]],
    seeds: list[int],
    output_path: Path,
) -> None:
    """
    Save cluster-size results as a CSV table.
    """

    all_clusters = sorted(
        {
            cluster_id
            for seed_results in cluster_sizes.values()
            for cluster_id in seed_results
        }
    )

    fieldnames = ["seed"]

    for cluster_id in all_clusters:
        fieldnames.extend(
            [
                f"cluster_{cluster_id}_count",
                f"cluster_{cluster_id}_percentage",
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

        for seed in seeds:
            row = {"seed": seed}

            for cluster_id in all_clusters:
                result = cluster_sizes[seed][cluster_id]

                row[f"cluster_{cluster_id}_count"] = (
                    result["count"]
                )

                row[f"cluster_{cluster_id}_percentage"] = (
                    result["percentage"]
                )

            writer.writerow(row)

    print(f"Saved CSV results to: {output_path}")


def plot_cluster_sizes(
    cluster_sizes: dict[int, dict[int, dict[str, float]]],
    seeds: list[int],
    output_path: Path,
) -> None:
    """
    Create a grouped bar chart showing cluster sizes
    across random seeds.

    Counts are shown because the total sample size is identical
    across all runs.
    """

    all_clusters = sorted(
        {
            cluster_id
            for seed_results in cluster_sizes.values()
            for cluster_id in seed_results
        }
    )

    x = np.arange(len(seeds))

    width = 0.8 / len(all_clusters)

    plt.figure(figsize=(10, 6))

    for i, cluster_id in enumerate(all_clusters):
        counts = [
            cluster_sizes[seed][cluster_id]["count"]
            for seed in seeds
        ]

        offset = (
            i - (len(all_clusters) - 1) / 2
        ) * width

        bars = plt.bar(
            x + offset,
            counts,
            width,
            label=f"Cluster {cluster_id}",
        )

        # Add count labels above each bar
        for bar, count in zip(bars, counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.xlabel("Random Seed")
    plt.ylabel("Number of Samples")
    plt.title(
        "Cluster Size Distributions Across Final DEC Runs"
    )

    plt.xticks(
        x,
        seeds,
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

    print(f"Saved cluster-size figure to: {output_path}")


def main():
    # Configuration

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    RUNS_ROOT = (
        PROJECT_ROOT
        / "experiments"
        / "final_stability"
        / "runs"
    )

    ANALYSIS_ROOT = (
        PROJECT_ROOT
        / "analysis"
        / "final_stability"
    )

    AGGREGATED_DIR = (
        ANALYSIS_ROOT
        / "aggregated"
    )

    FIGURES_DIR = (
        ANALYSIS_ROOT
        / "figures"
    )

    SEEDS = [42, 43, 44, 45, 46]

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
    print("FINAL DEC STABILITY — CLUSTER SIZE ANALYSIS")
    print("=" * 80)

    print(f"\nRuns root: {RUNS_ROOT}")
    print(f"Output directory: {ANALYSIS_ROOT}")
    print(f"Seeds: {SEEDS}")

    # Load labels

    labels_by_seed = load_labels(
        runs_root=RUNS_ROOT,
        seeds=SEEDS,
    )

    # Validate labels

    validate_labels(
        labels_by_seed=labels_by_seed,
    )

    # Compute cluster sizes

    cluster_sizes = compute_cluster_sizes(
        labels_by_seed=labels_by_seed,
        seeds=SEEDS,
    )

    # Print results

    print_cluster_sizes(
        cluster_sizes=cluster_sizes,
        seeds=SEEDS,
    )

    # Compute summary statistics

    summary = compute_summary_statistics(
        cluster_sizes=cluster_sizes,
        seeds=SEEDS,
    )

    print("\n" + "=" * 80)
    print("CLUSTER SIZE SUMMARY")
    print("=" * 80)

    for cluster_id, stats in summary.items():
        print(
            f"\nCluster {cluster_id}:"
        )

        print(
            f"  Count:"
            f" mean={stats['count']['mean']:.1f},"
            f" std={stats['count']['std']:.1f},"
            f" min={stats['count']['min']:,},"
            f" max={stats['count']['max']:,}"
        )

        print(
            f"  Percentage:"
            f" mean={stats['percentage']['mean']:.2f}%,"
            f" std={stats['percentage']['std']:.2f}%,"
            f" min={stats['percentage']['min']:.2f}%,"
            f" max={stats['percentage']['max']:.2f}%"
        )

    # Save JSON

    results = {
        "experiment": "final_stability",
        "seeds": SEEDS,
        "cluster_sizes": {
            str(seed): {
                str(cluster_id): values
                for cluster_id, values
                in cluster_results.items()
            }
            for seed, cluster_results
            in cluster_sizes.items()
        },
        "summary": summary,
    }

    save_json(
        data=results,
        output_path=(
            AGGREGATED_DIR
            / "cluster_sizes.json"
        ),
    )

    # Save CSV

    save_csv(
        cluster_sizes=cluster_sizes,
        seeds=SEEDS,
        output_path=(
            AGGREGATED_DIR
            / "cluster_sizes.csv"
        ),
    )

    # Plot

    plot_cluster_sizes(
        cluster_sizes=cluster_sizes,
        seeds=SEEDS,
        output_path=(
            FIGURES_DIR
            / "cluster_sizes.png"
        ),
    )

    print("\n" + "=" * 80)
    print("CLUSTER SIZE ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()