from pathlib import Path

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_rand_score


CONDITIONS = ["frozen", "trainable"]
N_CLUSTERS = 5


def load_labels(
    runs_root: Path,
    conditions: list[str],
) -> dict[str, np.ndarray]:
    """
    Load full-dataset cluster labels for each ablation condition.
    """

    labels_by_condition = {}

    for condition in conditions:

        run_dir = runs_root / condition
        label_path = run_dir / "full_labels.npy"

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing full_labels.npy for condition "
                f"'{condition}': {label_path}"
            )

        labels = np.load(label_path)

        if labels.ndim != 1:
            raise ValueError(
                f"Labels for condition '{condition}' must be "
                f"1-dimensional, but got shape {labels.shape}."
            )

        n_clusters = len(np.unique(labels))

        if n_clusters != N_CLUSTERS:
            raise ValueError(
                f"Condition '{condition}' contains "
                f"{n_clusters} clusters; expected {N_CLUSTERS}."
            )

        labels_by_condition[condition] = labels

        print(
            f"Loaded {condition}: "
            f"{len(labels)} samples, "
            f"{n_clusters} clusters"
        )

    return labels_by_condition


def validate_labels(
    labels_by_condition: dict[str, np.ndarray],
) -> None:
    """
    Verify that both conditions contain labels for the same
    observations in the same order.
    """

    lengths = {
        condition: len(labels)
        for condition, labels in labels_by_condition.items()
    }

    unique_lengths = set(lengths.values())

    if len(unique_lengths) != 1:
        raise ValueError(
            f"Inconsistent number of samples across conditions: "
            f"{lengths}"
        )

    n_samples = next(iter(unique_lengths))

    print(
        f"\nBoth conditions contain {n_samples} samples."
    )


def compute_ari(
    labels_by_condition: dict[str, np.ndarray],
) -> float:
    """
    Compute Adjusted Rand Index between the frozen and
    trainable encoder partitions.
    """

    ari = adjusted_rand_score(
        labels_by_condition["frozen"],
        labels_by_condition["trainable"],
    )

    return float(ari)


def compute_contingency_table(
    labels_by_condition: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compute the observation-level contingency table between
    frozen and trainable cluster assignments.
    """

    frozen_labels = labels_by_condition["frozen"]
    trainable_labels = labels_by_condition["trainable"]

    table = pd.crosstab(
        pd.Series(
            frozen_labels,
            name="frozen_cluster",
        ),
        pd.Series(
            trainable_labels,
            name="trainable_cluster",
        ),
    )

    table = table.reindex(
        index=range(N_CLUSTERS),
        columns=range(N_CLUSTERS),
        fill_value=0,
    )

    return table


def compute_row_percentages(
    contingency: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert contingency counts into row-wise percentages.

    Each row shows how observations from one frozen cluster
    are distributed across trainable clusters.
    """

    row_percentages = (
        contingency.div(
            contingency.sum(axis=1),
            axis=0,
        )
        * 100
    )

    return row_percentages


def save_results(
    ari: float,
    n_samples: int,
    contingency: pd.DataFrame,
    row_percentages: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save ARI summary and contingency tables.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "comparison": "frozen_vs_trainable",
        "n_samples": n_samples,
        "n_clusters": N_CLUSTERS,
        "adjusted_rand_index": ari,
        "interpretation_note": (
            "ARI is label-invariant and measures agreement between "
            "the frozen-encoder and trainable-encoder partitions."
        ),
    }

    with open(
        output_dir / "ari_comparison.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    contingency.to_csv(
        output_dir / "partition_contingency.csv"
    )

    row_percentages.to_csv(
        output_dir / "partition_contingency_row_percentages.csv"
    )

    print(
        f"Saved ARI summary to: "
        f"{output_dir / 'ari_comparison.json'}"
    )

    print(
        f"Saved contingency table to: "
        f"{output_dir / 'partition_contingency.csv'}"
    )

    print(
        f"Saved row percentages to: "
        f"{output_dir / 'partition_contingency_row_percentages.csv'}"
    )


def plot_contingency_heatmap(
    contingency: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot the observation-level contingency table between
    frozen and trainable partitions.
    """

    plt.figure(figsize=(6, 5))

    im = plt.imshow(
        contingency.values,
        cmap="Blues",
    )

    plt.colorbar(
        im,
        fraction=0.046,
        pad=0.04,
        label="Number of observations",
    )

    plt.xticks(
        range(N_CLUSTERS),
        contingency.columns,
    )

    plt.yticks(
        range(N_CLUSTERS),
        contingency.index,
    )

    for i in range(N_CLUSTERS):
        for j in range(N_CLUSTERS):
            plt.text(
                j,
                i,
                f"{contingency.iloc[i, j]:,}",
                ha="center",
                va="center",
            )

    plt.xlabel("Trainable encoder cluster")
    plt.ylabel("Frozen encoder cluster")
    plt.title("Partition Contingency: Frozen vs Trainable")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved contingency heatmap to: {output_path}"
    )


def print_results(
    ari: float,
    contingency: pd.DataFrame,
    row_percentages: pd.DataFrame,
) -> None:
    """
    Print the main partition-agreement results.
    """

    print("\n" + "=" * 60)
    print("PARTITION AGREEMENT — ARI")
    print("=" * 60)

    print(
        f"\nAdjusted Rand Index: {ari:.4f}"
    )

    print("\n" + "=" * 60)
    print("CONTINGENCY TABLE")
    print("=" * 60)

    print(contingency)

    print("\n" + "=" * 60)
    print("ROW-WISE PERCENTAGES")
    print("=" * 60)

    print(
        row_percentages.round(2)
    )


def main():

    # Project paths
    PROJECT_ROOT = (
        Path(__file__).resolve().parents[2]
    )

    RUNS_ROOT = (
        PROJECT_ROOT
        / "experiments"
        / "encoder_ablation"
        / "runs"
    )

    ANALYSIS_ROOT = (
        PROJECT_ROOT
        / "analysis"
        / "encoder_ablation"
    )

    AGGREGATED_DIR = (
        ANALYSIS_ROOT / "aggregated"
    )

    FIGURES_DIR = (
        ANALYSIS_ROOT / "figures"
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

    print("=" * 60)
    print("DEC ENCODER ABLATION — PARTITION AGREEMENT")
    print("=" * 60)

    print(
        f"\nRuns root: {RUNS_ROOT}"
    )

    print(
        f"Output directory: {ANALYSIS_ROOT}"
    )

    print(
        f"Conditions: {CONDITIONS}"
    )

    # Load labels
    labels_by_condition = load_labels(
        runs_root=RUNS_ROOT,
        conditions=CONDITIONS,
    )

    # Validate
    validate_labels(
        labels_by_condition
    )

    # Compute ARI
    print("\nComputing ARI...")

    ari = compute_ari(
        labels_by_condition
    )

    # Compute contingency table
    print(
        "Computing partition contingency table..."
    )

    contingency = compute_contingency_table(
        labels_by_condition
    )

    row_percentages = compute_row_percentages(
        contingency
    )

    # Print results
    print_results(
        ari=ari,
        contingency=contingency,
        row_percentages=row_percentages,
    )

    # Save results
    save_results(
        ari=ari,
        n_samples=len(
            labels_by_condition["frozen"]
        ),
        contingency=contingency,
        row_percentages=row_percentages,
        output_dir=AGGREGATED_DIR,
    )

    # Plot contingency heatmap
    plot_contingency_heatmap(
        contingency=contingency,
        output_path=(
            FIGURES_DIR
            / "partition_contingency_heatmap.png"
        ),
    )

    print("\n" + "=" * 60)
    print("PARTITION AGREEMENT ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()