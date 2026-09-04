from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score


def load_labels(runs_root: Path, seeds: list[int]) -> dict[int, np.ndarray]:
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


def validate_labels(labels_by_seed: dict[int, np.ndarray]) -> None:
    """
    Verify that all runs contain labels for the same number of samples.
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

    print(f"\nAll runs contain {next(iter(unique_lengths))} samples.")


def compute_ari_matrix(
    labels_by_seed: dict[int, np.ndarray],
    seeds: list[int],
) -> np.ndarray:
    """
    Compute pairwise Adjusted Rand Index between all seed runs.
    """

    n = len(seeds)

    ari_matrix = np.zeros((n, n), dtype=np.float64)

    for i, seed_i in enumerate(seeds):
        for j, seed_j in enumerate(seeds):
            ari_matrix[i, j] = adjusted_rand_score(
                labels_by_seed[seed_i],
                labels_by_seed[seed_j],
            )

    return ari_matrix


def summarize_ari(
    ari_matrix: np.ndarray,
    seeds: list[int],
) -> dict:
    """
    Compute summary statistics from the upper triangle
    of the pairwise ARI matrix, excluding the diagonal.
    """

    pairwise_results = []

    n = len(seeds)

    for i in range(n):
        for j in range(i + 1, n):
            pairwise_results.append(
                {
                    "seed_1": seeds[i],
                    "seed_2": seeds[j],
                    "ari": float(ari_matrix[i, j]),
                }
            )

    ari_values = np.array(
        [item["ari"] for item in pairwise_results],
        dtype=np.float64,
    )

    summary = {
        "num_runs": len(seeds),
        "seeds": seeds,
        "num_pairs": len(ari_values),
        "mean_ari": float(np.mean(ari_values)),
        "std_ari": float(np.std(ari_values)),
        "min_ari": float(np.min(ari_values)),
        "max_ari": float(np.max(ari_values)),
        "pairwise_ari": pairwise_results,
    }

    return summary


def save_ari_matrix(
    ari_matrix: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save the ARI matrix as a NumPy array.
    """

    np.save(output_path, ari_matrix)

    print(f"Saved ARI matrix to: {output_path}")


def save_ari_summary(
    summary: dict,
    output_path: Path,
) -> None:
    """
    Save ARI summary statistics as JSON.
    """

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved ARI summary to: {output_path}")


def plot_ari_heatmap(
    ari_matrix: np.ndarray,
    seeds: list[int],
    output_path: Path,
) -> None:
    """
    Create and save a heatmap of pairwise ARI values.
    """

    plt.figure(figsize=(5, 4))

    im = plt.imshow(
        ari_matrix,
        vmin=0,
        vmax=1,
        cmap="coolwarm",
    )

    plt.colorbar(
        im,
        fraction=0.046,
        pad=0.04,
        label="Adjusted Rand Index",
    )

    plt.xticks(
        range(len(seeds)),
        seeds,
    )

    plt.yticks(
        range(len(seeds)),
        seeds,
    )

    # Display ARI values inside the heatmap
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            plt.text(
                j,
                i,
                f"{ari_matrix[i, j]:.3f}",
                ha="center",
                va="center",
            )

    plt.xlabel("Seed")
    plt.ylabel("Seed")
    plt.title("Pairwise ARI Across Final DEC Runs")

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved ARI heatmap to: {output_path}")


def print_results(
    ari_matrix: np.ndarray,
    seeds: list[int],
    summary: dict,
) -> None:
    """
    Print the ARI matrix and summary statistics to the terminal.
    """

    print("\n" + "=" * 60)
    print("PAIRWISE ARI MATRIX")
    print("=" * 60)

    header = "       " + " ".join(f"{seed:>8}" for seed in seeds)
    print(header)

    for i, seed in enumerate(seeds):
        row = " ".join(
            f"{ari_matrix[i, j]:8.4f}"
            for j in range(len(seeds))
        )

        print(f"{seed:>5}  {row}")

    print("\n" + "=" * 60)
    print("ARI STABILITY SUMMARY")
    print("=" * 60)

    print(f"Number of runs : {summary['num_runs']}")
    print(f"Number of pairs: {summary['num_pairs']}")
    print(f"Mean ARI       : {summary['mean_ari']:.4f}")
    print(f"Std ARI        : {summary['std_ari']:.4f}")
    print(f"Minimum ARI    : {summary['min_ari']:.4f}")
    print(f"Maximum ARI    : {summary['max_ari']:.4f}")

    print("\nPairwise ARI:")

    for pair in summary["pairwise_ari"]:
        print(
            f"  Seed {pair['seed_1']} vs "
            f"Seed {pair['seed_2']}: "
            f"{pair['ari']:.4f}"
        )


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

    AGGREGATED_DIR = ANALYSIS_ROOT / "aggregated"
    FIGURES_DIR = ANALYSIS_ROOT / "figures"

    SEEDS = [42, 43, 44, 45, 46]

    # Prepare output directory

    AGGREGATED_DIR.mkdir(
        parents=True,
      exist_ok=True,
    )

    FIGURES_DIR.mkdir(
       parents=True,
       exist_ok=True,
    )

    print("=" * 60)
    print("FINAL DEC STABILITY — ARI ANALYSIS")
    print("=" * 60)

    print(f"\nRuns root: {RUNS_ROOT}")
    print(f"Output directory: {ANALYSIS_ROOT}")
    print(f"Seeds: {SEEDS}")

    # Load labels

    labels_by_seed = load_labels(
        runs_root=RUNS_ROOT,
        seeds=SEEDS,
    )

    # Validate labels

    validate_labels(labels_by_seed)

    # Compute pairwise ARI

    print("\nComputing pairwise ARI...")

    ari_matrix = compute_ari_matrix(
        labels_by_seed=labels_by_seed,
        seeds=SEEDS,
    )

    # Summarize ARI

    summary = summarize_ari(
        ari_matrix=ari_matrix,
        seeds=SEEDS,
    )

    # Print results

    print_results(
        ari_matrix=ari_matrix,
        seeds=SEEDS,
        summary=summary,
    )

    # Save results

    save_ari_matrix(
        ari_matrix=ari_matrix,
        output_path=AGGREGATED_DIR / "ari_matrix.npy",
    )

    save_ari_summary(
        summary=summary,
        output_path=AGGREGATED_DIR / "ari_summary.json",
    )

    # Plot heatmap

    plot_ari_heatmap(
        ari_matrix=ari_matrix,
        seeds=SEEDS,
        output_path=FIGURES_DIR / "ari_heatmap.png",
    )

    print("\n" + "=" * 60)
    print("ARI ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()