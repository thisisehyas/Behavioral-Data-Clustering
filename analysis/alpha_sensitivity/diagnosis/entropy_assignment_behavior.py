import json
from pathlib import Path

import numpy as np


# Configuration

ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0]
SEEDS = [0, 1, 2, 3, 4]

RUNS_ROOT = Path("experiments/alpha_sensitivity/runs")
RESULTS_DIR = Path("analysis/alpha_sensitivity/aggregated")

OUTPUT_PATH = RESULTS_DIR / "entropy_assignment_behavior.json"


# Helpers

def alpha_key(alpha):
    """Return the directory name for an alpha value."""
    return f"alpha_{alpha}"


def seed_key(seed):
    """Return the directory name for a seed."""
    return f"seed_{seed}"


def load_run_files(alpha, seed):
    """Load the files required for this diagnostic."""

    run_dir = RUNS_ROOT / alpha_key(alpha) / seed_key(seed)

    q_path = run_dir / "q_val.npy"
    labels_path = run_dir / "val_labels.npy"

    if not q_path.exists():
        raise FileNotFoundError(f"Missing q_val.npy: {q_path}")

    if not labels_path.exists():
        raise FileNotFoundError(f"Missing val_labels.npy: {labels_path}")

    q = np.load(q_path)
    labels = np.load(labels_path)

    return q, labels, run_dir


def validate_q(q, alpha, seed):
    """Validate soft assignment probabilities."""

    if q.ndim != 2:
        raise ValueError(
            f"Alpha={alpha}, seed={seed}: "
            f"q_val.npy must be 2D, got shape {q.shape}"
        )

    if not np.all(np.isfinite(q)):
        raise ValueError(
            f"Alpha={alpha}, seed={seed}: q contains non-finite values."
        )

    row_sums = np.sum(q, axis=1)

    if not np.allclose(row_sums, 1.0, atol=1e-5):
        max_error = np.max(np.abs(row_sums - 1.0))
        raise ValueError(
            f"Alpha={alpha}, seed={seed}: q rows do not sum to 1. "
            f"Maximum error = {max_error}"
        )

    if np.any(q < 0):
        raise ValueError(
            f"Alpha={alpha}, seed={seed}: q contains negative probabilities."
        )


def calculate_entropy(q):
    """
    Calculate mean assignment entropy.

    H_i = -sum_j q_ij log(q_ij)

    The returned value is the mean entropy across validation samples.
    """

    q_safe = np.clip(q, 1e-12, 1.0)

    sample_entropy = -np.sum(
        q_safe * np.log(q_safe),
        axis=1,
    )

    return float(np.mean(sample_entropy))


def calculate_cluster_statistics(labels):
    """Calculate smallest/largest cluster and imbalance."""

    cluster_ids, counts = np.unique(labels, return_counts=True)

    smallest = int(np.min(counts))
    largest = int(np.max(counts))

    imbalance_ratio = float(largest / smallest)

    return {
        "num_clusters": int(len(cluster_ids)),
        "smallest_cluster": smallest,
        "largest_cluster": largest,
        "imbalance_ratio": imbalance_ratio,
    }


def load_ari_matrix(alpha):
    """
    Load the pairwise ARI matrix for one alpha.

    Expected filename:
        ari_matrix_alpha_<alpha>.npy
    """

    path = RESULTS_DIR / f"ari_matrix_alpha_{alpha}.npy"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing ARI matrix for alpha={alpha}: {path}"
        )

    ari_matrix = np.load(path)

    expected_shape = (len(SEEDS), len(SEEDS))

    if ari_matrix.shape != expected_shape:
        raise ValueError(
            f"Alpha={alpha}: expected ARI matrix shape "
            f"{expected_shape}, got {ari_matrix.shape}"
        )

    return ari_matrix


def calculate_seed_ari(ari_matrix, seed_index):
    """
    Calculate the mean pairwise ARI of one seed against
    all other seeds.

    This is an assignment-stability measure, not ground-truth ARI.
    """

    other_indices = [
        i for i in range(len(SEEDS))
        if i != seed_index
    ]

    return float(np.mean(ari_matrix[seed_index, other_indices]))


def pearson_correlation(x, y):
    """Calculate Pearson correlation safely."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return None

    if np.std(x) == 0 or np.std(y) == 0:
        return None

    return float(np.corrcoef(x, y)[0, 1])


def summarize(values):
    """Return mean, std, min, and max."""

    values = np.asarray(values, dtype=float)

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


# Main diagnostic

def main():

    print("=" * 90)
    print("ENTROPY / ASSIGNMENT BEHAVIOR DIAGNOSTIC — ALPHA EXPERIMENT")
    print("=" * 90)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_seed = []

    # Per-seed analysis

    for alpha in ALPHAS:

        print()
        print("=" * 90)
        print(f"ALPHA = {alpha}")
        print("=" * 90)

        ari_matrix = load_ari_matrix(alpha)

        for seed_index, seed in enumerate(SEEDS):

            q, labels, run_dir = load_run_files(alpha, seed)

            validate_q(q, alpha, seed)

            if len(labels) != len(q):
                raise ValueError(
                    f"Alpha={alpha}, seed={seed}: "
                    f"q has {len(q)} samples but labels have "
                    f"{len(labels)} samples."
                )

            entropy = calculate_entropy(q)

            cluster_stats = calculate_cluster_statistics(labels)

            seed_ari = calculate_seed_ari(
                ari_matrix,
                seed_index,
            )

            result = {
                "alpha": alpha,
                "seed": seed,
                "entropy": entropy,
                "seed_mean_ari": seed_ari,
                **cluster_stats,
            }

            per_seed.append(result)

            print(
                f"Seed {seed}: "
                f"Entropy = {entropy:.6f} | "
                f"Mean ARI = {seed_ari:.6f} | "
                f"Smallest = {cluster_stats['smallest_cluster']} | "
                f"Largest = {cluster_stats['largest_cluster']} | "
                f"Imbalance = {cluster_stats['imbalance_ratio']:.2f}"
            )

    # Summary by alpha

    summary_by_alpha = {}

    for alpha in ALPHAS:

        rows = [
            row for row in per_seed
            if row["alpha"] == alpha
        ]

        entropy_values = [
            row["entropy"] for row in rows
        ]

        ari_values = [
            row["seed_mean_ari"] for row in rows
        ]

        imbalance_values = [
            row["imbalance_ratio"] for row in rows
        ]

        smallest_values = [
            row["smallest_cluster"] for row in rows
        ]

        largest_values = [
            row["largest_cluster"] for row in rows
        ]

        summary_by_alpha[str(alpha)] = {
            "entropy": summarize(entropy_values),
            "seed_mean_ari": summarize(ari_values),
            "imbalance_ratio": summarize(imbalance_values),
            "smallest_cluster": summarize(smallest_values),
            "largest_cluster": summarize(largest_values),
        }

    # Seeds 1-4 summary

    summary_seeds_1_to_4 = {}

    for alpha in ALPHAS:

        rows = [
            row for row in per_seed
            if row["alpha"] == alpha
            and row["seed"] in [1, 2, 3, 4]
        ]

        summary_seeds_1_to_4[str(alpha)] = {
            "entropy": summarize(
                [row["entropy"] for row in rows]
            ),
            "seed_mean_ari": summarize(
                [row["seed_mean_ari"] for row in rows]
            ),
            "imbalance_ratio": summarize(
                [row["imbalance_ratio"] for row in rows]
            ),
            "smallest_cluster": summarize(
                [row["smallest_cluster"] for row in rows]
            ),
            "largest_cluster": summarize(
                [row["largest_cluster"] for row in rows]
            ),
        }

    # Entropy relationships

    all_entropy = np.array(
        [row["entropy"] for row in per_seed]
    )

    all_ari = np.array(
        [row["seed_mean_ari"] for row in per_seed]
    )

    all_imbalance = np.array(
        [row["imbalance_ratio"] for row in per_seed]
    )

    entropy_vs_ari = pearson_correlation(
        all_entropy,
        all_ari,
    )

    entropy_vs_imbalance = pearson_correlation(
        all_entropy,
        all_imbalance,
    )

    relationships = {
        "entropy_vs_seed_ari": {
            "pearson_r": entropy_vs_ari,
            "interpretation": (
                "Negative values indicate that lower entropy tends "
                "to be associated with greater assignment stability."
            ),
        },
        "entropy_vs_cluster_imbalance": {
            "pearson_r": entropy_vs_imbalance,
            "interpretation": (
                "Positive values indicate that higher entropy tends "
                "to be associated with greater cluster-size imbalance."
            ),
        },
    }

    # Alpha 2 and Alpha 5 focused comparison

    focused_comparison = {}

    for alpha in [2.0, 5.0]:

        rows = [
            row for row in per_seed
            if row["alpha"] == alpha
        ]

        focused_comparison[str(alpha)] = {
            "per_seed": rows,
            "summary": summary_by_alpha[str(alpha)],
        }

    # Final JSON

    output = {
        "alphas": ALPHAS,
        "seeds": SEEDS,
        "interpretation": {
            "entropy": "lower_is_more_confident",
            "seed_mean_ari": (
                "higher_is_more_stable_across_random_seeds"
            ),
            "imbalance_ratio": (
                "higher_means_greater_cluster_size_imbalance"
            ),
            "ari_note": (
                "ARI here is pairwise inter-seed assignment stability, "
                "not ARI against ground-truth labels."
            ),
        },
        "per_seed": per_seed,
        "summary_by_alpha": summary_by_alpha,
        "summary_seeds_1_to_4": summary_seeds_1_to_4,
        "relationships": relationships,
        "focused_alpha_2_and_5": focused_comparison,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Concise diagnostic report

    print()
    print("=" * 90)
    print("MEAN ± STD — ALL SEEDS")
    print("=" * 90)

    for alpha in ALPHAS:

        summary = summary_by_alpha[str(alpha)]

        print(
            f"α={alpha}: "
            f"Entropy={summary['entropy']['mean']:.6f} "
            f"± {summary['entropy']['std']:.6f} | "
            f"ARI={summary['seed_mean_ari']['mean']:.6f} "
            f"± {summary['seed_mean_ari']['std']:.6f} | "
            f"Imbalance={summary['imbalance_ratio']['mean']:.2f} "
            f"± {summary['imbalance_ratio']['std']:.2f}"
        )

    print()
    print("=" * 90)
    print("SEEDS 1-4 ONLY")
    print("=" * 90)

    for alpha in ALPHAS:

        summary = summary_seeds_1_to_4[str(alpha)]

        print(
            f"α={alpha}: "
            f"Entropy={summary['entropy']['mean']:.6f} "
            f"± {summary['entropy']['std']:.6f} | "
            f"ARI={summary['seed_mean_ari']['mean']:.6f} "
            f"± {summary['seed_mean_ari']['std']:.6f} | "
            f"Imbalance={summary['imbalance_ratio']['mean']:.2f} "
            f"± {summary['imbalance_ratio']['std']:.2f}"
        )

    print()
    print("=" * 90)
    print("ENTROPY RELATIONSHIPS")
    print("=" * 90)

    print(
        f"Entropy vs. seed-ARI: "
        f"Pearson r = {entropy_vs_ari:.4f}"
        if entropy_vs_ari is not None
        else "Entropy vs. seed-ARI: unavailable"
    )

    print(
        f"Entropy vs. imbalance: "
        f"Pearson r = {entropy_vs_imbalance:.4f}"
        if entropy_vs_imbalance is not None
        else "Entropy vs. imbalance: unavailable"
    )

    print()
    print("=" * 90)
    print("ALPHA = 2.0 vs ALPHA = 5.0")
    print("=" * 90)

    for alpha in [2.0, 5.0]:

        summary = summary_by_alpha[str(alpha)]

        print(
            f"α={alpha}: "
            f"Entropy={summary['entropy']['mean']:.6f} ± "
            f"{summary['entropy']['std']:.6f} | "
            f"ARI={summary['seed_mean_ari']['mean']:.6f} ± "
            f"{summary['seed_mean_ari']['std']:.6f} | "
            f"Imbalance={summary['imbalance_ratio']['mean']:.2f} ± "
            f"{summary['imbalance_ratio']['std']:.2f}"
        )

    print()
    print("=" * 90)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 90)
    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()