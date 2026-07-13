import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

RUNS_DIR = Path("experiments/alpha_sensitivity/runs")
OUTPUT_DIR = Path("analysis/alpha_sensitivity/aggregated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def discover_alpha_dirs(runs_dir: Path):
    """
    Returns a dict:
    {
        "0.1": Path(.../alpha_0.1),
        "0.5": Path(.../alpha_0.5),
    }
    """
    alpha_dirs = {}

    for alpha_dir in sorted(runs_dir.iterdir()):
        if not alpha_dir.is_dir():
            continue

        if not alpha_dir.name.startswith("alpha_"):
            continue

        alpha_str = alpha_dir.name.replace("alpha_", "")
        try:
            alpha_value = str(float(alpha_str))
        except ValueError:
            continue

        alpha_dirs[alpha_value] = alpha_dir

    return alpha_dirs

def load_val_labels_for_alpha(alpha_dir: Path):
    """
    Loads val_labels.npy from all seed_* subfolders of an alpha directory.

    Returns:
        List[np.ndarray] — one array per seed
    """
    labels = []

    for seed_dir in sorted(alpha_dir.iterdir()):
        if not seed_dir.is_dir():
            continue

        if not seed_dir.name.startswith("seed_"):
            continue

        label_path = seed_dir / "val_labels.npy"
        if not label_path.exists():
            print(f"Missing val_labels.npy in {seed_dir}")
            continue

        arr = np.load(label_path)
        labels.append(arr)

    return labels

def validate_labels(label_arrays, alpha):
    """
    Ensures all label arrays have the same length.
    """
    if len(label_arrays) < 2:
        raise ValueError(f"Alpha={alpha}: fewer than 2 seeds found")

    lengths = {len(arr) for arr in label_arrays}
    if len(lengths) != 1:
        raise ValueError(
            f"Alpha={alpha}: inconsistent label lengths: {lengths}"
        )
    
def compute_ari_matrix(label_arrays):
    """
    Computes the Adjusted Rand Index (ARI) matrix for a list of label arrays.

    Args:
        label_arrays: List[np.ndarray] — each of shape (n_samples)

    Returns:
        ari_matrix: np.ndarray of shape (n_seeds, n_seeds)
    """
    n = len(label_arrays)
    ari_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            ari_matrix[i, j] = adjusted_rand_score(
                label_arrays[i], 
                label_arrays[j]
            )

    return ari_matrix

def summarize_ari_matrix(ari_matrix):
    """
    Compute mean and std of ARI excluding diagonal.
    """
    n = ari_matrix.shape[0]

    # upper triangle, excluding diagonal
    upper = []
    for i in range(n):
        for j in range(i + 1, n):
                upper.append(ari_matrix[i, j])

    upper = np.array(upper)

    return {
        "mean_ari": float(np.mean(upper)),
        "std_ari": float(np.std(upper))
    }

if __name__ == "__main__":
    alpha_dirs = discover_alpha_dirs(RUNS_DIR)

    labels_by_alpha = {}

    for alpha, alpha_dir in alpha_dirs.items():
        label_arrays = load_val_labels_for_alpha(alpha_dir)

        if len(label_arrays) < 2:
            print(f"Skipping Alpha={alpha}: not enough seeds")
            continue

        validate_labels(label_arrays, alpha)
        labels_by_alpha[alpha] = label_arrays

    if not labels_by_alpha:
        raise RuntimeError("No valid alphas with multiple seeds found.")

    print("Loaded validation labels:")
    for alpha, arrays in labels_by_alpha.items():
        print(f"  Alpha={alpha}: {len(arrays)} seeds, {len(arrays[0])} samples")

    ari_summaries = {}

    for alpha, label_arrays in labels_by_alpha.items():
        print(f"\nComputing ARI for Alpha={alpha}")

        ari_matrix = compute_ari_matrix(label_arrays)
        summary = summarize_ari_matrix(ari_matrix)

        ari_summaries[alpha] = summary

        # save matrix for plots later
        npy_path = OUTPUT_DIR / f"ari_matrix_alpha_{alpha}.npy"
        np.save(npy_path, ari_matrix)

        print(
            f" mean ARI = {summary['mean_ari']:.4f},"
            f"std = {summary['std_ari']:.4f}"
        )

    ari_json_path = OUTPUT_DIR / f"ari_summary_alpha.json"
    with open(ari_json_path, "w") as f:
        json.dump(ari_summaries, f, indent=2)

    print("\nSaved ARI summary to:", ari_json_path)