import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

RUNS_DIR = Path("experiments/lr-tuning/runs")
OUTPUT_DIR = Path("analysis/aggregated")

def discover_lr_dirs(runs_dir: Path):
    """
    Returns a dict:
    {
        "0.001": Path(.../lr_1e-03),
        "0.01": Path(.../lr_1e-02)
    }
    """
    lr_dirs = {}

    for lr_dir in sorted(runs_dir.iterdir()):
        if not lr_dir.is_dir():
            continue

        if not lr_dir.name.startswith("lr_"):
            continue

        lr_str = lr_dir.name.replace("lr_", "")
        try:
            lr_value = str(float(lr_str))
        except ValueError:
            continue

        lr_dirs[lr_value] = lr_dir

    return lr_dirs

def load_val_labels_for_lr(lr_dir: Path):
    """
    Loads val_labels.npy from all seed_* subfolders of an LR directory.

    Returns:
        List[np.ndarray] — one array per seed
    """
    labels = []

    for seed_dir in sorted(lr_dir.iterdir()):
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

def validate_labels(label_arrays, lr):
    """
    Ensures all label arrays have the same length.
    """
    if len(label_arrays) < 2:
        raise ValueError(f"LR={lr}: fewer than 2 seeds found")

    lengths = {len(arr) for arr in label_arrays}
    if len(lengths) != 1:
        raise ValueError(
            f"LR={lr}: inconsistent label lengths: {lengths}"
        )
    
def compute_ari_matrix(label_arrays):
    """
    Computes the Adjusted Rand Index (ARI) matrix for a list of label arrays.

    Args:
        label_arrays: List[np.ndarray] — shape (n_seed, n_samples)

    Returns:
        ari_matrix: np.ndarray of shape (n_seed, n_samples)
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
    lr_dirs = discover_lr_dirs(RUNS_DIR)

    labels_by_lr = {}

    for lr, lr_dir in lr_dirs.items():
        label_arrays = load_val_labels_for_lr(lr_dir)

        if len(label_arrays) < 2:
            print(f"Skipping LR={lr}: not enough seeds")
            continue

        validate_labels(label_arrays, lr)
        labels_by_lr[lr] = label_arrays

    if not labels_by_lr:
        raise RuntimeError("No valid LRs with multiple seeds found.")

    print("Loaded validation labels:")
    for lr, arrays in labels_by_lr.items():
        print(f"  LR={lr}: {len(arrays)} seeds, {len(arrays[0])} samples")

    ari_simmaries = {}

    for lr, label_arrays in labels_by_lr.items():
        print(f"\nComputing ARI for LR={lr}")

        ari_matrix = compute_ari_matrix(label_arrays)
        summary = summarize_ari_matrix(ari_matrix)

        ari_simmaries[lr] = summary

        # save matrix for plots later
        npy_path = OUTPUT_DIR / f"ari_matrix_lr_{lr}.npy"
        np.save(npy_path, ari_matrix)

        print(
            f" mean ARI = {summary['mean_ari']:.4f},"
            f"std = {summary['std_ari']:.4f}"
        )

        ari_json_path = OUTPUT_DIR / "ari_summary.json"
        with open(ari_json_path, "w") as f:
            json.dump(ari_simmaries, f, indent=2)

    print("\nSaved ARI summary to:", ari_json_path)