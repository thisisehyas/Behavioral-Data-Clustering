import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score


def discover_parameter_dirs(
    runs_root: Path,
    prefix: str,
):
    """
    Discover all hyperparameter directories.

    Example:
        alpha_0.5 -> {"0.5": Path(...)}

        lr_0.001 -> {"0.001": Path(...)}
    """

    parameter_dirs = {}

    for parameter_dir in sorted(runs_root.iterdir()):

        if not parameter_dir.is_dir():
            continue

        if not parameter_dir.name.startswith(prefix):
            continue

        parameter_str = parameter_dir.name.replace(prefix, "")

        try:
            parameter_value = str(float(parameter_str))
        except ValueError:
            continue

        parameter_dirs[parameter_value] = parameter_dir

    return parameter_dirs


def load_val_labels_for_parameter(parameter_dir: Path):
    """
    Load validation labels from all seed directories.
    """

    labels = []

    for seed_dir in sorted(parameter_dir.iterdir()):

        if not seed_dir.is_dir():
            continue

        if not seed_dir.name.startswith("seed_"):
            continue

        label_path = seed_dir / "val_labels.npy"

        if not label_path.exists():
            print(f"Missing val_labels.npy in {seed_dir}")
            continue

        labels.append(np.load(label_path))

    return labels


def validate_labels(
    label_arrays,
    parameter_value,
):
    """
    Ensure all label arrays have equal length.
    """

    if len(label_arrays) < 2:
        raise ValueError(
            f"Parameter={parameter_value}: fewer than 2 seeds found."
        )

    lengths = {len(arr) for arr in label_arrays}

    if len(lengths) != 1:
        raise ValueError(
            f"Parameter={parameter_value}: inconsistent label lengths: {lengths}"
        )


def compute_ari_matrix(label_arrays):
    """
    Compute pairwise ARI matrix.
    """

    n = len(label_arrays)

    ari_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):

            ari_matrix[i, j] = adjusted_rand_score(
                label_arrays[i],
                label_arrays[j],
            )

    return ari_matrix


def summarize_ari_matrix(ari_matrix):
    """
    Compute mean/std excluding diagonal.
    """

    n = ari_matrix.shape[0]

    upper = []

    for i in range(n):
        for j in range(i + 1, n):
            upper.append(ari_matrix[i, j])

    upper = np.array(upper)

    return {
        "mean_ari": float(np.mean(upper)),
        "std_ari": float(np.std(upper)),
    }


def compute_ari(
    runs_root: Path,
    output_dir: Path,
    prefix: str,
):
    """
    Compute ARI stability for every hyperparameter value.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    parameter_dirs = discover_parameter_dirs(
        runs_root,
        prefix,
    )

    labels_by_parameter = {}

    for parameter_value, parameter_dir in parameter_dirs.items():

        label_arrays = load_val_labels_for_parameter(parameter_dir)

        if len(label_arrays) < 2:
            print(
                f"Skipping parameter={parameter_value}: not enough seeds."
            )
            continue

        validate_labels(
            label_arrays,
            parameter_value,
        )

        labels_by_parameter[parameter_value] = label_arrays

    if not labels_by_parameter:
        raise RuntimeError(
            "No valid parameter values with multiple seeds found."
        )

    print("Loaded validation labels:")

    for parameter_value, arrays in labels_by_parameter.items():

        print(
            f"  Parameter={parameter_value}: "
            f"{len(arrays)} seeds, "
            f"{len(arrays[0])} samples"
        )

    ari_summaries = {}

    for parameter_value, label_arrays in labels_by_parameter.items():

        print(f"\nComputing ARI for {parameter_value}")

        ari_matrix = compute_ari_matrix(label_arrays)

        summary = summarize_ari_matrix(ari_matrix)

        ari_summaries[parameter_value] = summary

        np.save(
            output_dir / f"ari_matrix_{parameter_value}.npy",
            ari_matrix,
        )

        print(
            f"Mean ARI = {summary['mean_ari']:.4f}, "
            f"Std = {summary['std_ari']:.4f}"
        )

    summary_path = output_dir / "ari_summary.json"

    with open(summary_path, "w") as f:
        json.dump(ari_summaries, f, indent=2)

    print(f"\nSaved ARI summary to: {summary_path}")