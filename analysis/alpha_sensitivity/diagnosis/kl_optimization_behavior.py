from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Configuration

RUNS_ROOT = Path("experiments/alpha_sensitivity/runs")
OUTPUT_DIR = Path(
    "analysis/alpha_sensitivity/aggregated/"
)
OUTPUT_DIR_PLOT = Path("analysis/alpha_sensitivity/figures/")

ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0]
SEEDS = [0, 1, 2, 3, 4]

# DEC target-distribution update points.
# These are the points we specifically want to inspect.
UPDATE_EPOCHS = [20, 40]

# Previously generated ARI matrices.
ARI_DIR = Path("analysis/alpha_sensitivity/aggregated/")

# Utilities


def alpha_dir_name(alpha: float) -> str:
    return f"alpha_{alpha:.1f}"


def seed_dir_name(seed: int) -> str:
    return f"seed_{seed}"


def load_json(path: Path):
    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


def load_npy(path: Path):
    if not path.exists():
        return None

    return np.load(path, allow_pickle=True)


# KL loading


def extract_kl_values(array):
    """
    Convert a saved KL array into a clean 1-D float array.

    Supported common forms:
        [value, value, ...]
        [[epoch, value], ...]
        [[value], ...]
        scalar-like arrays

    The function intentionally avoids making assumptions about
    the exact storage format.
    """

    arr = np.asarray(array)

    if arr.ndim == 0:
        return np.asarray([float(arr)])

    if arr.ndim == 1:
        return arr.astype(float)

    if arr.ndim == 2:

        if arr.shape[1] == 1:
            return arr[:, 0].astype(float)

        if arr.shape[1] == 2:
            first = arr[:, 0]
            second = arr[:, 1]

            if np.all(np.isfinite(first)):
                if np.allclose(first, np.round(first)):
                    return second.astype(float)

            return arr[:, -1].astype(float)

        return arr[:, -1].astype(float)

    # Unexpected shape.
    return arr.reshape(-1).astype(float)


def load_kl_curve(seed_dir: Path, name: str):
    path = seed_dir / name

    raw = load_npy(path)

    if raw is None:
        return None

    values = extract_kl_values(raw)

    values = values[np.isfinite(values)]

    return values


# Epoch extraction


def extract_epochs(run_metrics, n_values: int):
    """
    Try to recover epoch indices from run_metrics.json.

    If no reliable epoch information is available, return
    0, 1, ..., n-1.
    """

    if run_metrics is None:
        return np.arange(n_values)

    candidate_keys = [
        "epochs",
        "epoch",
        "train_epochs",
        "val_epochs",
    ]

    for key in candidate_keys:

        value = run_metrics.get(key)

        if value is None:
            continue

        try:
            arr = np.asarray(value, dtype=float)

            if arr.ndim == 1 and len(arr) == n_values:
                return arr
        except (TypeError, ValueError):
            pass

    return np.arange(n_values)


# KL curve plotting


def plot_all_kl_curves(results):
    """
    Create one figure per alpha containing all seed curves.

    Separate figures are used so that alpha=5 can be inspected
    without being visually compressed by the other conditions.
    """

    for alpha in ALPHAS:

        alpha_results = results.get(alpha, {})

        if not alpha_results:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        for seed, data in alpha_results.items():

            epochs = data["epochs"]
            train_kl = data["train_kl"]
            val_kl = data["val_kl"]

            if train_kl is not None:
                ax.plot(
                    epochs[: len(train_kl)],
                    train_kl,
                    label=f"Seed {seed} — train",
                    alpha=0.75,
                )

            if val_kl is not None:
                ax.plot(
                    epochs[: len(val_kl)],
                    val_kl,
                    linestyle="--",
                    label=f"Seed {seed} — val",
                    alpha=0.75,
                )

        for epoch in UPDATE_EPOCHS:
            ax.axvline(
                epoch,
                linestyle=":",
                linewidth=1.5,
                label=f"Target update: {epoch}",
            )

        ax.set_title(f"KL Optimization Curves — α={alpha}")
        ax.set_xlabel("Epoch / recorded step")
        ax.set_ylabel("KL divergence")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

        fig.tight_layout()

        output_path = OUTPUT_DIR_PLOT / f"kl_curves_alpha_{alpha:g}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


# KL jump analysis


def nearest_index(epochs, target_epoch):
    epochs = np.asarray(epochs)

    return int(np.argmin(np.abs(epochs - target_epoch)))


def compute_jump(values, epochs, update_epoch):
    """
    Estimate the KL change immediately around a target-distribution
    update.

    jump = value after update - value before update

    We use the nearest recorded points before and after the update.
    """

    if values is None or len(values) < 3:
        return None

    epochs = np.asarray(epochs)
    values = np.asarray(values)

    idx = nearest_index(epochs, update_epoch)

    if idx <= 0 or idx >= len(values) - 1:
        return None

    before = float(values[idx - 1])
    at = float(values[idx])
    after = float(values[idx + 1])

    return {
        "epoch": float(epochs[idx]),
        "before": before,
        "at_update": at,
        "after": after,
        "jump_from_before": at - before,
        "jump_after_update": after - at,
        "absolute_jump": abs(at - before),
    }


# KL statistics


def summarize_kl(values):
    if values is None or len(values) == 0:
        return None

    values = np.asarray(values, dtype=float)

    minimum_idx = int(np.argmin(values))
    final_idx = len(values) - 1

    return {
        "final": float(values[final_idx]),
        "minimum": float(values[minimum_idx]),
        "minimum_index": minimum_idx,
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


# ARI loading


def find_ari_matrix(alpha: float):
    """
    Search for the previously generated ARI matrix.

    Supports the filename produced by the earlier ARI script.
    """

    candidates = [
        ARI_DIR / f"ari_matrix_alpha_{alpha:g}.npy",
        ARI_DIR / f"ari_matrix_alpha_{alpha}.npy",
        ARI_DIR / f"ari_matrix_alpha_{str(alpha)}.npy",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Fallback recursive search.
    matches = list(
        ARI_DIR.rglob(f"*ari_matrix*{alpha:g}*.npy")
    )

    if matches:
        return matches[0]

    return None


def compute_seed_mean_ari(ari_matrix, seed_index):
    """
    Mean ARI of one seed against all other seeds.
    """

    if ari_matrix is None:
        return None

    matrix = np.asarray(ari_matrix, dtype=float)

    if seed_index >= matrix.shape[0]:
        return None

    values = np.delete(matrix[seed_index], seed_index)

    if len(values) == 0:
        return None

    return float(np.mean(values))


def load_ari_by_alpha():
    ari_by_alpha = {}

    for alpha in ALPHAS:

        path = find_ari_matrix(alpha)

        if path is None:
            print(
                f"WARNING: No ARI matrix found for alpha={alpha}"
            )
            continue

        ari_by_alpha[alpha] = np.load(path)

        print(
            f"Loaded ARI matrix for alpha={alpha}: {path}"
        )

    return ari_by_alpha


# Internal clustering metric extraction


def recursive_find_metric(obj, names):
    """
    Recursively search JSON objects for one of the requested
    metric names.

    This is deliberately permissive because run_metrics.json
    structure may differ between experiments.
    """

    if isinstance(obj, dict):

        for key, value in obj.items():

            normalized = key.lower().replace("-", "_").replace(" ", "_")

            if normalized in names:

                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

            found = recursive_find_metric(value, names)

            if found is not None:
                return found

    elif isinstance(obj, list):

        for value in obj:

            found = recursive_find_metric(value, names)

            if found is not None:
                return found

    return None


def extract_internal_metrics(run_metrics):

    if run_metrics is None:
        return {
            "silhouette": None,
            "calinski_harabasz": None,
            "davies_bouldin": None,
        }

    silhouette_names = {
        "silhouette",
        "silhouette_score",
        "val_silhouette",
        "final_silhouette",
    }

    ch_names = {
        "calinski_harabasz",
        "calinski_harabasz_score",
        "ch",
        "ch_score",
        "val_ch",
        "final_ch",
    }

    db_names = {
        "davies_bouldin",
        "davies_bouldin_score",
        "db",
        "db_score",
        "val_db",
        "final_db",
    }

    return {
        "silhouette": recursive_find_metric(
            run_metrics,
            silhouette_names,
        ),
        "calinski_harabasz": recursive_find_metric(
            run_metrics,
            ch_names,
        ),
        "davies_bouldin": recursive_find_metric(
            run_metrics,
            db_names,
        ),
    }


# Main data collection


def collect_results():

    results = {}

    ari_by_alpha = load_ari_by_alpha()

    for alpha in ALPHAS:

        results[alpha] = {}

        alpha_dir = RUNS_ROOT / alpha_dir_name(alpha)

        if not alpha_dir.exists():
            print(
                f"WARNING: Missing directory: {alpha_dir}"
            )
            continue

        for seed in SEEDS:

            seed_dir = alpha_dir / seed_dir_name(seed)

            if not seed_dir.exists():
                print(
                    f"WARNING: Missing directory: {seed_dir}"
                )
                continue

            train_kl = load_kl_curve(
                seed_dir,
                "train_kl.npy",
            )

            val_kl = load_kl_curve(
                seed_dir,
                "val_kl.npy",
            )

            run_metrics = load_json(
                seed_dir / "run_metrics.json"
            )

            # Prefer the longest available curve for epoch indexing.
            reference_curve = (
                train_kl
                if train_kl is not None
                else val_kl
            )

            if reference_curve is None:
                print(
                    f"WARNING: No KL curves for "
                    f"alpha={alpha}, seed={seed}"
                )
                continue

            epochs = extract_epochs(
                run_metrics,
                len(reference_curve),
            )

            ari_matrix = ari_by_alpha.get(alpha)

            seed_mean_ari = compute_seed_mean_ari(
                ari_matrix,
                seed,
            )

            internal_metrics = extract_internal_metrics(
                run_metrics
            )

            results[alpha][seed] = {
                "epochs": epochs,
                "train_kl": train_kl,
                "val_kl": val_kl,
                "seed_mean_ari": seed_mean_ari,
                "internal_metrics": internal_metrics,
                "train_summary": summarize_kl(train_kl),
                "val_summary": summarize_kl(val_kl),
                "updates": {
                    "train": {
                        str(epoch): compute_jump(
                            train_kl,
                            epochs[: len(train_kl)]
                            if train_kl is not None
                            else epochs,
                            epoch,
                        )
                        for epoch in UPDATE_EPOCHS
                    },
                    "val": {
                        str(epoch): compute_jump(
                            val_kl,
                            epochs[: len(val_kl)]
                            if val_kl is not None
                            else epochs,
                            epoch,
                        )
                        for epoch in UPDATE_EPOCHS
                    },
                },
            }

    return results


# Alpha-level summary


def build_alpha_summary(results):

    summary = {}

    for alpha, seed_results in results.items():

        if not seed_results:
            continue

        entry = {
            "seeds": {},
            "across_seed": {},
        }

        final_train = []
        minimum_train = []

        final_val = []
        minimum_val = []

        ari_values = []

        for seed, data in seed_results.items():

            entry["seeds"][str(seed)] = {
                "train_kl": data["train_summary"],
                "val_kl": data["val_summary"],
                "seed_mean_ari": data["seed_mean_ari"],
                "internal_metrics": data[
                    "internal_metrics"
                ],
                "updates": data["updates"],
            }

            if data["train_summary"] is not None:
                final_train.append(
                    data["train_summary"]["final"]
                )
                minimum_train.append(
                    data["train_summary"]["minimum"]
                )

            if data["val_summary"] is not None:
                final_val.append(
                    data["val_summary"]["final"]
                )
                minimum_val.append(
                    data["val_summary"]["minimum"]
                )

            if data["seed_mean_ari"] is not None:
                ari_values.append(
                    data["seed_mean_ari"]
                )

        def mean_std(values):

            if not values:
                return {
                    "mean": None,
                    "std": None,
                }

            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

        entry["across_seed"] = {
            "final_train_kl": mean_std(final_train),
            "minimum_train_kl": mean_std(minimum_train),
            "final_val_kl": mean_std(final_val),
            "minimum_val_kl": mean_std(minimum_val),
            "seed_mean_ari": mean_std(ari_values),
        }

        summary[str(alpha)] = entry

    return summary


# α=5 focused diagnostic


def analyze_alpha_5(results):

    alpha = 5.0

    if alpha not in results:
        return None

    focused = {}

    for seed, data in results[alpha].items():

        focused[str(seed)] = {
            "seed_mean_ari": data["seed_mean_ari"],
            "train_summary": data["train_summary"],
            "val_summary": data["val_summary"],
            "updates": data["updates"],
        }

    return focused


# Main


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print(
        "KL OPTIMIZATION BEHAVIOR DIAGNOSTIC — "
        "ALPHA EXPERIMENT"
    )
    print("=" * 90)

    results = collect_results()

    # Plot KL curves

    plot_all_kl_curves(results)

    # Build summaries

    alpha_summary = build_alpha_summary(results)

    alpha_5 = analyze_alpha_5(results)

    # Print concise diagnostic summary

    print("\n" + "=" * 90)
    print("ALPHA-LEVEL KL SUMMARY")
    print("=" * 90)

    for alpha_str, data in alpha_summary.items():

        print(f"\nα={alpha_str}")

        print(
            "  Final train KL: "
            f"{data['across_seed']['final_train_kl']}"
        )

        print(
            "  Minimum train KL: "
            f"{data['across_seed']['minimum_train_kl']}"
        )

        print(
            "  Final validation KL: "
            f"{data['across_seed']['final_val_kl']}"
        )

        print(
            "  Minimum validation KL: "
            f"{data['across_seed']['minimum_val_kl']}"
        )

        print(
            "  Seed ARI: "
            f"{data['across_seed']['seed_mean_ari']}"
        )

    # Focus α=5

    print("\n" + "=" * 90)
    print("FOCUSED α=5 DIAGNOSTIC")
    print("=" * 90)

    if alpha_5 is not None:

        for seed, data in alpha_5.items():

            print(f"\nSeed {seed}")

            print(
                f"  Seed mean ARI: "
                f"{data['seed_mean_ari']}"
            )

            print(
                f"  Train KL: "
                f"{data['train_summary']}"
            )

            print(
                f"  Validation KL: "
                f"{data['val_summary']}"
            )

            for update_epoch, update_data in (
                data["updates"]["val"].items()
            ):

                print(
                    f"  Validation KL around "
                    f"update {update_epoch}: "
                    f"{update_data}"
                )

    # Save JSON

    output = {
        "alphas": ALPHAS,
        "seeds": SEEDS,
        "target_update_epochs": UPDATE_EPOCHS,
        "alpha_summary": alpha_summary,
        "alpha_5_focused": alpha_5,
    }

    output_path = (
        OUTPUT_DIR /
        "kl_optimization_behavior.json"
    )

    with open(output_path, "w") as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print("\n" + "=" * 90)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 90)

    print(
        f"Saved results to: {output_path}"
    )

    print(
        f"Saved KL plots to: {OUTPUT_DIR_PLOT}"
    )


if __name__ == "__main__":
    main()