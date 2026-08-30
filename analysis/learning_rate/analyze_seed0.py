"""
Seed-0 diagnostic for the DEC learning-rate experiment.

This analysis checks how consistent the clustering is across random seeds at two stages:

1. Initial K-Means clustering in the frozen autoencoder latent space.
2. Final DEC clustering on the validation set after training.

For each learning rate, pairwise ARI is reported for:
- all seed pairs,
- seed 0 versus seeds 1-4,
- seeds 1-4 only.

Initial K-Means cluster sizes and final DEC validation cluster sizes
are also recorded.

The main goal is to check whether the different behavior of seed 0
is already present at initialization or emerges during DEC training,
and whether the observed seed-specific behavior changes with the learning rate.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from src.dec.autoencoder import Autoencoder



# PATHS
X_PATH = Path("dataset/X_scaled.csv")

TRAIN_IDX_PATH = Path(
    "experiments/val_train_splits/train_idx.npy"
)

RUNS_ROOT = Path(
    "experiments/learning_rate/runs"
)

AE_CHECKPOINT_PATH = Path(
    "models/autoencoder_weights.pth"
)

RESULTS_PATH = Path(
    "analysis/learning_rate/aggregated/seed0_diagnostic.json"
)


# EXPERIMENT SETTINGS

LEARNING_RATES = [
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
]

SEEDS = [0, 1, 2, 3, 4]

N_CLUSTERS = 5
N_INIT = 20


# HELPERS

def cluster_sizes(labels):
    """Return cluster sizes as a dictionary."""
    unique, counts = np.unique(labels, return_counts=True)

    return {
        int(cluster): int(count)
        for cluster, count in zip(unique, counts)
    }


def pairwise_ari(labels_by_seed):
    """Calculate ARI for every unique pair of available seeds."""
    seeds = sorted(labels_by_seed.keys())

    pairs = []

    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            seed_i = seeds[i]
            seed_j = seeds[j]

            ari = adjusted_rand_score(
                labels_by_seed[seed_i],
                labels_by_seed[seed_j],
            )

            pairs.append(
                {
                    "seed_i": seed_i,
                    "seed_j": seed_j,
                    "ari": float(ari),
                }
            )

    return pairs


def summarize_ari(pairs):
    """Summarize ARI values using the population standard deviation."""
    values = np.array(
        [pair["ari"] for pair in pairs],
        dtype=np.float64,
    )

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
    }


def summarize_seed0_vs_others(pairs):
    """Summarize the four ARI pairs involving seed 0."""
    values = [
        pair["ari"]
        for pair in pairs
        if pair["seed_i"] == 0 or pair["seed_j"] == 0
    ]

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "values": values,
    }


def summarize_seeds1_to4(pairs):
    """Summarize the six ARI pairs among seeds 1–4."""
    values = [
        pair["ari"]
        for pair in pairs
        if pair["seed_i"] != 0 and pair["seed_j"] != 0
    ]

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "values": values,
    }


def summarize_stage(labels_by_seed):
    """Calculate pairwise ARI summaries for one clustering stage."""
    if len(labels_by_seed) < 2:
        return None

    pairs = pairwise_ari(labels_by_seed)

    return {
        "pairs": pairs,
        "all_pairs": summarize_ari(pairs),
        "seed0_vs_others": summarize_seed0_vs_others(pairs),
        "seeds1_to4": summarize_seeds1_to4(pairs),
    }


# MAIN

def main():

    print("=" * 90)
    print("SEED-0 INITIALIZATION DIAGNOSTIC — LEARNING RATE EXPERIMENT")
    print("=" * 90)

    # Load data

    if not X_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {X_PATH}")

    if not TRAIN_IDX_PATH.exists():
        raise FileNotFoundError(
            f"Missing train-index file: {TRAIN_IDX_PATH}"
        )

    if not AE_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing autoencoder checkpoint: {AE_CHECKPOINT_PATH}"
        )

    print("\nLoading X_scaled...")

    X = pd.read_csv(X_PATH).values.astype(np.float32)

    train_idx = np.load(TRAIN_IDX_PATH)

    X_train = X[train_idx]

    print(f"X shape: {X.shape}")
    print(f"Training samples: {len(train_idx)}")

    # Load frozen autoencoder

    print("\nLoading autoencoder...")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    autoencoder = Autoencoder()

    autoencoder.load_state_dict(
        torch.load(
            AE_CHECKPOINT_PATH,
            map_location=device,
        )
    )

    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    for param in autoencoder.encoder.parameters():
        param.requires_grad = False

    # Compute frozen training embeddings

    print("Computing frozen autoencoder embeddings...")

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        Z_train = (
            autoencoder.encoder(X_train_tensor)
            .cpu()
            .numpy()
        )

    print(f"Z_train shape: {Z_train.shape}")

    # Results container

    results = {
        "experiment": "learning_rate",
        "learning_rates": LEARNING_RATES,
        "seeds": SEEDS,
        "n_clusters": N_CLUSTERS,
        "kmeans_n_init": N_INIT,
        "initial_kmeans": {},
        "final_dec": {},
    }

    # EACH LEARNING RATE

    for learning_rate in LEARNING_RATES:

        lr_key = f"{learning_rate:.0e}"

        print("\n\n")
        print("=" * 90)
        print(f"LEARNING RATE = {lr_key}")
        print("=" * 90)

        parameter_dir = (
            RUNS_ROOT / f"lr_{learning_rate:.0e}"
        )

        if not parameter_dir.exists():
            raise FileNotFoundError(
                f"Missing learning-rate directory: {parameter_dir}"
            )

        initial_labels_by_seed = {}
        final_labels_by_seed = {}

        initial_sizes_by_seed = {}
        final_sizes_by_seed = {}

        # Each seed

        for seed in SEEDS:

            print("\n" + "-" * 80)
            print(f"SEED = {seed}")
            print("-" * 80)

            # Initial K-Means

            kmeans = KMeans(
                n_clusters=N_CLUSTERS,
                random_state=seed,
                n_init=N_INIT,
            )

            kmeans.fit(Z_train)

            initial_labels = kmeans.labels_

            initial_labels_by_seed[seed] = initial_labels

            initial_sizes = cluster_sizes(initial_labels)

            initial_sizes_by_seed[seed] = initial_sizes

            print("\nInitial K-Means cluster sizes:")
            print(f"  {initial_sizes}")

            # Final DEC labels

            seed_dir = parameter_dir / f"seed_{seed}"

            final_labels_path = seed_dir / "val_labels.npy"

            if not final_labels_path.exists():
                raise FileNotFoundError(
                    f"Missing final labels: {final_labels_path}"
                )

            final_labels = np.load(final_labels_path)

            final_labels_by_seed[seed] = final_labels

            final_sizes = cluster_sizes(final_labels)

            final_sizes_by_seed[seed] = final_sizes

            print("\nFinal DEC validation cluster sizes:")
            print(f"  {final_sizes}")

        # Initial K-Means ARI

        initial_summary = summarize_stage(
            initial_labels_by_seed
        )

        print("\n")
        print("-" * 80)
        print("INITIAL K-MEANS PAIRWISE ARI")
        print("-" * 80)

        for pair in initial_summary["pairs"]:
            print(
                f"  seed {pair['seed_i']} ↔ seed "
                f"{pair['seed_j']}: {pair['ari']:.6f}"
            )

        print(
            "\nSeed 0 vs seeds 1-4: "
            f"{initial_summary['seed0_vs_others']['mean']:.6f} "
            f"± {initial_summary['seed0_vs_others']['std']:.6f}"
        )

        print(
            "Seeds 1-4 only: "
            f"{initial_summary['seeds1_to4']['mean']:.6f} "
            f"± {initial_summary['seeds1_to4']['std']:.6f}"
        )

        # Final DEC ARI

        final_summary = summarize_stage(
            final_labels_by_seed
        )

        print("\n")
        print("-" * 80)
        print("FINAL DEC PAIRWISE ARI")
        print("-" * 80)

        for pair in final_summary["pairs"]:
            print(
                f"  seed {pair['seed_i']} ↔ seed "
                f"{pair['seed_j']}: {pair['ari']:.6f}"
            )

        print(
            "\nSeed 0 vs seeds 1-4: "
            f"{final_summary['seed0_vs_others']['mean']:.6f} "
            f"± {final_summary['seed0_vs_others']['std']:.6f}"
        )

        print(
            "Seeds 1-4 only: "
            f"{final_summary['seeds1_to4']['mean']:.6f} "
            f"± {final_summary['seeds1_to4']['std']:.6f}"
        )

        # Store results

        results["initial_kmeans"][lr_key] = {
            "ari": initial_summary,
            "cluster_sizes": {
                str(seed): sizes
                for seed, sizes in initial_sizes_by_seed.items()
            },
        }

        results["final_dec"][lr_key] = {
            "ari": final_summary,
            "cluster_sizes": {
                str(seed): sizes
                for seed, sizes in final_sizes_by_seed.items()
            },
        }

    # SAVE RESULTS

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(RESULTS_PATH, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print("\n\n")
    print("=" * 90)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 90)
    print(f"\nResults saved to:")
    print(f"  {RESULTS_PATH}")


if __name__ == "__main__":
    main()