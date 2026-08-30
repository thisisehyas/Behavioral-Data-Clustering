"""
Seed reproducibility diagnostic for the DEC alpha-sensitivity experiment.

For each alpha, this diagnostic compares:
1. Initial K-Means clustering across seeds.
2. Final DEC clustering across seeds.

For both stages, pairwise ARI is reported for:
- all 10 seed pairs,
- seed 0 versus seeds 1-4,
- seeds 1-4 only.

Cluster sizes are also recorded.

The main purpose is to determine whether the unusual behavior of
seed 0 is already present at initialization and whether it persists
or changes after DEC training.
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
    "experiments/alpha_sensitivity/runs"
)

AE_CHECKPOINT_PATH = Path(
    "models/autoencoder_weights.pth"
)

RESULTS_PATH = Path(
    "analysis/alpha_sensitivity/aggregated/"
    "seed_reproducibility_diagnostic.json"
)


# EXPERIMENT SETTINGS

ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0]

SEEDS = [0, 1, 2, 3, 4]

N_CLUSTERS = 5
N_INIT = 20


# HELPERS

def cluster_sizes(labels):
    """Return cluster sizes as a dictionary."""
    unique, counts = np.unique(
        labels,
        return_counts=True,
    )

    return {
        int(cluster): int(count)
        for cluster, count in zip(unique, counts)
    }


def pairwise_ari(labels_by_seed):
    """Calculate ARI for every unique pair of seeds."""

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


def summarize(values):
    """Return mean and population standard deviation."""

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "values": [float(v) for v in values],
    }


def summarize_ari(pairs):
    """Summarize all pairwise ARI values."""

    values = [
        pair["ari"]
        for pair in pairs
    ]

    return summarize(values)


def summarize_seed0_vs_others(pairs):
    """Summarize the four pairs involving seed 0."""

    values = [
        pair["ari"]
        for pair in pairs
        if pair["seed_i"] == 0
        or pair["seed_j"] == 0
    ]

    return summarize(values)


def summarize_seeds1_to4(pairs):
    """Summarize the six pairs among seeds 1-4."""

    values = [
        pair["ari"]
        for pair in pairs
        if pair["seed_i"] != 0
        and pair["seed_j"] != 0
    ]

    return summarize(values)

def summarize_per_seed_ari(pairs):
    """Calculate each seed's mean ARI against the other seeds."""

    per_seed = {}

    for seed in SEEDS:
        values = [
            pair["ari"]
            for pair in pairs
            if pair["seed_i"] == seed
            or pair["seed_j"] == seed
        ]

        per_seed[str(seed)] = summarize(values)

    return per_seed

def summarize_stage(labels_by_seed):
    """Calculate all ARI summaries for one clustering stage."""

    pairs = pairwise_ari(labels_by_seed)

    return {
        "pairwise_ari": pairs,
        "all_pairs": summarize_ari(pairs),
        "seed0_vs_seeds1_to4": summarize_seed0_vs_others(pairs),
        "seeds1_to4": summarize_seeds1_to4(pairs),
        "per_seed": summarize_per_seed_ari(pairs),
    }


# MAIN

def main():

    print("=" * 90)
    print("SEED REPRODUCIBILITY DIAGNOSTIC — ALPHA EXPERIMENT")
    print("=" * 90)

    # LOAD DATA

    print("\nLoading standardized data...")

    X = pd.read_csv(X_PATH).values.astype(np.float32)

    train_idx = np.load(TRAIN_IDX_PATH)

    X_train = X[train_idx]

    print(f"X shape: {X.shape}")
    print(f"Training samples: {len(train_idx)}")

    # LOAD AUTOENCODER

    print("\nLoading frozen autoencoder...")

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

    # COMPUTE INITIAL EMBEDDINGS

    print("Computing frozen training embeddings...")

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

    # RESULTS

    results = {
        "experiment": "alpha_sensitivity",
        "alphas": ALPHAS,
        "seeds": SEEDS,
        "n_clusters": N_CLUSTERS,
        "kmeans_n_init": N_INIT,
        "initial_kmeans": {},
        "final_dec": {},
    }

    # INITIAL K-MEANS

    print("\n")
    print("=" * 90)
    print("INITIAL K-MEANS")
    print("=" * 90)

    initial_labels_by_seed = {}
    initial_sizes_by_seed = {}

    for seed in SEEDS:

        kmeans = KMeans(
            n_clusters=N_CLUSTERS,
            random_state=seed,
            n_init=N_INIT,
        )

        kmeans.fit(Z_train)

        labels = kmeans.labels_

        initial_labels_by_seed[seed] = labels
        initial_sizes_by_seed[seed] = cluster_sizes(labels)

        print(f"\nSeed {seed}")
        print(
            f"Cluster sizes: "
            f"{initial_sizes_by_seed[seed]}"
        )

    initial_summary = summarize_stage(
        initial_labels_by_seed
    )

    print("\nInitial K-Means ARI")
    print("-" * 60)

    print(
        "Seed 0 vs seeds 1-4: "
        f"{initial_summary['seed0_vs_seeds1_to4']['mean']:.6f} "
        f"± "
        f"{initial_summary['seed0_vs_seeds1_to4']['std']:.6f}"
    )

    print(
        "Seeds 1-4 only: "
        f"{initial_summary['seeds1_to4']['mean']:.6f} "
        f"± "
        f"{initial_summary['seeds1_to4']['std']:.6f}"
    )

    print("\nPer-seed mean ARI")
    print("-" * 60)

    for seed, summary in initial_summary["per_seed"].items():
        print(
            f"Seed {seed}: "
            f"{summary['mean']:.6f} "
            f"± {summary['std']:.6f}"
        )

    print(
        "All pairs: "
        f"{initial_summary['all_pairs']['mean']:.6f} "
        f"± "
        f"{initial_summary['all_pairs']['std']:.6f}"
    )

    results["initial_kmeans"] = {
        "ari": initial_summary,
        "cluster_sizes": {
            str(seed): sizes
            for seed, sizes
            in initial_sizes_by_seed.items()
        },
    }

    # FINAL DEC — FOR EACH ALPHA

    for alpha in ALPHAS:

        alpha_key = f"{alpha:.1f}"

        print("\n\n")
        print("=" * 90)
        print(f"ALPHA = {alpha_key}")
        print("=" * 90)

        alpha_dir = (
            RUNS_ROOT / f"alpha_{alpha_key}"
        )

        if not alpha_dir.exists():
            raise FileNotFoundError(
                f"Missing alpha directory: {alpha_dir}"
            )

        final_labels_by_seed = {}
        final_sizes_by_seed = {}

        # LOAD FINAL LABELS

        for seed in SEEDS:

            seed_dir = (
                alpha_dir / f"seed_{seed}"
            )

            labels_path = (
                seed_dir / "val_labels.npy"
            )

            if not labels_path.exists():
                raise FileNotFoundError(
                    f"Missing labels: {labels_path}"
                )

            labels = np.load(labels_path)

            final_labels_by_seed[seed] = labels
            final_sizes_by_seed[seed] = cluster_sizes(labels)

            print(f"\nSeed {seed}")
            print(
                f"Final cluster sizes: "
                f"{final_sizes_by_seed[seed]}"
            )

        # FINAL ARI

        final_summary = summarize_stage(
            final_labels_by_seed
        )

        print("\nFinal DEC ARI")
        print("-" * 60)

        print(
            "Seed 0 vs seeds 1-4: "
            f"{final_summary['seed0_vs_seeds1_to4']['mean']:.6f} "
            f"± "
            f"{final_summary['seed0_vs_seeds1_to4']['std']:.6f}"
        )

        print(
            "Seeds 1-4 only: "
            f"{final_summary['seeds1_to4']['mean']:.6f} "
            f"± "
            f"{final_summary['seeds1_to4']['std']:.6f}"
        )

        print(
            "All 10 pairs: "
            f"{final_summary['all_pairs']['mean']:.6f} "
            f"± "
            f"{final_summary['all_pairs']['std']:.6f}"
        )

        # STORE

        results["final_dec"][alpha_key] = {
            "ari": final_summary,
            "cluster_sizes": {
                str(seed): sizes
                for seed, sizes
                in final_sizes_by_seed.items()
            },
        }

    # SAVE

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

    print(
        f"\nResults saved to:\n"
        f"  {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()