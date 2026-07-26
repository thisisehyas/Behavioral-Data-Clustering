from pathlib import Path
import numpy as np

RUNS_DIR = Path("experiments/alpha_sensitivity/runs")
ALPHA = "alpha_1.0"
SEEDS = [0, 1, 2]

def load_labels(alpha: str, seed: int) -> np.ndarray:
    """
    Load validation labels for one alpha/seed pair.
    """
    path = RUNS_DIR / alpha / f"seed_{seed}" / "val_labels.npy"
    return np.load(path)

def find_smallest_cluster(labels: np.ndarray):
    """
    Find the smallest cluster and return:
    - cluster id
    - indices of its members (within the validation set)
    """

    cluster_ids, counts = np.unique(labels, return_counts=True)

    smallest_cluster = cluster_ids[np.argmin(counts)]

    members = np.where(labels == smallest_cluster)[0]

    return smallest_cluster, members

def jaccard_overlap(members_a: np.ndarray, members_b: np.ndarray):
    """
    Compute the Jaccard overlap between two sets of samples.

    J(A, B) = |A ∩ B| / |A ∪ B|
    """

    set_a = set(members_a)
    set_b = set(members_b)

    intersection = set_a & set_b
    union = set_a | set_b

    score = len(intersection) / len(union)

    return score, len(intersection), len(union)

def track_cluster(members, labels):
    """
    Count where a set of samples appears in another clustering.

    Parameters
    ----------
    members : np.ndarray
        Sample indices to track.

    labels : np.ndarray
        Cluster labels from another seed.
    
    Returns
    -------
    dict
        cluster_id -> number of tracked samples
    """
    tracked_labels = labels[members]

    unique, counts = np.unique(tracked_labels, return_counts=True)

    return dict(zip(unique, counts))

def main():

    print(f"\nAlpha = {ALPHA}\n")

    smallest_clusters = {}
    all_labels = {}

    for seed in SEEDS:
        labels = load_labels(ALPHA, seed)
        all_labels[seed] = labels
        cluster_id, members = find_smallest_cluster(labels)
        smallest_clusters[seed] = members

        print(f"Seed {seed}")
        print(f"Smallest cluster id: {cluster_id}")
        print(f"Smallest cluster size: {len(members)}")
        print("-" * 30)

    print("\nSmallest Cluster Overlap")
    print("=" * 40)

    seed_pairs =  [(0, 1), (0, 2), (1, 2)]

    for seed_a, seed_b in seed_pairs:

        score, interseciton, union = jaccard_overlap(
            smallest_clusters[seed_a], 
            smallest_clusters[seed_b],
        )

        print(f"\nSeed {seed_a} vs Seed {seed_b}")
        print(f"Jaccard overlap: {score:.4f}")
        print(f"Intersection: {interseciton}")
        print(f"Union: {union}")

    print("\n Tracking Seed 1 Smallest Cluster into Seed 0")
    print("=" * 40)

    tracked = track_cluster(
        smallest_clusters[1], 
        all_labels[0]
    )

    total = len(smallest_clusters[1])

    for cluster_id, count in sorted(tracked.items()):
        print(
            f"Seed 0 Cluster {cluster_id}: "
            f"{count} samples ({100 * count / total:.1f}%)"
        )

if __name__ == "__main__":
    main()