import json
from pathlib import Path
import numpy as np


INPUT = Path(
    "analysis/alpha_sensitivity/aggregated/"
    "seed_reproducibility_diagnostic.json"
)


def summarize(values):
    values = np.array(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


with open(INPUT, "r") as f:
    results = json.load(f)


print("=" * 90)
print("CLUSTER-SIZE STABILITY DIAGNOSTIC — ALPHA EXPERIMENT")
print("=" * 90)

for alpha, data in results["final_dec"].items():

    cluster_sizes = data["cluster_sizes"]

    smallest = {}
    largest = {}

    for seed, sizes in cluster_sizes.items():
        values = list(sizes.values())

        smallest[int(seed)] = min(values)
        largest[int(seed)] = max(values)

    print(f"\n{'=' * 90}")
    print(f"ALPHA = {alpha}")
    print("=" * 90)

    print("\nPer-seed cluster-size extremes")
    print("-" * 60)

    for seed in sorted(smallest):
        print(
            f"Seed {seed}: "
            f"smallest = {smallest[seed]:,}, "
            f"largest = {largest[seed]:,}"
        )

    # All seeds
    print("\nAll seeds (0-4)")
    print("-" * 60)

    print(
        "Smallest cluster: "
        f"{summarize(list(smallest.values()))}"
    )

    print(
        "Largest cluster:   "
        f"{summarize(list(largest.values()))}"
    )

    # Seeds 1-4
    smallest_1_4 = [
        smallest[s] for s in smallest if s != 0
    ]
    largest_1_4 = [
        largest[s] for s in largest if s != 0
    ]

    print("\nSeeds 1-4 only")
    print("-" * 60)

    print(
        "Smallest cluster: "
        f"{summarize(smallest_1_4)}"
    )

    print(
        "Largest cluster:   "
        f"{summarize(largest_1_4)}"
    )

    # Seed 0 comparison
    print("\nSeed-0 effect")
    print("-" * 60)

    print(
        f"Seed 0 smallest: {smallest[0]:,} | "
        f"Seeds 1-4 mean: {np.mean(smallest_1_4):,.2f}"
    )

    print(
        f"Seed 0 largest:  {largest[0]:,} | "
        f"Seeds 1-4 mean: {np.mean(largest_1_4):,.2f}"
    )

print("\n" + "=" * 90)
print("DIAGNOSTIC COMPLETE")
print("=" * 90)