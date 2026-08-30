from pathlib import Path
import json

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0]
SEEDS = [0, 1, 2, 3, 4]

RUNS_ROOT = Path("experiments/alpha_sensitivity/runs")
OUTPUT_PATH = Path("analysis/aggregated/internal_clustering_quality.json")


def load_run(alpha, seed):
    run_dir = RUNS_ROOT / f"alpha_{alpha}" / f"seed_{seed}"

    labels_path = run_dir / "val_labels.npy"
    embeddings_path = run_dir / "Z_val.npy"

    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels: {labels_path}")

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings: {embeddings_path}")

    labels = np.load(labels_path)
    embeddings = np.load(embeddings_path)

    if len(labels) != len(embeddings):
        raise ValueError(
            f"Length mismatch for alpha={alpha}, seed={seed}: "
            f"{len(embeddings)} embeddings vs {len(labels)} labels"
        )

    return embeddings, labels


def calculate_metrics(embeddings, labels):
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        raise ValueError("Clustering must contain at least two clusters.")

    silhouette = silhouette_score(embeddings, labels)
    calinski = calinski_harabasz_score(embeddings, labels)
    davies_bouldin = davies_bouldin_score(embeddings, labels)

    return {
        "silhouette": float(silhouette),
        "calinski_harabasz": float(calinski),
        "davies_bouldin": float(davies_bouldin),
    }


def summarize(records, seeds):
    summary = {}

    for alpha in ALPHAS:
        alpha_records = [
            r for r in records
            if r["alpha"] == alpha and r["seed"] in seeds
        ]

        summary[str(alpha)] = {}

        for metric in [
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
        ]:
            values = np.array(
                [r[metric] for r in alpha_records],
                dtype=float,
            )

            summary[str(alpha)][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    return summary


def calculate_paired_differences(records):
    lookup = {
        (r["alpha"], r["seed"]): r
        for r in records
    }

    differences = {}

    for seed in SEEDS:
        differences[str(seed)] = {}

        for alpha in ALPHAS:
            if alpha == 1.0:
                continue

            differences[str(seed)][str(alpha)] = {}

            for metric in [
                "silhouette",
                "calinski_harabasz",
                "davies_bouldin",
            ]:
                current = lookup[(alpha, seed)][metric]
                baseline = lookup[(1.0, seed)][metric]

                differences[str(seed)][str(alpha)][metric] = (
                    current - baseline
                )

    return differences


def calculate_direction_consistency(records):
    lookup = {
        (r["alpha"], r["seed"]): r
        for r in records
    }

    consistency = {}

    for metric in [
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
    ]:
        consistency[metric] = {}

        for alpha in ALPHAS:
            if alpha == 1.0:
                continue

            differences = []

            for seed in SEEDS:
                diff = (
                    lookup[(alpha, seed)][metric]
                    - lookup[(1.0, seed)][metric]
                )
                differences.append(diff)

            if metric == "davies_bouldin":
                # Lower is better.
                improved = sum(d < 0 for d in differences)
                worsened = sum(d > 0 for d in differences)
            else:
                # Higher is better.
                improved = sum(d > 0 for d in differences)
                worsened = sum(d < 0 for d in differences)

            consistency[metric][str(alpha)] = {
                "improved_seeds": improved,
                "worsened_seeds": worsened,
                "unchanged_seeds": len(differences) - improved - worsened,
                "direction": (
                    "consistent_improvement"
                    if improved == len(SEEDS)
                    else "consistent_worsening"
                    if worsened == len(SEEDS)
                    else "mixed"
                ),
            }

    return consistency


def print_report(records, summary_all, summary_1_to_4, consistency):
    print("\n" + "=" * 90)
    print("INTERNAL CLUSTERING QUALITY DIAGNOSTIC — ALPHA EXPERIMENT")
    print("=" * 90)

    print("\nPer-seed metrics")
    print("-" * 90)

    for alpha in ALPHAS:
        print(f"\nALPHA = {alpha}")

        for record in records:
            if record["alpha"] == alpha:
                print(
                    f"Seed {record['seed']}: "
                    f"Silhouette = {record['silhouette']:.6f}, "
                    f"CH = {record['calinski_harabasz']:.2f}, "
                    f"DB = {record['davies_bouldin']:.6f}"
                )

    print("\n" + "=" * 90)
    print("MEAN ± STD — ALL SEEDS")
    print("=" * 90)

    for alpha in ALPHAS:
        s = summary_all[str(alpha)]
        print(
            f"α={alpha}: "
            f"Silhouette={s['silhouette']['mean']:.6f} ± "
            f"{s['silhouette']['std']:.6f} | "
            f"CH={s['calinski_harabasz']['mean']:.2f} ± "
            f"{s['calinski_harabasz']['std']:.2f} | "
            f"DB={s['davies_bouldin']['mean']:.6f} ± "
            f"{s['davies_bouldin']['std']:.6f}"
        )

    print("\n" + "=" * 90)
    print("MEAN ± STD — SEEDS 1-4")
    print("=" * 90)

    for alpha in ALPHAS:
        s = summary_1_to_4[str(alpha)]
        print(
            f"α={alpha}: "
            f"Silhouette={s['silhouette']['mean']:.6f} ± "
            f"{s['silhouette']['std']:.6f} | "
            f"CH={s['calinski_harabasz']['mean']:.2f} ± "
            f"{s['calinski_harabasz']['std']:.2f} | "
            f"DB={s['davies_bouldin']['mean']:.6f} ± "
            f"{s['davies_bouldin']['std']:.6f}"
        )

    print("\n" + "=" * 90)
    print("DIRECTION CONSISTENCY VS α=1.0")
    print("=" * 90)

    for metric, values in consistency.items():
        print(f"\n{metric}")

        for alpha, result in values.items():
            print(
                f"  α={alpha}: "
                f"{result['direction']} "
                f"(improved={result['improved_seeds']}, "
                f"worsened={result['worsened_seeds']}, "
                f"unchanged={result['unchanged_seeds']})"
            )

    print("\n" + "=" * 90)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 90)


def main():
    records = []

    for alpha in ALPHAS:
        for seed in SEEDS:
            embeddings, labels = load_run(alpha, seed)
            metrics = calculate_metrics(embeddings, labels)

            records.append({
                "alpha": alpha,
                "seed": seed,
                **metrics,
            })

    summary_all = summarize(records, SEEDS)
    summary_1_to_4 = summarize(records, [1, 2, 3, 4])
    paired_differences = calculate_paired_differences(records)
    consistency = calculate_direction_consistency(records)

    output = {
        "alphas": ALPHAS,
        "seeds": SEEDS,
        "metrics": {
            "silhouette": "higher_is_better",
            "calinski_harabasz": "higher_is_better",
            "davies_bouldin": "lower_is_better",
        },
        "per_seed": records,
        "summary_all_seeds": summary_all,
        "summary_seeds_1_to_4": summary_1_to_4,
        "paired_differences_vs_alpha_1.0": paired_differences,
        "direction_consistency_vs_alpha_1.0": consistency,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as f:
        json.dump(output, f, indent=2)

    print_report(
        records,
        summary_all,
        summary_1_to_4,
        consistency,
    )

    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()