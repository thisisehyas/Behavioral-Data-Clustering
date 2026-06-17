import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute clustering and DEC metrics for a single run"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="path to single LR/sweed run directory"
    )
    return parser.parse_args()

def load_run_artifacts(run_dir: Path) :
    artifacts = {}

    artifacts["Z_val"] = np.load(run_dir / "Z_val.npy")
    artifacts["val_labels"] = np.load(run_dir / "val_labels.npy")
    artifacts["q_val"] = np.load(run_dir / "q_val.npy")
    artifacts["train_kl"] = np.load(run_dir / "train_kl.npy")
    artifacts["val_kl"] = np.load(run_dir / "val_kl.npy")

    with open(run_dir / "meta.json", "r") as f:
        artifacts["meta"] = json.load(f)

    return artifacts

def compute_entropy(q: np.ndarray):
    """
    Compute per-sample entropy of soft assignments.
    """
    eps = 1e-10
    entropy = -np.sum(q * np.log(q + eps), axis=1)
    return entropy

def compute_clustering_metrics(Z, labels):
    metrics = {}

    metrics["silhouette"] = silhouette_score(Z, labels)
    metrics["calinski_harabasz"] = calinski_harabasz_score(Z, labels)
    metrics["davies_bouldin"] = davies_bouldin_score(Z, labels)

    return metrics

def compute_optimization_metrics(train_kl, val_kl):
    metrics = {}

    metrics["final_train_kl"] = float(train_kl[-1])
    metrics["final_val_kl"] = float(val_kl[-1])
    metrics["min_val_kl"] = float(np.min(val_kl))
    metrics["kl_gap"] = float(train_kl[-1] - val_kl[-1])

    return metrics

def compute_cluster_size_metrics(labels):
    unique, counts = np.unique(labels, return_counts=True)

    return {
        "cluster_sizes": {
            str(int(cluster_id)): int(size)
            for cluster_id, size in zip(unique, counts)
        }
    }

def save_run_metrics(run_dir: Path, metrics: dict):
    output_path = run_dir / "run_metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    artifacts = load_run_artifacts(run_dir)

    Z_val = artifacts["Z_val"]
    labels = artifacts["val_labels"]
    q_val = artifacts["q_val"]
    train_kl = artifacts["train_kl"]
    val_kl = artifacts["val_kl"]
    meta = artifacts["meta"]

    entropy = compute_entropy(q_val)

    metrics = {
        "alpha": meta["alpha"],
        "seed": meta["seed"],

        "clustering_metrics": compute_clustering_metrics(Z_val, labels),

        "cluster_size_metrics": compute_cluster_size_metrics(
        labels
        ), 

        "entropy_metrics": {
            "mean_entropy": float(np.mean(entropy)),
            "std_entropy": float(np.std(entropy)),
        },

        "optimization_metrics": compute_optimization_metrics(
            train_kl, val_kl
        ),
    }

    save_run_metrics(run_dir, metrics)

    print(f"Saved run_metrics.json to {run_dir}")

if __name__ == "__main__":
    main()