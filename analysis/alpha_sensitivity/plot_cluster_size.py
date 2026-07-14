import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INPUT_PATH = Path(
    "analysis/alpha_sensitivity/aggregated/aggregated_cluster_sizes.json"
)

FIGURES_DIR = Path("analysis/alpha_sensitivity/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

with open(INPUT_PATH, "r") as f:
    aggregated = json.load(f)

RANK_NAMES = [
    "smallest",
    "second_smallest",
    "middle",
    "second_largest",
    "largest",
]

plt.figure(figsize=(7, 5))

alphas = sorted(float(a) for a in aggregated.keys())

for rank in RANK_NAMES:

    means = []
    stds = []

    for alpha in alphas:
        stats = aggregated[str(alpha)][rank]
        means.append(stats["mean"])
        stds.append(stats["std"])

    plt.errorbar(
        alphas,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
        label=rank.replace("_", " ").title(),
    )

plt.xlabel("Alpha")
plt.ylabel("Cluster size")
plt.title("Cluster size by rank across alpha values")
plt.xticks(alphas, [str(a) for a in alphas])

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

plt.tight_layout()

output_path = FIGURES_DIR / "cluster_sizes_vs_alpha.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved figure to: {output_path}")