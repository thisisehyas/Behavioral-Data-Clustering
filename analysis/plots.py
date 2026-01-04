import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

AGG_METRICS_PATH = Path("analysis/aggregated/aggregated_lr_metrics.json")
FIGURES_DIR = Path("analysis/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

with open(AGG_METRICS_PATH, "r") as f:
    aggregated = json.load(f)

lrs = []
means = []
stds = []

for lr_str, metrics in aggregated.items():
    lrs.append(float(lr_str))
    means.append(metrics["silhouette"]["mean"])
    stds.append(metrics["silhouette"]["std"])

# sort by LR
order = np.argsort(lrs)
lrs = np.array(lrs)[order]
means = np.array(means)[order]
stds = np.array(stds)[order]

plt.figure(figsize=(6, 4))

plt.errorbar(
    lrs,
    means,
    yerr=stds,
    fmt='o-',
    capsize=4
)

plt.xscale('log')
plt.xlabel("Learning rate (log scale)")
plt.ylabel("Silhouette Score")
plt.title("DEC performance vs learning rate")

plt.grid(True, which="both", linestyle='--', alpha=0.4)

output_path = FIGURES_DIR / "silhouette_vs_lr.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved figure to: {output_path}")