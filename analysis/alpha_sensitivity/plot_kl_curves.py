from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = Path("experiments/alpha_sensitivity/runs")

FIGURES_DIR = Path("analysis/alpha_sensitivity/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(7, 5))

for alpha_dir in sorted(RUNS_DIR.glob("alpha_*")):

    alpha = alpha_dir.name.split("_")[1]

    seed_curves = []

    for seed_dir in alpha_dir.glob("seed_*"):
        val_kl_path = seed_dir / "val_kl.npy"

        if val_kl_path.exists():
            val_kl = np.load(val_kl_path)
            seed_curves.append(val_kl)

    if len(seed_curves) == 0:
        print(f"Skipping alpha={alpha}: no data")
        continue

    # mean across seeds
    mean_curve = np.mean(seed_curves, axis=0)
    iterations = np.arange(len(mean_curve))

    plt.plot(
        iterations,
        mean_curve,
        label=f"alpha={alpha}",
        linewidth=2,
    )

plt.xlabel("Iteration")
plt.ylabel("Validation KL divergence")
plt.title("DEC Training Dynamics Across alpha Values")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

output_path = FIGURES_DIR / "kl_curves_by_alpha.png"

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved figure to: {output_path}")
