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
        train_kl_path = seed_dir / "train_kl.npy"

        if train_kl_path.exists():
            train_kl = np.load(train_kl_path)
            seed_curves.append(train_kl)

    if len(seed_curves) == 0:
        print(f"Skipping alpha={alpha}: no data")
        continue

    # mean across seeds
    mean_curve = np.mean(seed_curves, axis=0)
    std_curve = np.std(seed_curves, axis=0)

    epochs = np.arange(len(mean_curve))

    plt.plot(
        epochs,
        mean_curve,
        label=f"alpha={alpha}",
        linewidth=2,
    )

    plt.fill_between(
        epochs,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.2
    )


plt.xlabel("Epoch")
plt.ylabel("Training KL")
plt.title("Training KL vs Epoch for Different alpha")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

output_path = FIGURES_DIR / "train_kl_curves_by_alpha.png"

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved figure to: {output_path}")
