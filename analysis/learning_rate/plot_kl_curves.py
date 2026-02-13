from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

FIGURES_DIR = Path("analysis/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RUNS_DIR = Path("experiments/lr-tuning/runs")

# ----------------------------
# Configuration
# ----------------------------
SEED = 0
LR_DIRS = {
    "0.0001": "lr_1e-04",
    "0.0003": "lr_3e-04",
    "0.01": "lr_1e-02",
    # optional:
    # "0.003": "lr_3e-03",
}

plt.figure(figsize=(7, 5))

for lr_value, lr_dir in LR_DIRS.items():
    run_dir = RUNS_DIR / lr_dir / f"seed_{SEED}"

    train_kl_path = run_dir / "train_kl.npy"
    val_kl_path = run_dir / "val_kl.npy"

    if not train_kl_path.exists() or not val_kl_path.exists():
        print(f"Skipping LR={lr_value}: missing KL files")
        continue

    train_kl = np.load(train_kl_path)
    val_kl = np.load(val_kl_path)

    epochs = np.arange(len(train_kl))

    plt.plot(
        epochs,
        train_kl,
        label=f"LR={lr_value} (train)",
        linewidth=2,
    )
    plt.plot(
        epochs,
        val_kl,
        linestyle="--",
        label=f"LR={lr_value} (val)",
        linewidth=2,
    )

plt.xlabel("Epoch")
plt.ylabel("KL divergence")
plt.title("DEC training dynamics across learning rates")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

output_path = FIGURES_DIR / "kl_curves_by_lr.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved figure to: {output_path}")
