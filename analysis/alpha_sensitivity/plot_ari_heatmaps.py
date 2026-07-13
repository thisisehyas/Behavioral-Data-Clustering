from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ARI_DIR = Path("analysis/alpha_sensitivity/aggregated")
FIGURES_DIR = Path("analysis/alpha_sensitivity/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Find all ARI matrix files
ari_files = sorted(ARI_DIR.glob("ari_matrix_alpha_*.npy"))

if not ari_files:
    raise RuntimeError("No ARI matrix files found.")

for ari_path in ari_files:
    # extract alpha from filename
    alpha_str = ari_path.stem.replace("ari_matrix_alpha_", "")
    alpha = float(alpha_str)

    ari_matrix = np.load(ari_path)

    plt.figure(figsize=(4, 4))
    im = plt.imshow(ari_matrix, vmin=0, vmax=1, cmap="coolwarm")

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"Pairwise ARI between seeds (α = {alpha})")

    n = ari_matrix.shape[0]
    plt.xticks(range(n))
    plt.yticks(range(n))
    plt.xlabel("Seed")
    plt.ylabel("Seed")

    output_path = FIGURES_DIR / f"ari_heatmap_alpha_{alpha_str}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved ARI heatmap to: {output_path}")