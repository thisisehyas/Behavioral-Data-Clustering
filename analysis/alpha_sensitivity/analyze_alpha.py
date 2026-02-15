import os
import numpy as np
import json

RESULT_DIR = "experiments/alpha_sensitivity/results"

def main():
    summary = []

    for folder in os.listdir(RESULT_DIR):
        folder_path = os.path.join(RESULT_DIR, folder)

        if os.path.isdir(folder_path) and folder.startswith("alpha_"):
            alpha = float(folder.split("_")[1])

            losses_path = os.path.join(folder_path, "epoch_losses.npy")
            meta_path = os.path.join(folder_path, "meta.json")

            if os.path.exists(losses_path):
                losses = np.load(losses_path)
                final_kl = float(losses[-1])

                summary.append({
                    "alpha": alpha,
                    "final_kl": final_kl
                })

    summary = sorted(summary, key=lambda x: x["alpha"])

    print("\nAlpha Sensitivity Summary")
    print("-" * 40)
    for item in summary:
        print(f"Alpha: {item['alpha']:>5} | Final KL: {item['final_kl']:.4f}")

if __name__ == "__main__":
    main()