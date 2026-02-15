import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "experiments/alpha_sensitivity/results"
OUTPUT_DIR = "analysis/alpha_sensitivity"

def main():
    summary = []

    for folder in os.listdir(RESULTS_DIR):
        folder_path = os.path.join(RESULTS_DIR, folder)

        if os.path.isdir(folder_path) and folder.startswith("alpha_"):
            alpha = float(folder.split("_")[1])

            losses_path = os.path.join(folder_path, "epoch_losses.npy")

            if os.path.exists(losses_path):
                losses = np.load(losses_path)
                final_kl = float(losses[-1])

                summary.append({
                    "alpha": alpha,
                    "final_kl": final_kl
                })

    # Sort by alpha
    summary = sorted(summary, key=lambda x: x["alpha"])

    # Convert to DataFrame
    df = pd.DataFrame(summary)

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "alpha_summary.csv")
    df.to_csv(csv_path, index=False)

    print("\nAlpha Sensitivity Summary")
    print("-" * 40)
    print(df)

    # Plot KL vs Alpha
    plt.figure()
    plt.plot(df["alpha"], df["final_kl"], marker="o")
    plt.xlabel("Alpha")
    plt.ylabel("Final KL Divergence")
    plt.title("KL Divergence vs Alpha")
    plt.xscale("log")
    plt.grid(True)

    plot_path = os.path.join(OUTPUT_DIR, "kl_vs_alpha.png")
    plt.savefig(plot_path)
    plt.close()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved plot to: {plot_path}")

if __name__ == "__main__":
    main()