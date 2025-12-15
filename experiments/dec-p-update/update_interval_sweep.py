"""
Experiment: Effect of target distribution update interval in DEC

This script evaluates how different update intervals for the target distribution p affect DEC training dynamics.

Outputs:
- results/dec_update_interval_losses.pkl
"""

import os
import pickle
import numpy as np
import torch 

from src.dec.autoencoder import Autoencoder
from src.dec.dec import DEC
from src.dec.train_dec import train_dec

# device setup 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load pretrained autoencoder
autoencoder = Autoencoder().to(device)

ae_ckpt_path = "./models/autoencoder_weights.pth"
if os.path.exists(ae_ckpt_path):
    autoencoder.load_state_dict(torch.load(ae_ckpt_path, map_location=device))
    print("Pretrained autoencoder weights loaded")
else:
    raise FileNotFoundError(f"Autoencoder checkpoint not found at {ae_ckpt_path}")

autoencoder.load_state_dict(torch.load(ae_ckpt_path, map_location=device))
autoencoder.eval()

# freeze encoder weights
for param in autoencoder.parameters():
    param.requires_grad = False

print("Encoder frozen")

# Load embeddings and KMeans initialization
X_scaled = np.load("embeddings/X_scaled.npy")
print("Loaded X_scaled:", X_scaled.shape)

cluster_centers = np.load("embeddings/kmeans_centers/npy")
print("Loaded KMeans centers:", cluster_centers.shape)

# Update-interval sweep
update_intervals = [1, 5, 10, 20, 50]
results = {}

print("\nStarting update-interval sweep...")

for ui in update_intervals:
    print(f"\n==== Training DEC with update_interval = {ui} ====")

    dec_model = DEC(
        autoencoder=autoencoder, 
        n_clusters=5, 
        cluster_centers=cluster_centers
    ).to(device)

    # ensure encoder is frozen
    for param in dec_model.encoder.parameters():
        param.requires_grad = False

    epoch_losses = train_dec(
        dec_model=dec_model, 
        X=X_scaled, 
        batch_size=256, 
        max_epochs=50, 
        update_interval=ui, 
        lr=1e-3
    )

    results[ui] = epoch_losses

# Save results
os.makedirs("results", exist_ok=True)

out_path = "results/dec_update_interval_losses.pkl"
with open(out_path, "wb") as f:
    pickle.dump(results, f)

print(f"\nSaved update-interval sweep results to {out_path}")
print("See docs/dec_update_interval.md for analysis of results.")