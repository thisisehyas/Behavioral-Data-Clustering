"""
Evaluattion of DEC update_interval

This script evaluates clustering quality and interpretability
for a fixed DEC configuration 

It assumes:
- A pretrained autoencoder (frozen encoder)
- Latent embeddings already prepared
- DEC trained with a fixed update interval
"""

import os 
import pickle
import numpy as np
import torch
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from src.dec.autoencoder import Autoencoder
from src.dec.dec import DEC
from src.dec.train_dec import train_dec
from src.dec.utils import compute_q_full

from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
)


# Configuration
UPDATE_INTERVAL = 20
N_CLUSTERS = 5
BATCH_SIZE = 256
MAX_EPOCHS = 50
LR = 1e-3

OUTPUT_DIR = f"results/eval_update_{UPDATE_INTERVAL}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Running DEC evaluation for update_interval={UPDATE_INTERVAL}")

    # load embeddings
    X_scaled = np.load("./embeddings/X_scaled.npy")
    print("Loaded embeddings:", X_scaled.shape)

    cluster_centers = np.load("embeddings/kmeans_centers_k5.npy")
    print("Loaded KMeans centers:", cluster_centers.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = Autoencoder().to(device)

    ae_ckpt_path = "models/autoencoder_weights.pth"
    autoencoder.load_state_dict(torch.load(ae_ckpt_path, map_location=device))
    autoencoder.eval()

    print("Loaded pretrained autoencoder")

    # initialize DEC model
    dec_model = DEC(
        autoencoder=autoencoder,
        n_clusters=N_CLUSTERS,
        cluster_centers=cluster_centers
    ).to(device)

    # freeze encoder
    for param in dec_model.encoder.parameters():
        param.requires_grad = False

    print("DEC model initialized and encoder frozen")

    # train DEC
    print("Training DEC model...")
    epoch_losses = train_dec(
        dec_model=dec_model,
        X=X_scaled,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        update_interval=UPDATE_INTERVAL,
        lr=LR,
    )

    print("DEC training finished")

    # Save trained DEC model
    model_path = os.path.join(OUTPUT_DIR, "dec_model.pt")
    torch.save(dec_model.state_dict(), model_path)

    with open(os.path.join(OUTPUT_DIR, "epoch_losses.pkl"), "wb") as f:
        pickle.dump(epoch_losses, f)
    
    print(f"Saved DEC model and losses to {OUTPUT_DIR}")

    print("Computing final soft assignments (q)...")

    q_final = compute_q_full(dec_model, X_scaled)
    labels = np.argmax(q_final, axis=1)

    print("q_final shape:", q_final.shape)
    print("labels shape:", labels.shape)
    print("Cluster sizes:", np.bincount(labels))

    np.save(os.path.join(OUTPUT_DIR, "q_final.npy"), q_final)
    np.save(os.path.join(OUTPUT_DIR, "labels_k5.npy"), labels)

    print("Saved q_final and labels")

    print("Computing clustering metrics...")

    embeddings = np.load('embeddings/embeddings.npy')

    sil = silhouette_score(embeddings, labels, metric="euclidean")
    ch = calinski_harabasz_score(embeddings, labels)
    db = davies_bouldin_score(embeddings, labels)

    print(f"Silhouette score: {sil:.4f}")
    print(f"Calinski-Harabasz score: {ch:.2f}")
    print(f"Davies-Bouldin score: {db:.4f}")

    # save metrics
    metrics = {
        "update_interval": UPDATE_INTERVAL,
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davis_bouldin": db
    }

    metrics_path = os.path.join(OUTPUT_DIR, "clustering_metrics.json")
    with open(metrics_path, "w") as f:
        import json
        json.dump(metrics, f, indent=2)
    
    print(f"Saved clustering metrics to {metrics_path}")

    print("Running UMAP visualization...")

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        hue=labels,
        palette="tab10",
        s=10,
        legend="full"
    )
    plt.title(f"UMAP projection (DEC, update_interval={UPDATE_INTERVAL})")
    plt.tight_layout()

    umap_path = os.path.join(OUTPUT_DIR, "umap_clusters.png")
    plt.savefig(umap_path, dpi=200)
    plt.close()

    print(f"Saved UMAP plot to {umap_path}")

if __name__ == "__main__":
    main()