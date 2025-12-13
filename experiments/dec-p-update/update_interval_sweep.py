import os
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# model imports
from src.dec.autoencoder import Autoencoder
from src.dec.dec import DEC

print("DEC modules imported successfully")

autoencoder = Autoencoder().to(device)
print("Autoencoder created")

dec_model = DEC(autoencoder=autoencoder, n_clusters=5).to(device)
print("DEC model created")

# load pretrained autoencoder weights
ae_ckpt_path = "./models/autoencoder_weights.pth"

if os.path.exists(ae_ckpt_path):
    autoencoder.load_state_dict(torch.load(ae_ckpt_path, map_location=device))
    print("Pretrained autoencoder weights loaded")

else:
    raise FileNotFoundError(f"Autoencoder checkpoint not found at {ae_ckpt_path}")

# freeze encoder weights
for param in autoencoder.parameters():
    param.requires_grad = False

print("Encoder frozen")

import numpy as np

# load KMeans centers
centers_path = "embeddings/kmeans_centers_k5.npy"
cluster_centers = np.load(centers_path)
print("Loaded KMeans centers:", cluster_centers.shape)

dec_model = DEC(autoencoder=autoencoder, n_clusters=5, cluster_centers=cluster_centers).to(device)
print("DEC model initialized with KMeans centers")

import pandas as pd

# load latent embeddings
X_path = "./embeddings/X_scaled.npy"
X_scaled = np.load("embeddings/X_scaled.npy")
print("Loaded X_scaled:", X_scaled.shape)

np.save("embeddings/X_scaled.npy", X_scaled)
print("Saved X_scaled.npy")

dec_model.eval()

with torch.no_grad():
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    q_init, z_out = dec_model(X_tensor)

q_np = q_init.cpu().numpy()

print("q shape:", q_np.shape)
print("q row sum (first 5):", q_np[:5].sum(axis=1))
print("q max prob (first 5):", q_np[:5].max(axis=1))
print("z_out shape:", z_out.shape)

labels_kmeans = np.load("embeddings/kmeans_labels_k5.npy")
labels_dec_init = q_np.argmax(axis=1)

agreement = np.mean(labels_kmeans == labels_dec_init)
print(f"DEC vs KMeans initial agreement: {agreement:.4f}")


# sanity test
from src.dec.target_distribution import target_distribution

with torch.no_grad():
    p_test = target_distribution(q_init)

print("p shape:", p_test.shape)
print("p row sum (first 5):", p_test[:5].sum(dim=1))

# prepate one training batch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# optimizer 
optimizer = optim.Adam(dec_model.parameters(), lr=1e-3)

# dataset and dataloader
dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
loader = DataLoader(dataset, batch_size=256, shuffle=False)

# take one batch
X_batch = next(iter(loader))[0].to(device)

print("x_batch shape:", X_batch.shape)

dec_model.train()
q_batch, z_batch = dec_model(X_batch)

print("q_batch shape:", q_batch.shape)
print("z_batch shape:", z_batch.shape)
print("q row sum (first 5):", q_batch[:5].sum(dim=1))

p_batch = target_distribution(q_batch)

print("p_batch shape:", p_batch.shape)
print("p row sum (first 5):", p_batch[:5].sum(dim=1))

import torch.nn.functional as F

loss = F.kl_div(q_batch.log(), p_batch, reduction="batchmean")

print("DEC KL loss (single batch):", loss.item())
