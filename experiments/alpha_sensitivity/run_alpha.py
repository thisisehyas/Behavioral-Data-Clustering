import os
import json
import argparse
import random
import numpy as np
import torch

from src.dec.autoencoder import Autoencoder
from src.dec.dec import DEC
from src.dec.train_dec import train_dec

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(alpha: float):
    # fixed settings
    seed = 42
    n_clusters = 5
    lr = 1e-3
    batch_size = 256
    max_epochs = 50
    update_interval = 20

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    X = np.load("embeddings/X_scaled.npy")
    cluster_centers = np.load("embeddings/kmeans_centers_k5.npy")

    # load autoencoder
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(
        torch.load("models/autoencoder_weights.pth", map_location=device)
    )
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    # initialize DEC
    dec_model = DEC(
        autoencoder=autoencoder,
        n_clusters=n_clusters,
        cluster_centers=cluster_centers,
        alpha=alpha
    ).to(device)

    # freeze encoder
    for param in dec_model.encoder.parameters():
        param.requires_grad = False

    # train DEC
    epoch_losses = train_dec(
        dec_model=dec_model,
        X=X,
        batch_size=batch_size,
        max_epochs=max_epochs,
        update_interval=update_interval,
        lr = lr,
    )

    # create output directory
    output_dir = f"experiments/alpha_sensitivity/results/alpha_{alpha}"
    os.makedirs(output_dir, exist_ok=True)

    # save losses
    np.save(os.path.join(output_dir, "epoch_losses.npy"), np.array(epoch_losses))

    # save meta info
    meta = {
        "alpha": alpha,
        "n_clusters": n_clusters,
        "learning_rate": lr,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "update_interval": update_interval,
        "seed": seed
    }

    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Finished alpha = {alpha}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DEC with different alpha values")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value for DEC")
    args = parser.parse_args()

    main(args.alpha)
