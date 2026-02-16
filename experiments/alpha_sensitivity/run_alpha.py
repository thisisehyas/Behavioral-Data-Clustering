import os
import random
import numpy as np
import json
import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans

from src.dec.autoencoder import Autoencoder
from src.dec.dec import DEC
from src.dec.target_distribution import target_distribution

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_dec_single_alpha_single_seed(
    X,
    train_idx,
    val_idx,
    ae_checkpoint_path,
    alpha,
    seed,
    config,
    output_dir,
):
    """
    Run a single DEC experiment for alpha sensitivity analysis where:
    - the Student-t distribution parameter (alpha) is varied
    - all other hyperparameters such as learning rate are fixed
    - the random seed and train/validation split are fixed

    Saves all artifacts needed for alpha sensitivity comparison.
    """
    set_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load pretrained autoencoder
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(ae_checkpoint_path, map_location=device))
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    # freeze encoder
    for param in autoencoder.encoder.parameters():
        param.requires_grad = False
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    X_train = X_tensor[train_idx]
    X_val = X_tensor[val_idx]

    # compute train embeddings
    with torch.no_grad():
        Z_train = autoencoder.encoder(X_train).cpu().numpy()

    kmeans = KMeans(
        n_clusters=config["dec"]["num_clusters"],
        random_state=seed,
        n_init=20
    )
    kmeans.fit(Z_train)

    cluster_centers = kmeans.cluster_centers_

    # initialize DEC model with KMeans centers
    dec_model = DEC(
        autoencoder=autoencoder,
        n_clusters=config["dec"]["num_clusters"],
        cluster_centers=cluster_centers,
        alpha=alpha
    ).to(device)

    optimizer = torch.optim.Adam(
        dec_model.clustering_layer.parameters(),
        lr=config["dec"]["learning_rate"]
    )

    train_kl = []
    val_kl = []

    with torch.no_grad():
        q_train, _ = dec_model(X_train)
        p_train = target_distribution(q_train)

    max_iterations = config["dec"]["max_iterations"]
    update_interval = config["dec"]["update_interval"]

    for iteration in range(max_iterations):
        if iteration % update_interval == 0:
            with torch.no_grad():
                q_train, _ = dec_model(X_train)
                p_train = target_distribution(q_train)
                
        dec_model.train()
        optimizer.zero_grad()

        q_train, _ = dec_model(X_train)
        loss_train = F.kl_div(
            q_train.log(),
            p_train,
            reduction="batchmean",
        )

        loss_train.backward()
        optimizer.step()

        dec_model.eval()
        with torch.no_grad():
            q_val, _ = dec_model(X_val)
            p_val = target_distribution(q_val)
            loss_val = F.kl_div(
                q_val.log(),
                p_val,
                reduction="batchmean",
            )

        train_kl.append(loss_train.item())
        val_kl.append(loss_val.item())

        print(
            f"[alpha={alpha} | Seed={seed}] "
            f"Iter {iteration+1}/{max_iterations} | "
            f"Train KL: {loss_train.item():.4f} | "
            f"Val KL: {loss_val.item():.4f}"
        )

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "train_kl.npy"), np.array(train_kl))
    np.save(os.path.join(output_dir, "val_kl.npy"), np.array(val_kl))
    np.save(
        os.path.join(output_dir, "cluster_centers.npy"),
        dec_model.clustering_layer.cluster_centers.detach().cpu().numpy(),
    )

    with torch.no_grad():
        q_val, _ = dec_model(X_val)
        val_labels = torch.argmax(q_val, dim=1).cpu().numpy()
        Z_val = autoencoder.encoder(X_val).cpu().numpy()

    np.save(os.path.join(output_dir, "val_labels.npy"), val_labels)
    np.save(os.path.join(output_dir, "q_val.npy"), q_val.cpu().numpy())
    np.save(os.path.join(output_dir, "Z_val.npy"), Z_val)

    meta = {
        "alpha": alpha,
        "learning_rate": config["dec"]["learning_rate"],
        "seed": seed,
        "num_clusters": config["dec"]["num_clusters"],
        "max_iterations": config["dec"]["max_iterations"],
        "update_interval": config["dec"]["update_interval"],
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "ae_checkpoint_path": ae_checkpoint_path,
    }

    with open(os.path.join(output_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)
