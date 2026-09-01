import random
import numpy as np
import torch
from src.dec.autoencoder import Autoencoder
from sklearn.cluster import KMeans
from src.dec.dec import DEC
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from src.dec.target_distribution import target_distribution

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_dec(
      X,
      train_idx,
      val_idx,
      ae_checkpoint_path,
      config,
      seed,
  ):
  """
  Train one DEC model using the canonical training pipeline.

  Parameters
  ----------
  X : np.ndarray
      Full dataset.

  train_idx : np.ndarray
      Indices of training split.

  val_idx : np.ndarray
      Indices of validation split.

  ae_checkpoint_path : str
      Path to the pretrained autoencoder.

  config : dict
      DEC configuration.

  seed : int
      Random seed.

  Returns
  -------
  dict
      Training results.
  """
  set_seed(seed)

  device = torch.device(
      "cuda" if torch.cuda.is_available() else "cpu"
  )

  # load pretrained autoencoder
  autoencoder = Autoencoder()

  autoencoder.load_state_dict(
    torch.load(ae_checkpoint_path, map_location=device)
  )

  autoencoder = autoencoder.to(device)
  autoencoder.eval()

  for param in autoencoder.encoder.parameters():
      param.requires_grad = False

  X_tensor = torch.tensor(
    X, 
    dtype=torch.float32, 
    device=device,
  )
  
  X_train = X_tensor[train_idx]
  X_val = X_tensor[val_idx]

  # compute train embeddings
  with torch.no_grad():
     Z_train = autoencoder.encoder(X_train).cpu().numpy()

  kmeans = KMeans(
     n_clusters=config["dec"]["num_clusters"],
     random_state=seed,
     n_init=20,
  )

  kmeans.fit(Z_train)

  cluster_centers = kmeans.cluster_centers_

  # initialize DEC model with KMeans centers
  dec_model = DEC(
        autoencoder=autoencoder,
        n_clusters=config["dec"]["num_clusters"],
        cluster_centers=cluster_centers,
        alpha=config["dec"]["alpha"],
  ).to(device)

  optimizer = torch.optim.Adam(
        dec_model.clustering_layer.parameters(),
        lr=config["dec"]["learning_rate"],
    )

  train_dataset = TensorDataset(X_train, torch.arange(len(X_train)))

  train_loader = DataLoader(
        train_dataset,
        batch_size=config["dec"]["batch_size"],
        shuffle=True,
    )
  
  # initialize the first target distribution
  with torch.no_grad():
        q_train, _ = dec_model(X_train)
        p_train = target_distribution(q_train)

  train_kl = []
  val_kl = []

  max_epochs = config["dec"]["max_epochs"]
  update_interval = config["dec"]["update_interval"]

  for epoch in range(max_epochs):
        if epoch % update_interval == 0:
            with torch.no_grad():
                q_train, _ = dec_model(X_train)
                p_train = target_distribution(q_train)

        dec_model.train()

        batch_losses = []
    
        for batch_idx, (x_batch, batch_indices) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            batch_indices = batch_indices.to(device)

            optimizer.zero_grad(set_to_none=True)

            q_batch, _ = dec_model(x_batch)

            p_batch = p_train[batch_indices]

            loss = F.kl_div(
                q_batch.log(),
                p_batch,
                reduction="batchmean",
            )

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        epoch_train_loss = sum(batch_losses) / len(batch_losses)
        train_kl.append(epoch_train_loss)

        dec_model.eval()

        with torch.no_grad():
            q_val, _ = dec_model(X_val)
            p_val = target_distribution(q_val)
            loss_val = F.kl_div(
                q_val.log(),
                p_val,
                reduction="batchmean",
            )

        val_kl.append(loss_val.item())

        print(
            f"Epoch {epoch+1}/{max_epochs}, "
            f"Train KL Loss = {epoch_train_loss:.4f}, "
            f"Val KL Loss = {loss_val.item():.4f}"
        )

  dec_model.eval()

  with torch.no_grad():
    # validation inference
    q_val, _ = dec_model(X_val)

    val_labels = torch.argmax(q_val, dim=1).cpu().numpy()

    Z_val = autoencoder.encoder(X_val).cpu().numpy()

    # Full dataset inference
    q_full, _ = dec_model(X_tensor)

    full_labels = torch.argmax(
        q_full,
        dim=1,
    ).cpu().numpy()

    z_full = autoencoder.encoder(
        X_tensor
    ).cpu().numpy()

  return {
    "train_kl": train_kl,
    "val_kl": val_kl,
    "cluster_centers":
        dec_model.clustering_layer.cluster_centers.detach().cpu().numpy(),
    "val_labels": val_labels,
    "q_val": q_val.cpu().numpy(),
    "p_val": p_val.cpu().numpy(),
    "Z_val": Z_val,
    "full_labels": full_labels,
    "q_full": q_full.cpu().numpy(),
    "Z_full": z_full,
    "p_train": p_train.cpu().numpy(),
    "seed": seed,
}

def train_dec_final(
    X,
    ae_checkpoint_path,
    config,
    seed,
):
    """
    Train the final DEC model using the canonical training pipeline.

    The final model is trained on the full dataset using:
    - a frozen pretrained encoder
    - fresh KMeans initialization
    - one fixed random seed
    - the selected DEC hyperparameters
    - mini-batch optimization

    No train/validation split is used.
    """

    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load pretrained autoencoder
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(
        torch.load(ae_checkpoint_path, map_location=device)
    )

    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    for param in autoencoder.encoder.parameters():
        param.requires_grad = False

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    # Compute full-dataset embeddings
    with torch.no_grad():
        Z_full = (
            autoencoder.encoder(X_tensor)
            .cpu()
            .numpy()
        )

    # Fresh KMeans initialization on the full dataset
    kmeans = KMeans(
        n_clusters=config["dec"]["num_clusters"],
        random_state=seed,
        n_init=20,
    )

    kmeans.fit(Z_full)

    cluster_centers = kmeans.cluster_centers_

    # Initialize DEC model with KMeans centers
    dec_model = DEC(
        autoencoder=autoencoder,
        n_clusters=config["dec"]["num_clusters"],
        cluster_centers=cluster_centers,
        alpha=config["dec"]["alpha"],
    ).to(device)

    optimizer = torch.optim.Adam(
        dec_model.clustering_layer.parameters(),
        lr=config["dec"]["learning_rate"],
    )

    train_dataset = TensorDataset(
        X_tensor,
        torch.arange(len(X_tensor), device=device),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["dec"]["batch_size"],
        shuffle=True,
    )

    # Initialize the first target distribution
    with torch.no_grad():
        q_full, _ = dec_model(X_tensor)
        p_full = target_distribution(q_full)

    train_kl = []

    max_epochs = config["dec"]["max_epochs"]
    update_interval = config["dec"]["update_interval"]

    for epoch in range(max_epochs):

        if epoch % update_interval == 0:
            with torch.no_grad():
                q_full, _ = dec_model(X_tensor)
                p_full = target_distribution(q_full)

        dec_model.train()

        batch_losses = []

        for batch_idx, (x_batch, batch_indices) in enumerate(train_loader):

            x_batch = x_batch.to(device)
            batch_indices = batch_indices.to(device)

            optimizer.zero_grad(set_to_none=True)

            q_batch, _ = dec_model(x_batch)

            p_batch = p_full[batch_indices]

            loss = F.kl_div(
                q_batch.log(),
                p_batch,
                reduction="batchmean",
            )

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        epoch_train_loss = sum(batch_losses) / len(batch_losses)
        train_kl.append(epoch_train_loss)

        print(
            f"Epoch {epoch + 1}/{max_epochs}, "
            f"Train KL Loss = {epoch_train_loss:.4f}"
        )

    # Final full-dataset inference
    dec_model.eval()

    with torch.no_grad():
        q_full, _ = dec_model(X_tensor)

        full_labels = torch.argmax(
            q_full,
            dim=1,
        ).cpu().numpy()

        Z_full = (
            autoencoder.encoder(X_tensor)
            .cpu()
            .numpy()
        )

        p_full = target_distribution(q_full)

    return {
        "train_kl": train_kl,
        "cluster_centers": (
            dec_model.clustering_layer
            .cluster_centers
            .detach()
            .cpu()
            .numpy()
        ),
        "full_labels": full_labels,
        "q_full": q_full.cpu().numpy(),
        "Z_full": Z_full,
        "p_full": p_full.cpu().numpy(),
        "seed": seed,
    }