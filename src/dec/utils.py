import numpy as np
import torch

def compute_q_full(dec_model, Z_numpy, batch_size=8192):
    """
    Compute the full soft assignment matrix (q) for all samples in Z_numpy using the DEC model.

    Args:
        dec_model: trained DEC model
        Z_numpy: np.array, latent embeddings of shape (n_samples, latent_dim)
        batch_size: int, number of samples per batch

    Returns:
        q_full: np.array, soft assignments of shape (n_samples, n_clusters)
    """
    device = next(dec_model.parameters()).device
    dec_model.eval()
    qs = []

    with torch.no_grad():
        for start in range(0, Z_numpy.shape[0], batch_size):
            end = min(start + batch_size, Z_numpy.shape[0])
            batch = torch.tensor(Z_numpy[start:end], dtype=torch.float32).to(device)
            q_batch, _ = dec_model(batch)
            qs.append(q_batch.cpu().numpy())

    q_full = np.vstack(qs)
    return q_full