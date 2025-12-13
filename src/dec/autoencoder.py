import torch
import torch.nn as nn

class Autoencoder(nn.Module):
  def __init__(self):
    super(Autoencoder, self).__init__()

    latent_dim = 5

    # Encoder
    self.encoder = nn.Sequential(
        nn.Linear(7, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Linear(16, latent_dim), # latent layer
    )

    # Decoder
    self.decoder = nn.Sequential(
        nn.Linear(latent_dim, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Linear(16, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Linear(32, 7) # reconstruct input
    )

  def forward(self, x):
    z = self.encoder(x) # compress
    x_recon = self.decoder(z) # reconstruct
    return x_recon
     