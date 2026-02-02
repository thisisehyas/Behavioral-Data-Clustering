import torch
import torch.nn as nn
import torch.nn.functional as F

from .autoencoder import Autoencoder
from .clustering_layer import ClusteringLayer

class DEC(nn.Module):
  def __init__(self, autoencoder, n_clusters=5, cluster_centers=None, alpha=1.0):
    super(DEC, self).__init__()

    # using the trained autoencoder
    self.encoder = autoencoder.encoder

    # initializing clustering layer
    self.clustering_layer = ClusteringLayer(n_clusters=n_clusters, embedding_dim=5, cluster_centers=cluster_centers, alpha=alpha)

  def forward(self, x):
    # encoding the input
    z = self.encoder(x)

    # computing soft assignment (q) from clustering layer
    q = self.clustering_layer(z)

    return q, z