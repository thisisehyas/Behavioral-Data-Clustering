import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusteringLayer(nn.Module):
  def __init__(self, n_clusters, embedding_dim, cluster_centers=None, alpha=1.0):
    super(ClusteringLayer, self).__init__()
    self.n_clusters = n_clusters
    self.alpha = alpha

    #  cluster centers = trainable parameters
    if cluster_centers is None:
      # random initialization if not provided
      initial_centers = torch.zeros(n_clusters, embedding_dim, dtype=torch.float32)
    else:
      # initialize with KMeans centers
      initial_centers = torch.tensor(cluster_centers, dtype=torch.float32)

    self.cluster_centers = nn.Parameter(initial_centers)

  def forward(self, z):
    """
    z: tensor of shape (batch_size, embedding_dim)
    Returns a: soft assignment (batch_size, n_clusters)
    """

    # expand dimentions to compute pairwise distances
    z_expanded = z.unsqueeze(1)
    centers_expanded = self.cluster_centers.unsqueeze(0)

    # squared Euclidean distance
    dist = torch.sum((z_expanded - centers_expanded)**2, dim=2)

    numerator = (1.0 + dist / self.alpha) ** (-(self.alpha + 1)/2)
    q = numerator / torch.sum(numerator, dim=1, keepdim=True) # normalize

    return q