import torch

def target_distribution(q):
  # q: (batch_size, n_clusters) -> soft assignment

  weight = q ** 2 / torch.sum(q, dim=0)
  p = (weight.t() / torch.sum(weight, dim=1)).t()

  return p.detach()
     