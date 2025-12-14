import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from src.dec.target_distribution import target_distribution

def train_dec(dec_model, X, batch_size=256, max_epochs=50, update_interval=10, lr=1e-3):
  """
  - dec_model: the encoder + clustering layer combined
  - X: the full dataset of embeddings
  - batch_size: minibatch size
  - max_epochs: number of passes over the dataset
  - update_interval: how often we recompute the target distribution `p`
  - lr: learning rate
  """

  # put model in training mode
  dec_model.train()
  device = next(dec_model.parameters()).device

  # optimizer (updates encoder + cluster centers)
  # using the Adam optimizer as the org paper does
  optimizer = torch.optim.Adam(dec_model.parameters(), lr=lr)

  # creating batches
  dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

  # initial q for the entire dataset
  X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
  q_init, _ = dec_model(X_tensor)
  p = target_distribution(q_init).to(device)

  epoch_losses = []

  for epoch in range(max_epochs):
    batch_losses = []

    # every 'update_interval', refresh the whole target distribution p
    if epoch % update_interval == 0:
      q_full, _ = dec_model(X_tensor)
      p = target_distribution(q_full).to(device)

    # batch training
    for batch_idx, (x_batch,) in enumerate(dataloader):
      x_batch = x_batch.to(device)

      optimizer.zero_grad()

      q_batch, _ = dec_model(x_batch)

      # extract p batch too (same indices)
      p_batch = p[batch_idx * batch_size : batch_idx * batch_size + len(x_batch)]

      # KL divergence loss
      loss = F.kl_div(q_batch.log(), p_batch, reduction='batchmean')

      loss.backward()
      optimizer.step()

      batch_losses.append(loss.item())
    
    epoch_loss = sum(batch_losses) / len(batch_losses)
    epoch_losses.append(epoch_loss)

    print(f"Epoch {epoch+1}/{max_epochs}, Loss = {epoch_loss:.4f}")

  return epoch_losses