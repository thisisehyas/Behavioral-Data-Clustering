import numpy as np
import yaml
from src.dec.training import run_dec_single_lr_single_seed

# load cofig 
with open("experiments/lr-tuning/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# load data
X = np.load("embeddings/X_scaled.npy")

train_idx = np.load("experiments/lr-tuning/splits/train_idx.npy")
val_idx = np.load("experiments/lr-tuning/splits/val_idx.npy")

lr = 1e-3
seed = 0

ae_checkpoints_path = "models/autoencoder_weights.pth"

output_dir = f"experiments/lr_tuning/debug_lr_{lr}_seed_{seed}"

# run single experiment
run_dec_single_lr_single_seed(
    X=X,
    train_idx=train_idx,
    val_idx=val_idx,
    ae_checkpoint_path=ae_checkpoints_path,
    lr=lr,
    seed=seed,
    config=config,
    output_dir=output_dir
)