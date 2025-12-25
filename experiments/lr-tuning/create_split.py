import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# configuration
SPLIT_SEED = 42
TRAIN_RATIO = 0.8

# paths
DATA_PATH = Path("dataset/X_scaled.csv")
SPLIT_DIR = Path("experiments/lr-tuning/splits")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# load data
X_df = pd.read_csv(DATA_PATH)
X_scaled = X_df.to_numpy()

N = X_scaled.shape[0]
indices = np.arange(N)

# train / validation split
train_idx, val_idx = train_test_split(
    indices,
    train_size=TRAIN_RATIO,
    random_state=SPLIT_SEED,
    shuffle=True
)

np.save(SPLIT_DIR / "train_idx.npy", train_idx)
np.save(SPLIT_DIR / "val_idx.npy", val_idx)

# sanity checks
assert len(set(train_idx).intersection(set(val_idx))) == 0
assert len(train_idx) + len(val_idx) == N

print(f"Total samples: {N}")
print(f"Train samples: {len(train_idx)}")
print(f"Validation samples: {len(val_idx)}")
print("Split saved successfully.")