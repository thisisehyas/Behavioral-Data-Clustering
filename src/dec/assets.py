import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "embeddings"
SPLIT_DIR = ROOT / "experiments" / "val_train_splits"

X_PATH = DATA_DIR / "X_scaled.npy"
TRAIN_IDX_PATH = SPLIT_DIR / "train_idx.npy"
VAL_IDX_PATH = SPLIT_DIR / "val_idx.npy"

def load_features():
    X = np.load(X_PATH)
    return X

def load_indices():
    train_idx = np.load(TRAIN_IDX_PATH)
    val_idx = np.load(VAL_IDX_PATH)
    return train_idx, val_idx

def load_fixed_assets():
    """
    Single entry-point for all fixed assets.
    """
    X = load_features()
    train_idx, val_idx = load_indices()

    assets = {
        "X": X,
        "train_idx": train_idx,
        "val_idx" : val_idx,
        "n_samples" : X.shape[0],
        "input_dim" : X.shape[1]
    }

    return assets