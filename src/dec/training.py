import os
import numpy as np
import json
from src.dec.train_dec import train_dec

def save_experiment_results(
    results,
    output_dir,
    meta,
):
    """
    Save all artifacts produced by one DEC training run.
    """

    os.makedirs(output_dir, exist_ok=True)

    np.save(
        os.path.join(output_dir, "train_kl.npy"),
        np.array(results["train_kl"]),
    )

    np.save(
        os.path.join(output_dir, "val_kl.npy"),
        np.array(results["val_kl"]),
    )

    np.save(
        os.path.join(output_dir, "cluster_centers.npy"),
        results["cluster_centers"],
    )

    np.save(
        os.path.join(output_dir, "val_labels.npy"),
        results["val_labels"],
    )

    np.save(
        os.path.join(output_dir, "q_val.npy"),
        results["q_val"],
    )

    np.save(
        os.path.join(output_dir, "Z_val.npy"),
        results["Z_val"],
    )

    np.save(
        os.path.join(output_dir, "full_labels.npy"),
        results["full_labels"],
    )

    np.save(
        os.path.join(output_dir, "q_full.npy"),
        results["q_full"],
    )

    np.save(
        os.path.join(output_dir, "Z_full.npy"),
        results["Z_full"],
    )

    with open(
        os.path.join(output_dir, "meta.json"),
        "w",
    ) as f:
        json.dump(meta, f, indent=2)

def run_dec_single_lr_single_seed(
    X,
    train_idx,
    val_idx,
    ae_checkpoint_path,
    lr,
    seed,
    config,
    output_dir,
):
    """
    Run one DEC experiment with:
    - fixed learning rate
    - fixed random seed
    - fixed train/validation split

    Saves all artifacts needed for LR comparison.
    """
    config = config.copy()
    config["dec"] = config["dec"].copy()
    config["dec"]["learning_rate"] = lr

    results = train_dec(
        X=X,
        train_idx=train_idx,
        val_idx=val_idx,
        ae_checkpoint_path=ae_checkpoint_path,
        config=config,
        seed=seed,
    )

    meta = {
        "learning_rate": lr,
        "seed": seed,
        "num_clusters": config["dec"]["num_clusters"],
        "alpha": config["dec"]["alpha"],
        "max_epochs": config["dec"]["max_epochs"],
        "update_interval": config["dec"]["update_interval"],
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "ae_checkpoint_path": ae_checkpoint_path,
    }

    save_experiment_results(
        results=results,
        output_dir=output_dir,
        meta=meta,
    )

def run_dec_single_alpha_single_seed(
    X,
    train_idx,
    val_idx,
    ae_checkpoint_path,
    alpha,
    seed,
    config,
    output_dir,
):
    """
    Run one DEC experiment with:
    - fixed alpha (Student-t parameter)
    - fixed random seed
    - fixed train/validation split

    Saves all artifacts needed for alpha comparison.
    """
    config = config.copy()
    config["dec"] = config["dec"].copy()
    config["dec"]["alpha"] = alpha

    results = train_dec(
        X=X,
        train_idx=train_idx,
        val_idx=val_idx,
        ae_checkpoint_path=ae_checkpoint_path,
        config=config,
        seed=seed,
    )

    meta = {
        "alpha": alpha,
        "seed": seed,
        "num_clusters": config["dec"]["num_clusters"],
        "learning_rate": config["dec"]["learning_rate"],
        "max_epochs": config["dec"]["max_epochs"],
        "update_interval": config["dec"]["update_interval"],
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "ae_checkpoint_path": ae_checkpoint_path,
    }

    save_experiment_results(
        results=results,
        output_dir=output_dir,
        meta=meta,
    )

def run_dec_single_update_interval_single_seed(
    X,
    train_idx,
    val_idx,
    ae_checkpoint_path,
    update_interval,
    seed,
    config,
    output_dir,
):
    """
    Run one DEC experiment with:
    - fixed update interval
    - fixed random seed
    - fixed train/validation split

    Saves all artifacts needed for update-interval comparison.
    """
    config = config.copy()
    config["dec"] = config["dec"].copy()
    config["dec"]["update_interval"] = update_interval

    results = train_dec(
        X=X,
        train_idx=train_idx,
        val_idx=val_idx,
        ae_checkpoint_path=ae_checkpoint_path,
        config=config,
        seed=seed,
    )

    meta = {
        "update_interval": update_interval,
        "seed": seed,
        "num_clusters": config["dec"]["num_clusters"],
        "alpha": config["dec"]["alpha"],
        "learning_rate": config["dec"]["learning_rate"],
        "max_epochs": config["dec"]["max_epochs"],
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "ae_checkpoint_path": ae_checkpoint_path,
    }

    save_experiment_results(
        results=results,
        output_dir=output_dir,
        meta=meta,
    )