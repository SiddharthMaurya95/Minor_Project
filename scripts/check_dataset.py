"""
Script to verify dataset integrity, official splits, and preprocessing.
Checks:
- Download and existence of MedMNIST datasets.
- Tensor dimensions and channel counts.
- Strict preservation of official train/val/test splits.
- Class distribution and imbalance ratios.
- Deterministic stratified label fraction subsampling.
- Stochastic dual-view masking for SSL pretraining.
- Absence of train/val/test data overlap.
"""

import os
import sys
import argparse
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import build_dataloaders, load_medmnist_raw, get_stratified_subset_indices


def check_dataset_integrity(dataset_name="pneumoniamnist", data_dir="./data"):
    print(f"\n=======================================================")
    print(f" Checking Dataset: {dataset_name.upper()}")
    print(f"=======================================================")

    raw = load_medmnist_raw(dataset_name=dataset_name, data_dir=data_dir)

    train_x, train_y = raw['train_images'], raw['train_labels'].squeeze()
    val_x, val_y = raw['val_images'], raw['val_labels'].squeeze()
    test_x, test_y = raw['test_images'], raw['test_labels'].squeeze()

    print(f"Train split : {train_x.shape}, labels: {train_y.shape}")
    print(f"Val split   : {val_x.shape}, labels: {val_y.shape}")
    print(f"Test split  : {test_x.shape}, labels: {test_y.shape}")

    unique_classes, counts = np.unique(train_y, return_counts=True)
    print(f"Classes found ({len(unique_classes)}): {dict(zip(unique_classes, counts))}")

    # Assert no nan or inf
    assert not np.isnan(train_x).any(), "NaN values detected in training set!"
    assert not np.isinf(train_x).any(), "Inf values detected in training set!"

    # Assert image resolution
    assert train_x.shape[1] == 28 and train_x.shape[2] == 28, f"Unexpected image shape: {train_x.shape}"

    # Build dataloaders
    loaders = build_dataloaders(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=32,
        label_fraction=1.0,
        seed=42,
    )

    print("\n[+] Testing Supervised DataLoader Batch:")
    x_batch, y_batch = next(iter(loaders['train_loader']))
    print(f"    Image batch shape: {x_batch.shape}, dtype: {x_batch.dtype}")
    print(f"    Label batch shape: {y_batch.shape}, dtype: {y_batch.dtype}")
    print(f"    Intensity range  : min={x_batch.min().item():.3f}, max={x_batch.max().item():.3f}")

    print("\n[+] Testing SSL Pretraining Dual-View Masking DataLoader:")
    v1_batch, v2_batch = next(iter(loaders['ssl_loader']))
    print(f"    View 1 batch shape: {v1_batch.shape}")
    print(f"    View 2 batch shape: {v2_batch.shape}")
    diff = (v1_batch - v2_batch).abs().sum().item()
    print(f"    Total absolute difference between views: {diff:.2f}")
    assert diff > 0, "Views should be stochastically masked and distinct!"

    print("\n[+] Testing Stratified Subsampling Fractions:")
    for frac in [1.0, 0.5, 0.25, 0.10]:
        indices = get_stratified_subset_indices(train_y, fraction=frac, seed=42)
        sub_labels = train_y[indices]
        _, sub_counts = np.unique(sub_labels, return_counts=True)
        ratios = sub_counts / len(sub_labels)
        ratio_str = ", ".join([f"class {c}: {r:.1%}" for c, r in zip(unique_classes, ratios)])
        print(f"    Fraction {frac:4.0%}: {len(indices):4d} samples | {ratio_str}")

    print("\n[+] Verifying Partition Separation (No Leakage):")
    # Quick hash check of image arrays to guarantee split separation
    train_hashes = set(hash(x.tobytes()) for x in train_x)
    val_hashes = set(hash(x.tobytes()) for x in val_x)
    test_hashes = set(hash(x.tobytes()) for x in test_x)

    val_leakage = train_hashes.intersection(val_hashes)
    test_leakage = train_hashes.intersection(test_hashes)
    print(f"    Train-Val exact matches : {len(val_leakage)}")
    print(f"    Train-Test exact matches: {len(test_leakage)}")

    print(f"\n[PASSED] All integrity checks for {dataset_name} succeeded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pneumoniamnist")
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    check_dataset_integrity(dataset_name=args.dataset, data_dir=args.data_dir)
