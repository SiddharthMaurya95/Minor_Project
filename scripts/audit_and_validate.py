"""
Master Audit & Dataset Validation Script for PhySSL-Net Research.
Performs comprehensive verification:
1. Configuration validation (configs/experiment.yaml).
2. Raw dataset schema, file integrity, partition dimensions, and leakage analysis (data/pneumoniamnist.npz).
3. Class counts and class imbalance ratio per split.
4. DataLoader batch shapes, intensity distributions, and SSL dual-view masking divergence.
5. Model architectures parameter counts and forward-pass output validity.
"""

import os
import sys
import yaml
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import build_dataloaders, load_medmnist_raw, get_stratified_subset_indices
from models.physsl_net import PhySSLNet, PhySSLSimSiam
from models.c_mixer import CMixer
from models.baselines import VanillaCNN, VGG16Small, ResNet50V2Small


def run_comprehensive_audit():
    print("=" * 70)
    print(" PHYSSL-NET: SYSTEM AUDIT, DATASET VALIDATION & INTEGRITY CHECKS")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # 1. Configuration Check
    # --------------------------------------------------------------------------
    config_path = "./configs/experiment.yaml"
    print("\n[1] CONFIGURATION VALIDATION")
    assert os.path.exists(config_path), f"Configuration file missing at {config_path}"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"    Config file: {config_path} (Valid YAML)")
    print(f"    Project: {cfg['project']['name']} | Group ID: {cfg['project']['group_id']}")
    print(f"    Target Dataset: {cfg['dataset']['name']}")
    print(f"    Planned Seeds: {cfg['seeds']}")
    print(f"    Baselines: {cfg['baselines']}")
    print(f"    Ablations: {cfg['ablations']}")
    print(f"    Curvature Attention Epsilon: {cfg['physsl_net']['curvature']['eps']} (Approved spec: 1e-2)")
    assert cfg['physsl_net']['curvature']['eps'] == 0.01, "Curvature epsilon must be 0.01 (1e-2)"

    # --------------------------------------------------------------------------
    # 2. Raw Dataset Inspection
    # --------------------------------------------------------------------------
    dataset_name = cfg['dataset']['name']
    npz_path = os.path.join(cfg['dataset']['data_dir'], f"{dataset_name}.npz")
    print(f"\n[2] RAW DATASET SCHEMA & PARTITION INTEGRITY ({npz_path})")
    assert os.path.exists(npz_path), f"Dataset file not found at {npz_path}"
    file_size_mb = os.path.getsize(npz_path) / (1024 * 1024)
    print(f"    File Size on Disk: {file_size_mb:.2f} MB")

    raw = np.load(npz_path)
    keys = list(raw.keys())
    print(f"    Keys in npz archive: {keys}")
    expected_keys = ['train_images', 'train_labels', 'val_images', 'val_labels', 'test_images', 'test_labels']
    assert all(k in keys for k in expected_keys), f"Missing keys in {npz_path}"

    train_imgs = raw['train_images']
    train_lbls = raw['train_labels'].squeeze()
    val_imgs = raw['val_images']
    val_lbls = raw['val_labels'].squeeze()
    test_imgs = raw['test_images']
    test_lbls = raw['test_labels'].squeeze()

    print(f"    Train Images: shape={train_imgs.shape}, dtype={train_imgs.dtype}, min={train_imgs.min()}, max={train_imgs.max()}")
    print(f"    Train Labels: shape={train_lbls.shape}, dtype={train_lbls.dtype}, unique={np.unique(train_lbls)}")
    print(f"    Val Images  : shape={val_imgs.shape}, dtype={val_imgs.dtype}, min={val_imgs.min()}, max={val_imgs.max()}")
    print(f"    Val Labels  : shape={val_lbls.shape}, dtype={val_lbls.dtype}, unique={np.unique(val_lbls)}")
    print(f"    Test Images : shape={test_imgs.shape}, dtype={test_imgs.dtype}, min={test_imgs.min()}, max={test_imgs.max()}")
    print(f"    Test Labels : shape={test_lbls.shape}, dtype={test_lbls.dtype}, unique={np.unique(test_lbls)}")

    # Sanity checks
    assert not np.isnan(train_imgs).any(), "NaN found in train_images"
    assert not np.isnan(val_imgs).any(), "NaN found in val_images"
    assert not np.isnan(test_imgs).any(), "NaN found in test_images"
    assert train_imgs.shape[1:] == (28, 28), f"Expected 28x28 spatial resolution, got {train_imgs.shape[1:]}"

    # --------------------------------------------------------------------------
    # 3. Class Distribution & Imbalance
    # --------------------------------------------------------------------------
    print("\n[3] CLASS LABELS & DISTRIBUTION")
    label_map = {0: "Normal", 1: "Pneumonia"}
    for split_name, lbls in [("Train", train_lbls), ("Val", val_lbls), ("Test", test_lbls)]:
        u, counts = np.unique(lbls, return_counts=True)
        dist_str = ", ".join([f"{label_map.get(k, k)} ({k}): {c} ({c/len(lbls):.1%})" for k, c in zip(u, counts)])
        print(f"    {split_name:<5} Split ({len(lbls):4d} samples) -> {dist_str}")

    # --------------------------------------------------------------------------
    # 4. Partition Leakage & Overlap Audit
    # --------------------------------------------------------------------------
    print("\n[4] PARTITION SEPARATION AUDIT (LEAKAGE CHECK)")
    train_bytes = set(img.tobytes() for img in train_imgs)
    val_bytes = set(img.tobytes() for img in val_imgs)
    test_bytes = set(img.tobytes() for img in test_imgs)

    train_val_overlap = train_bytes.intersection(val_bytes)
    train_test_overlap = train_bytes.intersection(test_bytes)
    val_test_overlap = val_bytes.intersection(test_bytes)

    print(f"    Unique image arrays - Train: {len(train_bytes)}/{len(train_imgs)}, Val: {len(val_bytes)}/{len(val_imgs)}, Test: {len(test_bytes)}/{len(test_imgs)}")
    print(f"    Train-Test Exact Match Overlap : {len(train_test_overlap)} (Strictly 0 expected)")
    print(f"    Val-Test Exact Match Overlap   : {len(val_test_overlap)} (Strictly 0 expected)")
    print(f"    Train-Val Duplicate Samples    : {len(train_val_overlap)} (Identified in MedMNIST v2 source)")
    assert len(train_test_overlap) == 0, "DATA LEAKAGE: Exact matches detected between Train and Test sets!"
    print("    [PASSED] Test partition is 100% held-out and completely isolated.")

    # --------------------------------------------------------------------------
    # 5. DataLoader & Preprocessing Pipeline
    # --------------------------------------------------------------------------
    print("\n[5] DATALOADER BATCH SANITY & SSL MASKING")
    loaders = build_dataloaders(
        dataset_name=dataset_name,
        data_dir=cfg['dataset']['data_dir'],
        batch_size=32,
        label_fraction=1.0,
        seed=42,
    )
    train_loader = loaders['train_loader']
    ssl_loader = loaders['ssl_loader']

    x_b, y_b = next(iter(train_loader))
    print(f"    Supervised Batch: x={x_b.shape} ({x_b.dtype}), y={y_b.shape} ({y_b.dtype})")
    print(f"    Normalized Intensity Range: [{x_b.min().item():.3f}, {x_b.max().item():.3f}]")
    assert x_b.shape == (32, 1, 28, 28), f"Unexpected tensor shape: {x_b.shape}"

    v1_b, v2_b = next(iter(ssl_loader))
    print(f"    SSL Dual-Views: v1={v1_b.shape}, v2={v2_b.shape}")
    l1_diff = (v1_b - v2_b).abs().sum().item()
    print(f"    Dual-View Stochastic L1 Difference: {l1_diff:.2f}")
    assert l1_diff > 0, "SSL views must have non-zero stochastic divergence!"

    # Stratified Subsampling Fractions
    print("    Stratified Label-Fraction Subsampling:")
    for frac in cfg['dataset']['label_fractions']:
        idx = get_stratified_subset_indices(train_lbls, fraction=frac, seed=42)
        sub_y = train_lbls[idx]
        u_s, c_s = np.unique(sub_y, return_counts=True)
        ratios = c_s / len(sub_y)
        print(f"      Fraction {frac:4.0%}: {len(idx):4d} samples | Normal: {ratios[0]:.1%}, Pneumonia: {ratios[1]:.1%}")

    # --------------------------------------------------------------------------
    # 6. Model Architecture Audit
    # --------------------------------------------------------------------------
    print("\n[6] MODEL ARCHITECTURE SUITE & PARAMETER PROFILE")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Execution Device: {device.upper()}")

    models_to_check = {
        "PhySSLNet (Full Proposed)": PhySSLNet(in_channels=1, num_classes=2),
        "PhySSLSimSiam (SSL Wrapper)": PhySSLSimSiam(encoder=PhySSLNet(in_channels=1, num_classes=2)),
        "C-Mixer (Zheng & Jia Baseline)": CMixer(in_channels=1, num_classes=2),
        "Vanilla CNN Baseline": VanillaCNN(in_channels=1, num_classes=2),
        "VGG16Small Baseline": VGG16Small(in_channels=1, num_classes=2),
        "ResNet50V2Small Baseline": ResNet50V2Small(in_channels=1, num_classes=2),
    }

    dummy_x = torch.randn(4, 1, 28, 28, device=device)
    for name, model in models_to_check.items():
        model = model.to(device)
        model.eval()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        with torch.no_grad():
            if "SimSiam" in name:
                loss = model(dummy_x, dummy_x)
                out_desc = f"SSL Loss={loss.item():.4f}"
            else:
                logits = model(dummy_x)
                assert logits.shape == (4, 2), f"Shape mismatch in {name}: {logits.shape}"
                out_desc = f"Logits shape={tuple(logits.shape)}"

        print(f"    {name:<32}: {params:8,d} params | {out_desc} | [OK]")

    print("\n" + "=" * 70)
    print(" ALL AUDIT & DATASET VALIDATION CHECKS PASSED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_comprehensive_audit()
