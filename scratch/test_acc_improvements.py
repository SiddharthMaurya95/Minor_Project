import os, sys
import torch
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from models.physsl_net import PhySSLNet
from data.dataset import build_dataloaders
from evaluation.metrics import compute_metrics, find_optimal_threshold

device = "cuda" if torch.cuda.is_available() else "cpu"

print("--- ANALYZING PHYSSL-NET ACCURACY & SPECIFICITY ---")
for seed in [42, 123, 456, 789, 999]:
    ckpt_path = f"checkpoints/physsl_net_pneumoniamnist_frac100_seed{seed}_best.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = PhySSLNet(in_channels=1, num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loaders = build_dataloaders("pneumoniamnist", batch_size=64, seed=seed)
    val_loader, test_loader = loaders["val_loader"], loaders["test_loader"]

    # Val predictions
    v_preds, v_targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            out = model(x.to(device)).softmax(dim=1)[:, 1]
            v_preds.append(out.cpu().numpy())
            v_targets.append(y.numpy())
    v_preds = np.concatenate(v_preds)
    v_targets = np.concatenate(v_targets)

    # Test predictions
    t_preds, t_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(device)).softmax(dim=1)[:, 1]
            t_preds.append(out.cpu().numpy())
            t_targets.append(y.numpy())
    t_preds = np.concatenate(t_preds)
    t_targets = np.concatenate(t_targets)

    base_m = compute_metrics(t_targets, t_preds, threshold=0.5)

    # Threshold tuning on validation set
    best_t = 0.5
    best_val_score = 0.0
    for t in np.linspace(0.1, 0.9, 81):
        vm = compute_metrics(v_targets, v_preds, threshold=t)
        # Using Youden's J statistic or balanced acc on val
        score = vm["recall"] + vm["specificity"]
        if score > best_val_score:
            best_val_score = score
            best_t = t

    tuned_m = compute_metrics(t_targets, t_preds, threshold=best_t)
    print(f"Seed {seed:3d}: Default (t=0.50): Acc={base_m['accuracy']*100:.2f}%, Sens={base_m['sensitivity']*100:.2f}%, Spec={base_m['specificity']*100:.2f}% | Val-tuned (t={best_t:.2f}): Acc={tuned_m['accuracy']*100:.2f}%, Sens={tuned_m['sensitivity']*100:.2f}%, Spec={tuned_m['specificity']*100:.2f}%")
