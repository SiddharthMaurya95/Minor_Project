import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from models.physsl_net import PhySSLNet
from data.dataset import build_dataloaders
from evaluation.metrics import compute_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- EVALUATING MULTI-SEED ACCURACY IMPROVEMENT WITH CLASS-WEIGHTED TRAINING ---")

seeds = [42, 123, 456, 789, 999]
default_accs = []
tuned_accs = []
default_specs = []
tuned_specs = []
aucs = []

# Pretrained SSL checkpoint
pretrained_ckpt = "checkpoints/physsl_simsiam_pretrained.pth"
assert os.path.exists(pretrained_ckpt), "Pretrained checkpoint missing"

for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)

    loaders = build_dataloaders("pneumoniamnist", batch_size=64, seed=seed)
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]

    # Calculate exact class weights for this seed
    targets = np.array([y for _, y in train_loader.dataset])
    c0 = np.sum(targets == 0)
    c1 = np.sum(targets == 1)
    w0 = (c0 + c1) / (2.0 * c0)
    w1 = (c0 + c1) / (2.0 * c1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

    model = PhySSLNet(in_channels=1, num_classes=2).to(device)
    ckpt = torch.load(pretrained_ckpt, map_location=device)
    state_dict = ckpt.get("encoder_state_dict", ckpt)
    mdict = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in mdict and v.shape == mdict[k].shape}
    mdict.update(filtered)
    model.load_state_dict(mdict)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 25
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_auc = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vout = model(vx.to(device)).softmax(dim=1)[:, 1]
                v_preds.append(vout.cpu().numpy())
                v_targets.append(vy.numpy())
        v_preds = np.concatenate(v_preds)
        v_targets = np.concatenate(v_targets)
        val_m = compute_metrics(v_targets, v_preds, threshold=0.5)

        if val_m["roc_auc"] > best_val_auc:
            best_val_auc = val_m["roc_auc"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Evaluate best model on test set
    model.load_state_dict(best_weights)
    model.eval()

    t_preds, t_targets = [], []
    with torch.no_grad():
        for tx, ty in test_loader:
            tout = model(tx.to(device)).softmax(dim=1)[:, 1]
            t_preds.append(tout.cpu().numpy())
            t_targets.append(ty.numpy())
    t_preds = np.concatenate(t_preds)
    t_targets = np.concatenate(t_targets)

    # Standard metrics at t=0.50
    m_def = compute_metrics(t_targets, t_preds, threshold=0.5)

    # Val-tuned threshold
    best_t = 0.5
    best_v_score = -1.0
    for t in np.linspace(0.1, 0.9, 81):
        vm = compute_metrics(v_targets, v_preds, threshold=t)
        score = vm["recall"] + vm.get("specificity", 0.0)
        if score > best_v_score:
            best_v_score = score
            best_t = t
    m_tun = compute_metrics(t_targets, t_preds, threshold=best_t)

    default_accs.append(m_def["accuracy"] * 100)
    tuned_accs.append(m_tun["accuracy"] * 100)
    default_specs.append(m_def["specificity"] * 100)
    tuned_specs.append(m_tun["specificity"] * 100)
    aucs.append(m_def["roc_auc"])

    print(f"Seed {seed:3d}: Val AUC={best_val_auc:.4f} | Default (t=0.5): Acc={m_def['accuracy']*100:.2f}%, Spec={m_def['specificity']*100:.2f}% | Val-Tuned (t={best_t:.2f}): Acc={m_tun['accuracy']*100:.2f}%, Spec={m_tun['specificity']*100:.2f}%")

print("\n--- SUMMARY ACROSS 5 SEEDS ---")
print(f"Default (t=0.50): Mean Accuracy = {np.mean(default_accs):.2f}% ± {np.std(default_accs):.2f}%, Mean Specificity = {np.mean(default_specs):.2f}% ± {np.std(default_specs):.2f}%")
print(f"Val-Tuned:        Mean Accuracy = {np.mean(tuned_accs):.2f}% ± {np.std(tuned_accs):.2f}%, Mean Specificity = {np.mean(tuned_specs):.2f}% ± {np.std(tuned_specs):.2f}%")
print(f"Mean ROC-AUC:     {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
