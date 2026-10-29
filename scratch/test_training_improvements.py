import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from models.physsl_net import PhySSLNet
from data.dataset import build_dataloaders
from evaluation.metrics import compute_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running data augmentation + weighted loss on: {device}")

loaders = build_dataloaders("pneumoniamnist", batch_size=64, seed=42)
train_loader = loaders["train_loader"]
val_loader = loaders["val_loader"]
test_loader = loaders["test_loader"]

# Class weights
targets = np.array([y for _, y in train_loader.dataset])
c0 = np.sum(targets == 0)
c1 = np.sum(targets == 1)
total = c0 + c1
w0 = total / (2.0 * c0)
w1 = total / (2.0 * c1)
class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

# Model setup
torch.manual_seed(42)
np.random.seed(42)

model = PhySSLNet(in_channels=1, num_classes=2).to(device)
pretrained_ckpt = "checkpoints/physsl_simsiam_pretrained.pth"
if os.path.exists(pretrained_ckpt):
    ckpt = torch.load(pretrained_ckpt, map_location=device)
    state_dict = ckpt.get("encoder_state_dict", ckpt)
    mdict = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in mdict and v.shape == mdict[k].shape}
    mdict.update(filtered)
    model.load_state_dict(mdict)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
epochs = 30
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# Data augmentation for training: random horizontal flip
hflip = T.RandomHorizontalFlip(p=0.5)

best_val_score = -1.0
best_model_weights = None

for epoch in range(1, epochs + 1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        # Apply random horizontal flip on the fly
        x_aug = hflip(x)
        optimizer.zero_grad()
        out = model(x_aug)
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

    val_score = 0.5 * (val_m["recall"] + val_m.get("specificity", 0.0))
    if val_score > best_val_score:
        best_val_score = val_score
        best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_model_weights)
model.eval()

# Test evaluation with and without Test-Time Augmentation (TTA)
t_preds, t_preds_tta, t_targets = [], [], []
with torch.no_grad():
    for tx, ty in test_loader:
        tx = tx.to(device)
        # Standard
        p1 = model(tx).softmax(dim=1)[:, 1]
        # Flipped
        p2 = model(torch.flip(tx, dims=[-1])).softmax(dim=1)[:, 1]
        p_tta = 0.5 * (p1 + p2)

        t_preds.append(p1.cpu().numpy())
        t_preds_tta.append(p_tta.cpu().numpy())
        t_targets.append(ty.numpy())

t_preds = np.concatenate(t_preds)
t_preds_tta = np.concatenate(t_preds_tta)
t_targets = np.concatenate(t_targets)

# Standard evaluation (t=0.50)
m_std = compute_metrics(t_targets, t_preds, threshold=0.5)
m_tta = compute_metrics(t_targets, t_preds_tta, threshold=0.5)

# Validation-tuned threshold on TTA
best_t = 0.5
best_v_score = -1.0
for t in np.linspace(0.1, 0.9, 81):
    vm = compute_metrics(v_targets, v_preds, threshold=t)
    score = vm["recall"] + vm.get("specificity", 0.0)
    if score > best_v_score:
        best_v_score = score
        best_t = t

m_tta_tuned = compute_metrics(t_targets, t_preds_tta, threshold=best_t)

print("\n--- RESULTS: DATA AUGMENTATION + CLASS WEIGHTING (SEED 42) ---")
print(f"Standard (t=0.50): Acc={m_std['accuracy']*100:.2f}%, AUC={m_std['roc_auc']:.4f}, Sens={m_std['sensitivity']*100:.2f}%, Spec={m_std['specificity']*100:.2f}%")
print(f"TTA (t=0.50):      Acc={m_tta['accuracy']*100:.2f}%, AUC={m_tta['roc_auc']:.4f}, Sens={m_tta['sensitivity']*100:.2f}%, Spec={m_tta['specificity']*100:.2f}%")
print(f"TTA + Val-Tuned (t={best_t:.2f}): Acc={m_tta_tuned['accuracy']*100:.2f}%, AUC={m_tta_tuned['roc_auc']:.4f}, Sens={m_tta_tuned['sensitivity']*100:.2f}%, Spec={m_tta_tuned['specificity']*100:.2f}%")
