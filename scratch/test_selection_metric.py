import os, sys, torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from models.physsl_net import PhySSLNet
from data.dataset import build_dataloaders
from evaluation.metrics import compute_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"

loaders = build_dataloaders("pneumoniamnist", batch_size=64, seed=42)
train_loader, val_loader, test_loader = loaders["train_loader"], loaders["val_loader"], loaders["test_loader"]

torch.manual_seed(42)
np.random.seed(42)

model = PhySSLNet(in_channels=1, num_classes=2).to(device)
ckpt = torch.load("checkpoints/physsl_simsiam_pretrained.pth", map_location=device)
mdict = model.state_dict()
filtered = {k: v for k, v in ckpt["encoder_state_dict"].items() if k in mdict and v.shape == mdict[k].shape}
mdict.update(filtered)
model.load_state_dict(mdict)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
epochs = 25
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

best_val_acc = -1.0
best_weights_acc = None

best_val_loss = 999.0
best_weights_loss = None

best_val_auc = -1.0
best_weights_auc = None

for epoch in range(1, epochs + 1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    scheduler.step()

    model.eval()
    v_preds, v_targets, v_losses = [], [], []
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(device), vy.to(device)
            out = model(vx)
            v_losses.append(criterion(out, vy).item())
            v_preds.append(out.softmax(dim=1)[:, 1].cpu().numpy())
            v_targets.append(vy.cpu().numpy())
    v_preds = np.concatenate(v_preds)
    v_targets = np.concatenate(v_targets)
    val_loss = np.mean(v_losses)
    val_m = compute_metrics(v_targets, v_preds, threshold=0.5)

    if val_m["accuracy"] > best_val_acc:
        best_val_acc = val_m["accuracy"]
        best_weights_acc = (epoch, {k: v.cpu().clone() for k, v in model.state_dict().items()})

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights_loss = (epoch, {k: v.cpu().clone() for k, v in model.state_dict().items()})

    if val_m["roc_auc"] > best_val_auc:
        best_val_auc = val_m["roc_auc"]
        best_weights_auc = (epoch, {k: v.cpu().clone() for k, v in model.state_dict().items()})

def eval_weights(w_tuple, name):
    ep, w = w_tuple
    model.load_state_dict(w)
    model.eval()
    t_preds, t_targets = [], []
    with torch.no_grad():
        for tx, ty in test_loader:
            tout = model(tx.to(device)).softmax(dim=1)[:, 1]
            t_preds.append(tout.cpu().numpy())
            t_targets.append(ty.numpy())
    t_preds = np.concatenate(t_preds)
    t_targets = np.concatenate(t_targets)
    m = compute_metrics(t_targets, t_preds, threshold=0.5)
    print(f"Selected by {name:15s} (Epoch {ep:2d}): Test Acc={m['accuracy']*100:.2f}%, AUC={m['roc_auc']:.4f}, Sens={m['sensitivity']*100:.2f}%, Spec={m['specificity']*100:.2f}%")

print("--- COMPARISON OF CHECKPOINT SELECTION METRICS (SEED 42) ---")
eval_weights(best_weights_auc, "Val ROC-AUC")
eval_weights(best_weights_acc, "Val Accuracy")
eval_weights(best_weights_loss, "Val Loss")
