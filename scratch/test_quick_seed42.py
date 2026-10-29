import os, sys, time, torch
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
print(f"Using device: {device}", flush=True)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

loaders = build_dataloaders("pneumoniamnist", batch_size=64, seed=seed)
train_loader, val_loader, test_loader = loaders["train_loader"], loaders["val_loader"], loaders["test_loader"]

# Class weights
targets = np.array([y for _, y in train_loader.dataset])
c0 = np.sum(targets == 0)
c1 = np.sum(targets == 1)
w0 = (c0 + c1) / (2.0 * c0)
w1 = (c0 + c1) / (2.0 * c1)
class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
print(f"Class counts: normal={c0}, pneumonia={c1} | weights: w0={w0:.3f}, w1={w1:.3f}", flush=True)

model = PhySSLNet(in_channels=1, num_classes=2).to(device)
ckpt = torch.load("checkpoints/physsl_simsiam_pretrained.pth", map_location=device)
mdict = model.state_dict()
filtered = {k: v for k, v in ckpt["encoder_state_dict"].items() if k in mdict and v.shape == mdict[k].shape}
mdict.update(filtered)
model.load_state_dict(mdict)
print(f"Loaded {len(filtered)} tensors from SSL checkpoint.", flush=True)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
epochs = 25
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
hflip = T.RandomHorizontalFlip(p=0.5)

best_val_score = -1.0
best_weights = None
best_ep = 0

t0 = time.time()
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x_aug = hflip(x)
        optimizer.zero_grad()
        loss = criterion(model(x_aug), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    scheduler.step()

    # Val
    model.eval()
    v_preds, v_targets = [], []
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(device)
            out = model(vx)
            v_preds.append(out.softmax(dim=1)[:, 1].cpu().numpy())
            v_targets.append(vy.numpy())
    v_preds = np.concatenate(v_preds)
    v_targets = np.concatenate(v_targets)
    val_m = compute_metrics(v_targets, v_preds, threshold=0.5)

    score = val_m["accuracy"]
    if score > best_val_score:
        best_val_score = score
        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_ep = epoch

    if epoch % 5 == 0 or epoch == epochs:
        print(f"Epoch {epoch:02d}: Train Loss={total_loss/len(train_loader.dataset):.4f} | Val Acc={val_m['accuracy']*100:.2f}% | Val AUC={val_m['roc_auc']:.4f} | Best Val Acc={best_val_score*100:.2f}% (ep {best_ep})", flush=True)

elapsed = time.time() - t0
print(f"Training took {elapsed:.1f}s", flush=True)

model.load_state_dict(best_weights)
model.eval()

t_preds, t_targets = [], []
with torch.no_grad():
    for tx, ty in test_loader:
        tx = tx.to(device)
        out = model(tx).softmax(dim=1)[:, 1]
        t_preds.append(out.cpu().numpy())
        t_targets.append(ty.numpy())
t_preds = np.concatenate(t_preds)
t_targets = np.concatenate(t_targets)

# Standard evaluation
m_std = compute_metrics(t_targets, t_preds, threshold=0.5)

# Best threshold on validation set
best_t = 0.5
best_v_acc = -1.0
for t in np.linspace(0.1, 0.9, 81):
    vm = compute_metrics(v_targets, v_preds, threshold=t)
    if vm["accuracy"] > best_v_acc:
        best_v_acc = vm["accuracy"]
        best_t = t

m_cal = compute_metrics(t_targets, t_preds, threshold=best_t)

print(f"\n--- SEED {seed} TEST RESULTS ---", flush=True)
print(f"Standard (t=0.50): Acc={m_std['accuracy']*100:.2f}%, Spec={m_std['specificity']*100:.2f}%, Sens={m_std['sensitivity']*100:.2f}%, AUC={m_std['roc_auc']:.4f}", flush=True)
print(f"Calibrated (t={best_t:.2f}): Acc={m_cal['accuracy']*100:.2f}%, Spec={m_cal['specificity']*100:.2f}%, Sens={m_cal['sensitivity']*100:.2f}%, AUC={m_cal['roc_auc']:.4f}", flush=True)
