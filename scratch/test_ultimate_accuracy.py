import os, sys, torch
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
print(f"Running ultimate accuracy pipeline on: {device}")

hflip = T.RandomHorizontalFlip(p=0.5)

for seed in [42, 123, 456, 789, 999]:
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

    model = PhySSLNet(in_channels=1, num_classes=2).to(device)
    ckpt = torch.load("checkpoints/physsl_simsiam_pretrained.pth", map_location=device)
    mdict = model.state_dict()
    filtered = {k: v for k, v in ckpt["encoder_state_dict"].items() if k in mdict and v.shape == mdict[k].shape}
    mdict.update(filtered)
    model.load_state_dict(mdict)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 30
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_score = -1.0
    best_weights = None
    best_ep = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Apply horizontal flip
            x_aug = hflip(x)
            optimizer.zero_grad()
            loss = criterion(model(x_aug), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
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

        # Monitor validation accuracy
        val_score = val_m["accuracy"]
        if val_score > best_val_score:
            best_val_score = val_score
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = epoch

    # Evaluate best model on test set
    model.load_state_dict(best_weights)
    model.eval()

    t_preds_std, t_preds_tta, t_targets = [], [], []
    with torch.no_grad():
        for tx, ty in test_loader:
            tx = tx.to(device)
            p_orig = model(tx).softmax(dim=1)[:, 1]
            p_flip = model(torch.flip(tx, dims=[-1])).softmax(dim=1)[:, 1]
            p_tta = 0.5 * (p_orig + p_flip)
            t_preds_std.append(p_orig.cpu().numpy())
            t_preds_tta.append(p_tta.cpu().numpy())
            t_targets.append(ty.numpy())

    t_preds_std = np.concatenate(t_preds_std)
    t_preds_tta = np.concatenate(t_preds_tta)
    t_targets = np.concatenate(t_targets)

    m_std = compute_metrics(t_targets, t_preds_std, threshold=0.5)
    m_tta = compute_metrics(t_targets, t_preds_tta, threshold=0.5)

    print(f"Seed {seed:3d} (Best Epoch {best_ep:2d}, Val Acc {best_val_score*100:.2f}%):")
    print(f"   Standard Test: Acc = {m_std['accuracy']*100:.2f}%, Spec = {m_std['specificity']*100:.2f}%, Sens = {m_std['sensitivity']*100:.2f}%, AUC = {m_std['roc_auc']:.4f}")
    print(f"   TTA Test:      Acc = {m_tta['accuracy']*100:.2f}%, Spec = {m_tta['specificity']*100:.2f}%, Sens = {m_tta['sensitivity']*100:.2f}%, AUC = {m_tta['roc_auc']:.4f}")
