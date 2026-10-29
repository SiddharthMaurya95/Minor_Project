import os, sys, torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from models.physsl_net import PhySSLNet, SpatialPyramidHead
from data.dataset import build_dataloaders
from evaluation.metrics import compute_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"

class EnhancedSpatialPyramidHead(nn.Module):
    def __init__(self, in_channels=64, hidden_dim=128, num_classes=2, dropout=0.2):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        total_dim = in_channels * 5 # 320
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def extract_features(self, x):
        p1 = self.pool1(x).flatten(1)
        p2 = self.pool2(x).flatten(1)
        return torch.cat([p1, p2], dim=1)

    def forward(self, x):
        feat = self.extract_features(x)
        return self.classifier(feat)

class EnhancedPhySSLNet(PhySSLNet):
    def __init__(self, in_channels=1, num_classes=2, dropout=0.2):
        super().__init__(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
        self.head = EnhancedSpatialPyramidHead(in_channels=64, hidden_dim=128, num_classes=num_classes, dropout=dropout)

print(f"Testing EnhancedPhySSLNet on {device}...")
model_enh = EnhancedPhySSLNet(in_channels=1, num_classes=2).to(device)
print(f"Parameters: {sum(p.numel() for p in model_enh.parameters() if p.requires_grad):,}")

# Test on Seed 42 and Seed 123
for seed in [42, 123, 789]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    loaders = build_dataloaders("pneumoniamnist", batch_size=64, seed=seed)
    train_loader, val_loader, test_loader = loaders["train_loader"], loaders["val_loader"], loaders["test_loader"]
    
    model = EnhancedPhySSLNet(in_channels=1, num_classes=2).to(device)
    # Load pretrained SSL backbone
    ckpt = torch.load("checkpoints/physsl_simsiam_pretrained.pth", map_location=device)
    mdict = model.state_dict()
    filtered = {k: v for k, v in ckpt["encoder_state_dict"].items() if k in mdict and v.shape == mdict[k].shape}
    mdict.update(filtered)
    model.load_state_dict(mdict)

    # Class weights
    targets = np.array([y for _, y in train_loader.dataset])
    c0 = np.sum(targets == 0)
    c1 = np.sum(targets == 1)
    w0 = (c0 + c1) / (2.0 * c0)
    w1 = (c0 + c1) / (2.0 * c1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 25
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_score = -1.0
    best_weights = None
    best_ep = 0

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

        model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device)
                v_preds.append(model(vx).softmax(dim=1)[:, 1].cpu().numpy())
                v_targets.append(vy.numpy())
        v_preds = np.concatenate(v_preds)
        v_targets = np.concatenate(v_targets)
        val_m = compute_metrics(v_targets, v_preds, threshold=0.5)

        # Monitor val balanced accuracy
        score = 0.5 * (val_m["recall"] + val_m.get("specificity", 0.0))
        if score > best_val_score:
            best_val_score = score
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = epoch

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

    m = compute_metrics(t_targets, t_preds, threshold=0.5)
    print(f"Seed {seed:3d} (Epoch {best_ep:2d}): Test Acc = {m['accuracy']*100:.2f}%, Spec = {m['specificity']*100:.2f}%, Sens = {m['sensitivity']*100:.2f}%, AUC = {m['roc_auc']:.4f}")
