import os, sys, time, torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from models.physsl_net import PhySSLNet, SpatialPyramidHead
from data.dataset import build_dataloaders
from evaluation.metrics import compute_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

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

def run_experiment(head_type="standard", lr_backbone=1e-4, lr_head=1e-3, use_aug=True, use_weights=True, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    loaders = build_dataloaders("pneumoniamnist", batch_size=64, seed=seed)
    train_loader, val_loader, test_loader = loaders["train_loader"], loaders["val_loader"], loaders["test_loader"]

    if head_type == "enhanced":
        model = PhySSLNet(in_channels=1, num_classes=2, dropout=0.2).to(device)
        model.head = EnhancedSpatialPyramidHead(in_channels=64, hidden_dim=128, num_classes=2, dropout=0.2).to(device)
    else:
        model = PhySSLNet(in_channels=1, num_classes=2, dropout=0.2).to(device)

    # Load SSL weights
    ckpt = torch.load("checkpoints/physsl_simsiam_pretrained.pth", map_location=device)
    state = ckpt.get("encoder_state_dict", ckpt)
    mdict = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in mdict and v.shape == mdict[k].shape}
    mdict.update(filtered)
    model.load_state_dict(mdict)

    # Loss
    if use_weights:
        targets = np.array([y for _, y in train_loader.dataset])
        c0, c1 = np.sum(targets == 0), np.sum(targets == 1)
        w0 = (c0 + c1) / (2.0 * c0)
        w1 = (c0 + c1) / (2.0 * c1)
        class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Differential LR
    head_params = list(model.head.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone, 'weight_decay': 1e-4},
        {'params': head_params, 'lr': lr_head, 'weight_decay': 1e-4},
    ])

    epochs = 25
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    hflip = T.RandomHorizontalFlip(p=0.5) if use_aug else None

    best_val_acc = -1.0
    best_weights = None
    best_ep = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if hflip is not None:
                x = hflip(x)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Val evaluation
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
        val_score = 0.5 * (val_m["recall"] + val_m.get("specificity", 0.0))
        if val_score > best_val_acc:
            best_val_acc = val_score
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = epoch

    model.load_state_dict(best_weights)
    model.eval()

    # Val predictions on best weights
    v_preds_best, v_targets_best = [], []
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(device)
            v_preds_best.append(model(vx).softmax(dim=1)[:, 1].cpu().numpy())
            v_targets_best.append(vy.numpy())
    v_preds_best = np.concatenate(v_preds_best)
    v_targets_best = np.concatenate(v_targets_best)

    # Test evaluation
    t_preds_std, t_preds_tta, t_targets = [], [], []
    with torch.no_grad():
        for tx, ty in test_loader:
            tx = tx.to(device)
            p1 = model(tx).softmax(dim=1)[:, 1]
            p2 = model(torch.flip(tx, dims=[-1])).softmax(dim=1)[:, 1]
            p_tta = 0.5 * (p1 + p2)
            t_preds_std.append(p1.cpu().numpy())
            t_preds_tta.append(p_tta.cpu().numpy())
            t_targets.append(ty.numpy())

    t_preds_std = np.concatenate(t_preds_std)
    t_preds_tta = np.concatenate(t_preds_tta)
    t_targets = np.concatenate(t_targets)

    # 1. Standard (t=0.50)
    m_std = compute_metrics(t_targets, t_preds_std, threshold=0.5)
    # 2. TTA (t=0.50)
    m_tta = compute_metrics(t_targets, t_preds_tta, threshold=0.5)

    # 3. Val-calibrated threshold on Accuracy
    best_t_acc = 0.5
    best_acc_score = -1.0
    for t in np.linspace(0.1, 0.9, 81):
        vm = compute_metrics(v_targets_best, v_preds_best, threshold=t)
        if vm["accuracy"] > best_acc_score:
            best_acc_score = vm["accuracy"]
            best_t_acc = t
    m_cal_acc = compute_metrics(t_targets, t_preds_tta, threshold=best_t_acc)

    # 4. Val-calibrated threshold on Balanced Accuracy
    best_t_bal = 0.5
    best_bal_score = -1.0
    for t in np.linspace(0.1, 0.9, 81):
        vm = compute_metrics(v_targets_best, v_preds_best, threshold=t)
        score = 0.5 * (vm["recall"] + vm.get("specificity", 0.0))
        if score > best_bal_score:
            best_bal_score = score
            best_t_bal = t
    m_cal_bal = compute_metrics(t_targets, t_preds_tta, threshold=best_t_bal)

    print(f"[{head_type.upper():8s} | lr_b={lr_backbone:.0e}, lr_h={lr_head:.0e} | seed={seed} | ep={best_ep:2d}]:", flush=True)
    print(f"   Std (t=0.50):    Acc = {m_std['accuracy']*100:.2f}%, Spec = {m_std['specificity']*100:.2f}%, Sens = {m_std['sensitivity']*100:.2f}%, AUC = {m_std['roc_auc']:.4f}", flush=True)
    print(f"   TTA (t=0.50):    Acc = {m_tta['accuracy']*100:.2f}%, Spec = {m_tta['specificity']*100:.2f}%, Sens = {m_tta['sensitivity']*100:.2f}%, AUC = {m_tta['roc_auc']:.4f}", flush=True)
    print(f"   TTA+ValAcc(t={best_t_acc:.2f}): Acc = {m_cal_acc['accuracy']*100:.2f}%, Spec = {m_cal_acc['specificity']*100:.2f}%, Sens = {m_cal_acc['sensitivity']*100:.2f}%", flush=True)
    print(f"   TTA+ValBal(t={best_t_bal:.2f}): Acc = {m_cal_bal['accuracy']*100:.2f}%, Spec = {m_cal_bal['specificity']*100:.2f}%, Sens = {m_cal_bal['sensitivity']*100:.2f}%", flush=True)
    return m_std, m_tta, m_cal_acc

# Run grid on seed 42 and seed 123
print("\n--- EXPERIMENT 1: STANDARD HEAD, UNIFORM LR (1e-3) ---", flush=True)
run_experiment(head_type="standard", lr_backbone=1e-3, lr_head=1e-3, seed=42)
run_experiment(head_type="standard", lr_backbone=1e-3, lr_head=1e-3, seed=123)

print("\n--- EXPERIMENT 2: STANDARD HEAD, DIFFERENTIAL LR (backbone 1e-4, head 1e-3) ---", flush=True)
run_experiment(head_type="standard", lr_backbone=1e-4, lr_head=1e-3, seed=42)
run_experiment(head_type="standard", lr_backbone=1e-4, lr_head=1e-3, seed=123)

print("\n--- EXPERIMENT 3: ENHANCED HEAD, DIFFERENTIAL LR (backbone 2e-4, head 1e-3) ---", flush=True)
run_experiment(head_type="enhanced", lr_backbone=2e-4, lr_head=1e-3, seed=42)
run_experiment(head_type="enhanced", lr_backbone=2e-4, lr_head=1e-3, seed=123)
