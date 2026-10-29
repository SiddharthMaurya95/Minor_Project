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
print(f"Executing 5-Seed Benchmark on: {device}", flush=True)

seeds = [42, 123, 456, 789, 999]
ckpt_dir = "checkpoints/optimized"
os.makedirs(ckpt_dir, exist_ok=True)

hflip = T.RandomHorizontalFlip(p=0.5)

std_metrics_list = []
tta_metrics_list = []
all_test_probs_std = []
all_test_probs_tta = []
test_targets_global = None

for seed in seeds:
    t0 = time.time()
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

    model = PhySSLNet(in_channels=1, num_classes=2, dropout=0.2).to(device)

    # Load SSL Pretrained weights
    ckpt = torch.load("checkpoints/physsl_simsiam_pretrained.pth", map_location=device)
    mdict = model.state_dict()
    filtered = {k: v for k, v in ckpt["encoder_state_dict"].items() if k in mdict and v.shape == mdict[k].shape}
    mdict.update(filtered)
    model.load_state_dict(mdict)

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
                vx = vx.to(device)
                v_preds.append(model(vx).softmax(dim=1)[:, 1].cpu().numpy())
                v_targets.append(vy.numpy())
        v_preds = np.concatenate(v_preds)
        v_targets = np.concatenate(v_targets)
        vm = compute_metrics(v_targets, v_preds, threshold=0.5)

        # Selection on balanced accuracy
        score = 0.5 * (vm["recall"] + vm.get("specificity", 0.0))
        if score > best_val_score:
            best_val_score = score
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = epoch

    # Save optimized checkpoint
    save_path = os.path.join(ckpt_dir, f"physsl_net_opt_seed{seed}.pth")
    torch.save({
        'model_state_dict': best_weights,
        'seed': seed,
        'best_epoch': best_ep,
        'best_val_score': best_val_score,
    }, save_path)

    # Evaluate on test set
    model.load_state_dict(best_weights)
    model.eval()

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

    if test_targets_global is None:
        test_targets_global = t_targets

    all_test_probs_std.append(t_preds_std)
    all_test_probs_tta.append(t_preds_tta)

    m_std = compute_metrics(t_targets, t_preds_std, threshold=0.5)
    m_tta = compute_metrics(t_targets, t_preds_tta, threshold=0.5)

    std_metrics_list.append(m_std)
    tta_metrics_list.append(m_tta)

    elapsed = time.time() - t0
    print(f"Seed {seed:3d} (ep {best_ep:2d}, {elapsed:.1f}s):", flush=True)
    print(f"   Standard (t=0.50): Acc={m_std['accuracy']*100:.2f}%, Spec={m_std['specificity']*100:.2f}%, Sens={m_std['sensitivity']*100:.2f}%, AUC={m_std['roc_auc']:.4f}", flush=True)
    print(f"   TTA      (t=0.50): Acc={m_tta['accuracy']*100:.2f}%, Spec={m_tta['specificity']*100:.2f}%, Sens={m_tta['sensitivity']*100:.2f}%, AUC={m_tta['roc_auc']:.4f}", flush=True)

# Compute multi-seed statistics
def print_summary(metrics_list, name):
    accs = [m['accuracy'] * 100 for m in metrics_list]
    specs = [m['specificity'] * 100 for m in metrics_list]
    senss = [m['sensitivity'] * 100 for m in metrics_list]
    aucs = [m['roc_auc'] for m in metrics_list]
    f1s = [m['f1'] for m in metrics_list]
    print(f"\n=======================================================", flush=True)
    print(f" {name.upper()} RESULTS (5 SEEDS)", flush=True)
    print(f"=======================================================", flush=True)
    print(f" Accuracy:    {np.mean(accs):.2f}% ± {np.std(accs):.2f}% (range: {np.min(accs):.2f}% - {np.max(accs):.2f}%)", flush=True)
    print(f" Specificity: {np.mean(specs):.2f}% ± {np.std(specs):.2f}%", flush=True)
    print(f" Sensitivity: {np.mean(senss):.2f}% ± {np.std(senss):.2f}%", flush=True)
    print(f" ROC-AUC:     {np.mean(aucs):.4f} ± {np.std(aucs):.4f}", flush=True)
    print(f" Macro-F1:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}", flush=True)

print_summary(std_metrics_list, "Standard Evaluation")
print_summary(tta_metrics_list, "Test-Time Augmentation (TTA)")

# 5-Seed Ensemble
ens_probs_std = np.mean(all_test_probs_std, axis=0)
ens_probs_tta = np.mean(all_test_probs_tta, axis=0)

m_ens_std = compute_metrics(test_targets_global, ens_probs_std, threshold=0.5)
m_ens_tta = compute_metrics(test_targets_global, ens_probs_tta, threshold=0.5)

print(f"\n=======================================================", flush=True)
print(f" 5-SEED ENSEMBLE RESULTS", flush=True)
print(f"=======================================================", flush=True)
print(f" Standard Ensemble: Acc = {m_ens_std['accuracy']*100:.2f}%, Spec = {m_ens_std['specificity']*100:.2f}%, Sens = {m_ens_std['sensitivity']*100:.2f}%, AUC = {m_ens_std['roc_auc']:.4f}", flush=True)
print(f" TTA Ensemble:      Acc = {m_ens_tta['accuracy']*100:.2f}%, Spec = {m_ens_tta['specificity']*100:.2f}%, Sens = {m_ens_tta['sensitivity']*100:.2f}%, AUC = {m_ens_tta['roc_auc']:.4f}", flush=True)
