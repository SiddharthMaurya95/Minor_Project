import os, sys, torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from evaluation.metrics import compute_metrics
from models.physsl_net import PhySSLNet
from data.dataset import build_dataloaders

loaders = build_dataloaders('pneumoniamnist', batch_size=64, seed=42)
test_loader = loaders['test_loader']
val_loader = loaders['val_loader']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_dir = 'checkpoints/optimized'

probs_test = []
probs_val = []
targets_test = []
targets_val = []

for seed in [42, 123, 456, 789, 999]:
    ckpt = torch.load(f'{ckpt_dir}/physsl_net_opt_seed{seed}.pth', map_location=device)
    model = PhySSLNet(in_channels=1, num_classes=2, dropout=0.2).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Val
    v_p, v_y = [], []
    with torch.no_grad():
        for vx, vy in val_loader:
            p1 = model(vx.to(device)).softmax(dim=1)[:, 1]
            p2 = model(torch.flip(vx.to(device), dims=[-1])).softmax(dim=1)[:, 1]
            v_p.append((0.5 * (p1 + p2)).cpu().numpy())
            v_y.append(vy.numpy())
    probs_val.append(np.concatenate(v_p))
    if len(targets_val) == 0:
        targets_val = np.concatenate(v_y)

    # Test
    t_p, t_y = [], []
    with torch.no_grad():
        for tx, ty in test_loader:
            p1 = model(tx.to(device)).softmax(dim=1)[:, 1]
            p2 = model(torch.flip(tx.to(device), dims=[-1])).softmax(dim=1)[:, 1]
            t_p.append((0.5 * (p1 + p2)).cpu().numpy())
            t_y.append(ty.numpy())
    probs_test.append(np.concatenate(t_p))
    if len(targets_test) == 0:
        targets_test = np.concatenate(t_y)

val_ens = np.mean(probs_val, axis=0)
test_ens = np.mean(probs_test, axis=0)

print(f"Test targets: Normal (0) = {np.sum(targets_test == 0)}, Pneumonia (1) = {np.sum(targets_test == 1)}")
print(f"Validation targets: Normal (0) = {np.sum(targets_val == 0)}, Pneumonia (1) = {np.sum(targets_val == 1)}")

# Threshold tuning strictly on validation set:
best_t_val = 0.5
best_val_acc = -1.0
for t in np.linspace(0.30, 0.80, 51):
    vm = compute_metrics(targets_val, val_ens, threshold=t)
    if vm['accuracy'] > best_val_acc:
        best_val_acc = vm['accuracy']
        best_t_val = t

m_test_default = compute_metrics(targets_test, test_ens, threshold=0.50)
m_test_val_tuned = compute_metrics(targets_test, test_ens, threshold=best_t_val)

print("\n--- 5-SEED ENSEMBLE RESULTS ---")
print(f"Default (t=0.50): Acc = {m_test_default['accuracy']*100:.2f}%, Spec = {m_test_default['specificity']*100:.2f}%, Sens = {m_test_default['sensitivity']*100:.2f}%, AUC = {m_test_default['roc_auc']:.4f}")
print(f"Val-Tuned (t={best_t_val:.2f}): Acc = {m_test_val_tuned['accuracy']*100:.2f}%, Spec = {m_test_val_tuned['specificity']*100:.2f}%, Sens = {m_test_val_tuned['sensitivity']*100:.2f}%, AUC = {m_test_val_tuned['roc_auc']:.4f}")

print("\n--- INDIVIDUAL SEEDS (AT t=0.50 and VAL-TUNED) ---")
for i, seed in enumerate([42, 123, 456, 789, 999]):
    p_v = probs_val[i]
    p_t = probs_test[i]
    # Tune on val
    b_t = 0.5
    b_v_acc = 0.0
    for t in np.linspace(0.30, 0.80, 51):
        vm = compute_metrics(targets_val, p_v, threshold=t)
        if vm['accuracy'] > b_v_acc:
            b_v_acc = vm['accuracy']
            b_t = t
    m_def = compute_metrics(targets_test, p_t, threshold=0.50)
    m_tun = compute_metrics(targets_test, p_t, threshold=b_t)
    print(f"Seed {seed:3d}: Std Acc={m_def['accuracy']*100:.2f}%, Spec={m_def['specificity']*100:.2f}% | Tuned(t={b_t:.2f}): Acc={m_tun['accuracy']*100:.2f}%, Spec={m_tun['specificity']*100:.2f}%, AUC={m_def['roc_auc']:.4f}")

# Test set potential threshold scan (for theoretical ceiling analysis)
print("\n--- TEST SET CEILING SCAN (For theoretical frontier documentation) ---")
for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    m = compute_metrics(targets_test, test_ens, threshold=t)
    print(f"t = {t:.2f}: Acc = {m['accuracy']*100:.2f}%, Spec = {m['specificity']*100:.2f}%, Sens = {m['sensitivity']*100:.2f}%")
