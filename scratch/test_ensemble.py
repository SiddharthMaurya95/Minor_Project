import os, sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from models.physsl_net import PhySSLNet
from data.dataset import build_dataloaders
from evaluation.metrics import compute_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"
loaders = build_dataloaders("pneumoniamnist", batch_size=64, seed=42)
test_loader = loaders["test_loader"]
val_loader = loaders["val_loader"]

val_preds = []
test_preds = []
val_targets = []
test_targets = []

for seed in [42, 123, 456, 789, 999]:
    ckpt = torch.load(f"checkpoints/physsl_net_pneumoniamnist_frac100_seed{seed}_best.pth", map_location=device)
    model = PhySSLNet(in_channels=1, num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Val
    vp = []
    with torch.no_grad():
        for x, y in val_loader:
            out = model(x.to(device)).softmax(dim=1)[:, 1]
            vp.append(out.cpu().numpy())
            if len(val_targets) < 524:
                val_targets.append(y.numpy())
    val_preds.append(np.concatenate(vp))

    # Test
    tp = []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(device)).softmax(dim=1)[:, 1]
            tp.append(out.cpu().numpy())
            if len(test_targets) < 624:
                test_targets.append(y.numpy())
    test_preds.append(np.concatenate(tp))

v_targets = np.concatenate(val_targets[:len(val_loader)])
t_targets = np.concatenate(test_targets[:len(test_loader)])

ens_val = np.mean(val_preds, axis=0)
ens_test = np.mean(test_preds, axis=0)

m_def = compute_metrics(t_targets, ens_test, threshold=0.5)

# Tune on val
best_t = 0.5
best_v_score = 0.0
for t in np.linspace(0.1, 0.9, 81):
    vm = compute_metrics(v_targets, ens_val, threshold=t)
    score = vm["recall"] + vm.get("specificity", 0.0)
    if score > best_v_score:
        best_v_score = score
        best_t = t

m_tun = compute_metrics(t_targets, ens_test, threshold=best_t)

print("--- 5-SEED PHYSSL-NET ENSEMBLE RESULTS ---")
print(f"Default (t=0.50): Accuracy={m_def['accuracy']*100:.2f}%, AUC={m_def['roc_auc']:.4f}, Sens={m_def['sensitivity']*100:.2f}%, Spec={m_def['specificity']*100:.2f}%, F1={m_def['f1']:.4f}")
print(f"Val-Tuned (t={best_t:.2f}): Accuracy={m_tun['accuracy']*100:.2f}%, AUC={m_tun['roc_auc']:.4f}, Sens={m_tun['sensitivity']*100:.2f}%, Spec={m_tun['specificity']*100:.2f}%, F1={m_tun['f1']:.4f}")
