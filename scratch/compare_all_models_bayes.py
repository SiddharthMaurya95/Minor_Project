import os, sys, torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from evaluation.metrics import compute_metrics
from models.physsl_net import PhySSLNet
from models.c_mixer import CMixer
from models.baselines import VanillaCNN, VGG16Small, ResNet50V2Small
from data.dataset import build_dataloaders

loaders = build_dataloaders('pneumoniamnist', batch_size=64, seed=42)
test_loader = loaders['test_loader']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training prior
pi_train = 3883.0 / (3883.0 + 1349.0) # 0.742168
t_bayes = pi_train
print(f"Bayesian Prior Threshold: t* = {t_bayes:.4f}")

def eval_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()
    t_preds, t_targets = [], []
    with torch.no_grad():
        for tx, ty in test_loader:
            p = model(tx.to(device)).softmax(dim=1)[:, 1]
            t_preds.append(p.cpu().numpy())
            t_targets.append(ty.numpy())
    probs = np.concatenate(t_preds)
    targets = np.concatenate(t_targets)
    m_default = compute_metrics(targets, probs, threshold=0.50)
    m_bayes = compute_metrics(targets, probs, threshold=t_bayes)
    return m_default, m_bayes, probs, targets

# 1. Compare PhySSL-Net Optimized across 5 seeds
print("\n=== PHYSSL-NET OPTIMIZED (5 SEEDS) ===")
phy_accs_def, phy_accs_bayes = [], []
phy_aucs = []
phy_probs = []

for seed in [42, 123, 456, 789, 999]:
    ckpt = f"checkpoints/optimized/physsl_net_opt_seed{seed}.pth"
    model = PhySSLNet(in_channels=1, num_classes=2, dropout=0.2).to(device)
    m_def, m_bay, p, targets = eval_model(model, ckpt)
    phy_accs_def.append(m_def['accuracy']*100)
    phy_accs_bayes.append(m_bay['accuracy']*100)
    phy_aucs.append(m_def['roc_auc'])
    phy_probs.append(p)
    print(f"Seed {seed:3d}: Default (t=0.50) Acc = {m_def['accuracy']*100:.2f}%, Spec = {m_def['specificity']*100:.2f}% | Bayes (t={t_bayes:.2f}) Acc = {m_bay['accuracy']*100:.2f}%, Spec = {m_bay['specificity']*100:.2f}%, Sens = {m_bay['sensitivity']*100:.2f}%, AUC = {m_def['roc_auc']:.4f}")

print(f"PhySSL-Net Default Mean: {np.mean(phy_accs_def):.2f}% ± {np.std(phy_accs_def):.2f}%")
print(f"PhySSL-Net Bayes Mean:   {np.mean(phy_accs_bayes):.2f}% ± {np.std(phy_accs_bayes):.2f}%")
print(f"PhySSL-Net Mean AUC:     {np.mean(phy_aucs):.4f} ± {np.std(phy_aucs):.4f}")

# Ensemble
ens_p = np.mean(phy_probs, axis=0)
m_ens_def = compute_metrics(targets, ens_p, threshold=0.50)
m_ens_bay = compute_metrics(targets, ens_p, threshold=t_bayes)
print(f"PhySSL-Net 5-Seed Ensemble: Default Acc = {m_ens_def['accuracy']*100:.2f}% | Bayes Acc = {m_ens_bay['accuracy']*100:.2f}% (Spec={m_ens_bay['specificity']*100:.2f}%, Sens={m_ens_bay['sensitivity']*100:.2f}%, AUC={m_ens_def['roc_auc']:.4f})")

# 2. Check Baselines across 5 seeds with Bayes threshold
def evaluate_baseline(model_cls, name, prefix):
    print(f"\n=== {name.upper()} (5 SEEDS) ===")
    b_accs_def, b_accs_bayes = [], []
    b_aucs = []
    b_probs = []
    for seed in [42, 123, 456, 789, 999]:
        ckpt = f"checkpoints/{prefix}_pneumoniamnist_frac100_seed{seed}_best.pth"
        if not os.path.exists(ckpt):
            continue
        model = model_cls(in_channels=1, num_classes=2, dropout=0.2).to(device)
        m_def, m_bay, p, _ = eval_model(model, ckpt)
        b_accs_def.append(m_def['accuracy']*100)
        b_accs_bayes.append(m_bay['accuracy']*100)
        b_aucs.append(m_def['roc_auc'])
        b_probs.append(p)
        print(f"Seed {seed:3d}: Default Acc = {m_def['accuracy']*100:.2f}% | Bayes Acc = {m_bay['accuracy']*100:.2f}%, Spec = {m_bay['specificity']*100:.2f}%, Sens = {m_bay['sensitivity']*100:.2f}%, AUC = {m_def['roc_auc']:.4f}")
    if len(b_accs_def) > 0:
        print(f"{name} Default Mean: {np.mean(b_accs_def):.2f}% ± {np.std(b_accs_def):.2f}%")
        print(f"{name} Bayes Mean:   {np.mean(b_accs_bayes):.2f}% ± {np.std(b_accs_bayes):.2f}%")
        print(f"{name} Mean AUC:     {np.mean(b_aucs):.4f} ± {np.std(b_aucs):.4f}")
        ens = np.mean(b_probs, axis=0)
        m_e_def = compute_metrics(targets, ens, threshold=0.50)
        m_e_bay = compute_metrics(targets, ens, threshold=t_bayes)
        print(f"{name} Ensemble: Default Acc = {m_e_def['accuracy']*100:.2f}% | Bayes Acc = {m_e_bay['accuracy']*100:.2f}%")

evaluate_baseline(CMixer, "C-Mixer", "c_mixer")
evaluate_baseline(VanillaCNN, "Vanilla CNN", "vanilla_cnn")
evaluate_baseline(VGG16Small, "VGG16", "vgg16")
evaluate_baseline(ResNet50V2Small, "ResNet50V2", "resnet50v2")
