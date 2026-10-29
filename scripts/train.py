"""
Reusable Supervised Training & Fine-Tuning Engine for All Models.
Supports:
- PhySSLNet (with optional SSL pretrained encoder initialization)
- Architectural ablations (no_rbf, no_bandpass, no_curvature, no_token_mixer)
- Baselines (CMixer, VanillaCNN, VGG16Small, ResNet50V2Small)
- Strictly validation-driven checkpoint selection (Zero test leakage)
- Label fractions (1.0, 0.5, 0.25, 0.10)
- Fixed reproducible random seed
"""

import os
import sys
import json
import time
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import torchvision.transforms as T

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import build_dataloaders
from models.physsl_net import PhySSLNet
from models.c_mixer import CMixer
from models.baselines import VanillaCNN, VGG16Small, ResNet50V2Small
from evaluation.metrics import compute_metrics


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def build_model(model_name, in_channels, num_classes, dropout=0.2):
    model_name = model_name.lower()
    if model_name == "physsl_net" or model_name == "physslnet":
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
    elif model_name == "no_rbf":
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, use_rbf=False, dropout=dropout)
    elif model_name == "no_bandpass":
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, use_bandpass=False, dropout=dropout)
    elif model_name == "no_curvature":
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, use_curvature=False, dropout=dropout)
    elif model_name == "no_token_mixer":
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, use_token_mixer=False, dropout=dropout)
    elif model_name == "c_mixer" or model_name == "cmixer":
        return CMixer(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
    elif model_name == "vanilla_cnn":
        return VanillaCNN(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
    elif model_name == "vgg16":
        return VGG16Small(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
    elif model_name == "resnet50v2":
        return ResNet50V2Small(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def evaluate_loader(model, loader, device, criterion, num_classes):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.cpu().numpy())

    total_samples = len(loader.dataset)
    avg_loss = total_loss / max(1, total_samples)
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs, axis=0)

    metrics = compute_metrics(
        y_true=all_targets,
        y_pred_prob=all_probs,
        is_multiclass=(num_classes > 2),
    )
    metrics['loss'] = float(avg_loss)
    return metrics, all_targets, all_probs


def train_model(
    model_name="physsl_net",
    dataset_name="pneumoniamnist",
    data_dir="./data",
    epochs=25,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    optimizer_type=None,
    scheduler_type=None,
    label_fraction=1.0,
    seed=42,
    pretrained_path=None,
    early_stopping_patience=8,
    class_weighted=False,
    val_metric="balanced_acc",
    hflip_aug=True,
    save_dir="./checkpoints",
    log_dir="./results",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1. Load model-specific recipe from configs/experiment.yaml if available
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'experiment.yaml')
    recipe = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
                recipes = cfg.get('model_recipes', {})
                recipe = recipes.get(model_name.lower(), {})
        except Exception:
            pass

    # Use recipe defaults when arguments are default or unspecified
    opt_name = (optimizer_type or recipe.get('optimizer', 'adamw')).lower()
    sched_name = (scheduler_type or recipe.get('scheduler', 'cosine')).lower()
    effective_lr = recipe.get('lr', lr) if lr == 1e-3 and 'lr' in recipe else lr
    effective_wd = recipe.get('weight_decay', weight_decay) if weight_decay == 1e-4 and 'weight_decay' in recipe else weight_decay

    loaders = build_dataloaders(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=batch_size,
        label_fraction=label_fraction,
        seed=seed,
    )
    train_loader = loaders['train_loader']
    val_loader = loaders['val_loader']
    in_channels = loaders['info']['in_channels']
    num_classes = loaders['info']['num_classes']

    model = build_model(model_name, in_channels, num_classes).to(device)

    # Optional: Load pretrained SSL weights
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"[+] Loading SSL Pretrained Encoder from: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=device)
        state_dict = ckpt.get('encoder_state_dict', ckpt)
        # Load backbone parameters, ignoring classification head if shape differs
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f"    Loaded {len(filtered_dict)}/{len(model_dict)} parameter tensors into {model_name}.")

    if class_weighted:
        print("[+] Applying inverse class-frequency weighting to CrossEntropyLoss...")
        targets = np.array([y for _, y in train_loader.dataset])
        classes, counts = np.unique(targets, return_counts=True)
        total_samples = len(targets)
        weights = total_samples / (len(classes) * counts.astype(np.float32))
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"    Computed class weights for {len(classes)} classes: {weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Configure Optimizer according to recipe
    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=effective_lr, weight_decay=effective_wd)
    elif opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=effective_lr, momentum=0.9, weight_decay=effective_wd)
    else: # default adamw
        optimizer = optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=effective_wd)

    # Configure Learning Rate Scheduler according to recipe
    if sched_name == "multistep":
        milestones = recipe.get('milestones', [50, 75])
        gamma = recipe.get('gamma', 0.1)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    else: # default cosine
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_score = -999.0
    best_epoch = 0
    patience_counter = 0

    run_tag = f"{model_name}_{dataset_name}_frac{int(label_fraction*100)}_seed{seed}"
    ckpt_save_path = os.path.join(save_dir, f"{run_tag}_best.pth")
    log_save_path = os.path.join(log_dir, f"{run_tag}_log.json")

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
    }

    hflip = T.RandomHorizontalFlip(p=0.5) if hflip_aug else None

    start_time = time.time()
    print(f"\n=======================================================")
    print(f" Supervised Training: {model_name.upper()}")
    print(f" Dataset : {dataset_name} | Label Fraction: {label_fraction:.0%} ({loaders['info']['labelled_train']} samples)")
    print(f" Epochs  : {epochs} | Seed: {seed} | Device: {device.upper()}")
    print(f" Val Metric: {val_metric} | HFlip Aug: {hflip_aug} | Class Weighted: {class_weighted}")
    print(f"=======================================================")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        num_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            if hflip is not None:
                x = hflip(x)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)

            if not torch.isfinite(loss):
                print(f"[ERROR] Non-finite loss encountered at epoch {epoch}: {loss.item()}")
                sys.exit(1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / max(1, num_batches)
        val_metrics, _, _ = evaluate_loader(model, val_loader, device, criterion, num_classes)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['roc_auc'])
        scheduler.step()

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val ACC: {val_metrics['accuracy']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f}"
        )

        # Checkpoint selection strictly based on validation performance
        if val_metric == 'balanced_acc':
            val_score = 0.5 * (val_metrics['recall'] + val_metrics.get('specificity', 0.0))
        elif val_metric == 'accuracy':
            val_score = val_metrics['accuracy']
        elif val_metric == 'loss':
            val_score = -val_metrics['loss']
        else: # default roc_auc
            val_score = val_metrics['roc_auc']

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_metrics': val_metrics,
                'model_name': model_name,
                'dataset': dataset_name,
                'seed': seed,
                'label_fraction': label_fraction,
                'val_metric': val_metric,
                'best_val_score': val_score,
            }, ckpt_save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"[*] Early stopping triggered at epoch {epoch} (no val {val_metric} improvement for {early_stopping_patience} epochs).")
                break

    total_duration = time.time() - start_time
    print(f"\n[+] Training completed in {total_duration:.2f}s. Best Val {val_metric}: {best_val_score:.4f} at epoch {best_epoch}.")

    # Write training log
    log_content = {
        'model_name': model_name,
        'dataset': dataset_name,
        'label_fraction': label_fraction,
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'total_duration_sec': total_duration,
        'history': history,
        'checkpoint_path': ckpt_save_path,
    }
    with open(log_save_path, 'w') as f:
        json.dump(log_content, f, indent=2)

    return model, log_content, ckpt_save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="physsl_net")
    parser.add_argument("--dataset", type=str, default="pneumoniamnist")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--label_fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--class_weighted", action="store_true", help="Apply inverse class frequency weighting")
    parser.add_argument("--val_metric", type=str, default="balanced_acc", choices=["balanced_acc", "accuracy", "roc_auc", "loss"], help="Validation metric for checkpoint selection")
    parser.add_argument("--hflip_aug", action="store_true", default=True, help="Apply random horizontal flip during training")
    parser.add_argument("--no_hflip", dest="hflip_aug", action="store_false", help="Disable random horizontal flip")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(
        model_name=args.model,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        label_fraction=args.label_fraction,
        seed=args.seed,
        pretrained_path=args.pretrained_path,
        early_stopping_patience=args.patience,
        class_weighted=args.class_weighted,
        val_metric=args.val_metric,
        hflip_aug=args.hflip_aug,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        device=device,
    )
