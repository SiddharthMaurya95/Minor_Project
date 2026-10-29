"""
Self-Supervised Learning Pretraining Script (SimSiam Masked Consistency)
Trains PhySSLNet encoder on unlabelled medical images using dual stochastic masked views.
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import build_dataloaders
from models.physsl_net import PhySSLNet, PhySSLSimSiam


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


def pretrain_ssl(
    dataset_name="pneumoniamnist",
    data_dir="./data",
    epochs=15,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    mask_ratio=0.25,
    patch_size=4,
    seed=42,
    resume_from=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_path="./checkpoints/physsl_simsiam_pretrained.pth",
    log_path="./results/ssl_pretrain_log.json",
):
    set_seed(seed)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)

    print(f"\n=======================================================")
    print(f" Starting Self-Supervised Masked Pretraining")
    print(f" Dataset : {dataset_name} | Target Epochs: {epochs} | Batch Size: {batch_size}")
    print(f" Device  : {device.upper()} | Seed: {seed}")
    print(f"=======================================================")

    loaders = build_dataloaders(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=batch_size,
        seed=seed,
        mask_ratio=mask_ratio,
        patch_size=patch_size,
    )
    ssl_loader = loaders['ssl_loader']
    in_channels = loaders['info']['in_channels']

    # Initialize encoder and SSL SimSiam wrapper
    encoder = PhySSLNet(in_channels=in_channels, num_classes=loaders['info']['num_classes'])
    ssl_model = PhySSLSimSiam(encoder=encoder).to(device)

    optimizer = optim.AdamW(ssl_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    start_epoch = 1
    epoch_losses = []

    # Optional checkpoint resumption
    if resume_from and os.path.exists(resume_from):
        print(f"[+] Resuming SSL pretraining from: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        if 'ssl_state_dict' in ckpt:
            ssl_model.load_state_dict(ckpt['ssl_state_dict'])
        elif 'encoder_state_dict' in ckpt:
            encoder.load_state_dict(ckpt['encoder_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
            print(f"    Resuming from epoch {start_epoch}")
        if 'loss_history' in ckpt:
            epoch_losses = list(ckpt['loss_history'])

    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        ssl_model.train()
        total_loss = 0.0
        num_batches = 0

        for v1, v2 in ssl_loader:
            v1 = v1.to(device)
            v2 = v2.to(device)

            optimizer.zero_grad()
            loss = ssl_model(v1, v2)

            if not torch.isfinite(loss):
                print(f"[ERROR] Non-finite loss encountered at epoch {epoch}: {loss.item()}")
                sys.exit(1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ssl_model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        epoch_losses.append(avg_loss)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - SSL Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

    total_duration = time.time() - start_time
    print(f"\n[+] SSL Pretraining Completed in {total_duration:.2f} seconds ({total_duration/60:.2f} mins).")

    # Save encoder weights with full checkpointing for fine-tuning and resumption
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'ssl_state_dict': ssl_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'dataset': dataset_name,
        'epochs': epochs,
        'seed': seed,
        'final_loss': epoch_losses[-1] if epoch_losses else 0.0,
        'loss_history': epoch_losses,
    }, save_path)
    print(f"[+] Pretrained encoder checkpoint saved to: {save_path}")

    # Save training logs
    log_data = {
        'dataset': dataset_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'seed': seed,
        'total_duration_sec': total_duration,
        'loss_history': epoch_losses,
        'checkpoint_path': save_path,
    }
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    return encoder, log_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pneumoniamnist")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="./checkpoints/physsl_simsiam_pretrained.pth")
    parser.add_argument("--log_path", type=str, default="./results/ssl_pretrain_log.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrain_ssl(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=device,
        save_path=args.save_path,
        log_path=args.log_path,
    )
