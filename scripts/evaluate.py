import os
import sys
import json
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import build_dataloaders
from models.physsl_net import PhySSLNet
from models.c_mixer import CMixer
from models.baselines import VanillaCNN, VGG16Small, ResNet50V2Small
from evaluation.metrics import compute_metrics, apply_logit_adjustment, find_optimal_threshold


def build_model(model_name, in_channels, num_classes):
    model_name = model_name.lower()
    if "no_rbf" in model_name:
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, use_rbf=False)
    elif "no_bandpass" in model_name:
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, use_bandpass=False)
    elif "no_curvature" in model_name:
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, use_curvature=False)
    elif "no_token_mixer" in model_name:
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, use_token_mixer=False)
    elif "c_mixer" in model_name or "cmixer" in model_name:
        return CMixer(in_channels=in_channels, num_classes=num_classes)
    elif "vanilla_cnn" in model_name:
        return VanillaCNN(in_channels=in_channels, num_classes=num_classes)
    elif "vgg16" in model_name:
        return VGG16Small(in_channels=in_channels, num_classes=num_classes)
    elif "resnet50" in model_name:
        return ResNet50V2Small(in_channels=in_channels, num_classes=num_classes)
    else:
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes)


def evaluate_checkpoint(
    checkpoint_path,
    dataset_name="pneumoniamnist",
    data_dir="./data",
    split="test",
    batch_size=64,
    threshold=0.5,
    tune_threshold=False,
    threshold_metric="accuracy",
    use_tta=False,
    use_logit_adj=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir="./results",
):
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    os.makedirs(output_dir, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt.get('model_name', 'physsl_net')
    seed = ckpt.get('seed', 42)
    label_fraction = ckpt.get('label_fraction', 1.0)

    loaders = build_dataloaders(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=batch_size,
        seed=seed,
    )
    loader = loaders[f"{split}_loader"]
    in_channels = loaders['info']['in_channels']
    num_classes = loaders['info']['num_classes']

    model = build_model(model_name, in_channels, num_classes).to(device)
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state_dict)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Optional Threshold Tuning on Validation Split
    active_threshold = threshold
    if tune_threshold and num_classes == 2 and split != "val":
        print("[+] Tuning decision threshold on validation partition...")
        val_loader = loaders['val_loader']
        val_logits, val_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device)
                vlogits = model(vx)
                val_logits.append(vlogits.cpu().numpy())
                val_targets.append(vy.numpy())
        val_logits = np.concatenate(val_logits, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_probs = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))
        val_probs = val_probs / np.sum(val_probs, axis=1, keepdims=True)
        active_threshold, opt_score = find_optimal_threshold(val_targets, val_probs[:, 1], metric=threshold_metric)
        print(f"    Optimal validation threshold selected: {active_threshold:.4f} (Val {threshold_metric}: {opt_score:.4f})")

    # Warm-up GPU/CPU for reproducible inference timing
    warmup_x = torch.randn(min(16, batch_size), in_channels, 28, 28, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(warmup_x)
    if device == "cuda":
        torch.cuda.synchronize()

    all_logits = []
    all_targets = []

    start_timer = time.perf_counter()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if use_tta:
                # Test-Time Augmentation: average original and horizontally flipped view
                x_flip = torch.flip(x, dims=[3])
                logits = 0.5 * (model(x) + model(x_flip))
            else:
                logits = model(x)

            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.numpy())

    if device == "cuda":
        torch.cuda.synchronize()
    total_eval_time = time.perf_counter() - start_timer

    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    total_samples = len(all_targets)
    ms_per_sample = (total_eval_time / max(1, total_samples)) * 1000.0
    throughput_fps = total_samples / max(1e-6, total_eval_time)

    # Optional Logit Adjustment
    if use_logit_adj:
        class_counts = np.bincount(loaders['train_loader'].dataset.dataset.labels.numpy(), minlength=num_classes)
        all_logits = apply_logit_adjustment(all_logits, class_counts)

    # Convert logits to probabilities
    exp_logits = np.exp(all_logits - np.max(all_logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    metrics = compute_metrics(
        y_true=all_targets,
        y_pred_prob=probs,
        threshold=active_threshold,
        is_multiclass=(num_classes > 2),
    )
    metrics['model_name'] = model_name
    metrics['dataset'] = dataset_name
    metrics['split'] = split
    metrics['seed'] = seed
    metrics['label_fraction'] = label_fraction
    metrics['num_parameters'] = num_params
    metrics['latency_ms_per_sample'] = round(ms_per_sample, 3)
    metrics['throughput_fps'] = round(throughput_fps, 1)
    metrics['threshold'] = round(active_threshold, 4)
    metrics['tta_applied'] = use_tta
    metrics['logit_adj_applied'] = use_logit_adj
    metrics['checkpoint_path'] = checkpoint_path

    print(f"\n=======================================================")
    print(f" Evaluation Results on {split.upper()} set: {model_name.upper()}")
    print(f" Parameters: {num_params:,} | Latency: {ms_per_sample:.2f} ms/sample ({throughput_fps:.1f} fps)")
    print(f" Threshold : {active_threshold:.4f} | TTA: {use_tta} | LogitAdj: {use_logit_adj}")
    print(f" ACC : {metrics['accuracy']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f" F1  : {metrics['f1']:.4f} | Recall (Sens): {metrics['recall']:.4f} | Spec: {metrics.get('specificity', 0.0):.4f}")
    print(f" Confusion Matrix: {metrics['confusion_matrix']}")
    print(f"=======================================================")

    # Export to JSON
    out_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    json_path = os.path.join(output_dir, f"{out_stem}_{split}_eval.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Export to summary CSV
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    csv_exists = os.path.exists(csv_path)
    fieldnames = [
        'model_name', 'dataset', 'split', 'seed', 'label_fraction',
        'accuracy', 'roc_auc', 'f1', 'recall', 'specificity',
        'threshold', 'tta_applied', 'logit_adj_applied',
        'num_parameters', 'latency_ms_per_sample', 'throughput_fps',
        'checkpoint_path'
    ]

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not csv_exists:
            writer.writeheader()
        writer.writerow(metrics)

    return metrics, all_targets, probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate any PhySSL-Net or Baseline checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--dataset", type=str, default="pneumoniamnist", help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Partition split to evaluate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification decision threshold")
    parser.add_argument("--tune_threshold", action="store_true", help="Tune threshold on validation data")
    parser.add_argument("--threshold_metric", type=str, default="accuracy", choices=["accuracy", "balanced_acc", "f1", "youden"], help="Metric to optimize during threshold tuning")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation (horizontal flip)")
    parser.add_argument("--logit_adj", action="store_true", help="Enable post-hoc logit adjustment")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for reports")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        threshold=args.threshold,
        tune_threshold=args.tune_threshold,
        threshold_metric=args.threshold_metric,
        use_tta=args.tta,
        use_logit_adj=args.logit_adj,
        device=device,
        output_dir=args.output_dir,
    )
