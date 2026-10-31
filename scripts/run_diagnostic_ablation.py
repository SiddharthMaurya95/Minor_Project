"""
Diagnostic Ablation Runner for PathMNIST and BreastMNIST.
Tests:
1. Full PhySSL-Net (Original Baseline with IntensityRBF)
2. PhySSL-Net with AdaptiveSplineGaussianRBF (New Unified Basis)
3. PhySSL-Net w/o Curvature Attention (A_curv bypassed via identity x = x)
4. PhySSL-Net w/o Token Mixer (Spatial MLP bypassed via identity x = x)
5. PhySSL-Net w/o BandPass (Residual modulus bypassed via identity x = x)
6. Vanilla CNN + Gaussian RBF (Vanilla CNN backbone with Gaussian RBF encoding)
7. Vanilla CNN (Standard reference baseline)
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import build_dataloaders
from scripts.train import train_model
from scripts.evaluate import evaluate_checkpoint

def run_diagnostic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running Diagnostic Ablation on Device: {device.upper()}", flush=True)

    datasets = ["breastmnist", "pathmnist"]
    models = [
        ("PhySSL-Net (Orig Intensity-RBF)", "physsl_intensity_rbf"),
        ("PhySSL-Net (ASG-RBF)", "physsl_asg_rbf"),
        ("w/o Curvature Attention", "no_curvature"),
        ("w/o Token Mixer", "no_token_mixer"),
        ("w/o BandPass", "no_bandpass"),
        ("Vanilla CNN + RBF", "vanilla_cnn_rbf"),
        ("Vanilla CNN", "vanilla_cnn"),
    ]

    all_results = []
    results_json_path = "./results/diagnostic_ablation_results.json"
    results_csv_path = "./results/diagnostic_ablation_results.csv"

    # Resume if partial results exist
    if os.path.exists(results_json_path):
        try:
            with open(results_json_path, "r") as f:
                all_results = json.load(f)
            print(f"[+] Loaded {len(all_results)} existing records from {results_json_path}", flush=True)
        except Exception:
            all_results = []

    os.makedirs("./results/diagnostic", exist_ok=True)
    os.makedirs("./checkpoints/diagnostic", exist_ok=True)

    for d_name in datasets:
        print(f"\n{'='*70}\n DATASET: {d_name.upper()}\n{'='*70}", flush=True)
        epochs = 15 if d_name == "breastmnist" else 10
        batch_size = 64 if d_name == "breastmnist" else 256

        for display_name, m_key in models:
            # Check if already completed
            already_done = any(r.get('dataset') == d_name and r.get('model_key') == m_key for r in all_results)
            if already_done:
                print(f"[-] Skipping already evaluated {display_name} ({m_key}) on {d_name}.", flush=True)
                continue

            print(f"\n[+] Testing {display_name} ({m_key}) on {d_name} ({epochs} epochs, batch_size={batch_size})...", flush=True)
            t0 = time.time()
            model, log_content, ckpt_path = train_model(
                model_name=m_key,
                dataset_name=d_name,
                epochs=epochs,
                batch_size=batch_size,
                seed=42,
                device=device,
                save_dir="./checkpoints/diagnostic",
                log_dir="./results/diagnostic",
            )
            train_time = time.time() - t0

            # Evaluate on held-out test split
            eval_metrics, y_true, y_probs = evaluate_checkpoint(
                checkpoint_path=ckpt_path,
                dataset_name=d_name,
                split="test",
                device=device,
                output_dir="./results/diagnostic",
            )

            spec = eval_metrics.get('specificity')
            spec_val = round(spec * 100, 2) if spec is not None else 0.0

            record = {
                'dataset': d_name,
                'model_display': display_name,
                'model_key': m_key,
                'test_accuracy': round(eval_metrics.get('accuracy', 0.0) * 100, 2),
                'test_roc_auc': round(eval_metrics.get('roc_auc', 0.0), 4),
                'test_f1': round(eval_metrics.get('f1', 0.0), 4),
                'test_recall': round(eval_metrics.get('recall', 0.0) * 100, 2),
                'test_specificity': spec_val,
                'parameters': eval_metrics['num_parameters'],
                'train_time_sec': round(train_time, 1),
            }
            all_results.append(record)
            print(f"    --> {display_name} on {d_name}: Acc = {record['test_accuracy']}% | AUC = {record['test_roc_auc']} | Params = {record['parameters']:,} | Time = {record['train_time_sec']}s", flush=True)

            # Persist incrementally
            with open(results_json_path, "w") as f:
                json.dump(all_results, f, indent=2)
            df = pd.DataFrame(all_results)
            df.to_csv(results_csv_path, index=False)

    print("\n" + "="*80, flush=True)
    print(" FINAL DIAGNOSTIC ABLATION SUMMARY TABLE", flush=True)
    print("="*80, flush=True)
    df = pd.DataFrame(all_results)
    pivot = df.pivot(index='model_display', columns='dataset', values='test_accuracy')
    print(pivot.to_string(), flush=True)
    print("="*80, flush=True)

if __name__ == "__main__":
    run_diagnostic()
