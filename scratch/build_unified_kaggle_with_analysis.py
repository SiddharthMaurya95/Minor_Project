"""
Builds the unified kaggle_benchmark.py and kaggle_physsl_medmnist_benchmark.ipynb
with:
1. Multi-Seed Benchmark Execution (loops through configurable seeds e.g. [42, 123, 456] or [42, 123, 456, 789, 999])
2. Multi-Seed Statistical Aggregation (calculates Mean ± SD for accuracy, balanced acc, ROC-AUC, F1, precision, recall, specificity)
3. Direct display of Per-Seed Table AND Multi-Seed Mean ± SD Summary Table
4. Complete Research Visualisation, Statistical Analysis, and Results Generation pipeline embedded directly into the single Kaggle notebook and script.
"""

import os
import json
import ast

os.makedirs("kaggle", exist_ok=True)

# Helper to split text into lines with keepends=True for clean nbformat
def to_cell_source(text):
    return text.splitlines(keepends=True)

# Read generate_unified_kaggle.py to reuse models, datasets, header code
with open("C:/Users/maury/.gemini/antigravity/brain/6b6e073f-62c2-4fcd-a6ff-86f3e24925a6/scratch/generate_unified_kaggle.py", "r", encoding="utf-8") as f:
    orig_gen = f.read()

# Extract header_code, models_code, datasets_code
header_code = orig_gen.split("models_code = '''")[0].split("header_code = '''")[1].rstrip("'''\n")
models_code = orig_gen.split("models_code = '''")[1].split("datasets_code = '''")[0].rstrip("'''\n")
datasets_code = orig_gen.split("datasets_code = '''")[1].split("metrics_training_code = '''")[0].rstrip("'''\n")

# Define updated metrics_training_code supporting explicit seed and output logging
metrics_training_code = '''# =====================================================================
# 3. METRICS & 2-PHASE TRAINING ENGINE (PRETRAIN SSL + FINE-TUNING)
# =====================================================================

def compute_all_metrics(y_true, y_pred_prob, is_multiclass=False, is_multilabel=False):
    y_true = np.asarray(y_true).squeeze()
    y_pred_prob = np.asarray(y_pred_prob)

    if is_multilabel:
        try:
            auc_score = roc_auc_score(y_true, y_pred_prob, average='macro')
        except Exception:
            auc_score = 0.5
        preds = (y_pred_prob >= 0.5).astype(int)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average='macro', zero_division=0)
        rec = recall_score(y_true, preds, average='macro', zero_division=0)
        prec = precision_score(y_true, preds, average='macro', zero_division=0)
        return {
            'accuracy': float(acc), 'roc_auc': float(auc_score), 'f1': float(f1),
            'recall': float(rec), 'precision': float(prec), 'specificity': 0.0,
            'balanced_acc': float(acc)
        }

    if not is_multiclass:
        scores = y_pred_prob[:, 1] if y_pred_prob.ndim == 2 else y_pred_prob
        y_pred = (scores >= 0.5).astype(int)
        acc = float(accuracy_score(y_true, y_pred))
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc_score = float(auc(fpr, tpr))
        except Exception:
            auc_score = 0.5
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        bal_acc = float(0.5 * (rec + spec))
        return {
            'accuracy': acc, 'roc_auc': auc_score, 'f1': f1, 'recall': rec,
            'precision': prec, 'specificity': spec, 'balanced_acc': bal_acc,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }
    else:
        y_pred = np.argmax(y_pred_prob, axis=1)
        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        try:
            auc_score = float(roc_auc_score(y_true, y_pred_prob, multi_class='ovr', average='macro'))
        except (ValueError, IndexError):
            auc_score = 0.5
        f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        rec = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        prec = float(precision_score(y_true, y_pred, average='macro', zero_division=0))

        # Compute Macro-Averaged One-vs-Rest Specificity
        num_classes = y_pred_prob.shape[1]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        total_samples = np.sum(cm)
        specs = []
        for i in range(num_classes):
            fp_i = np.sum(cm[:, i]) - cm[i, i]
            fn_i = np.sum(cm[i, :]) - cm[i, i]
            tp_i = cm[i, i]
            tn_i = total_samples - (tp_i + fp_i + fn_i)
            if (tn_i + fp_i) > 0:
                specs.append(tn_i / (tn_i + fp_i))
        spec = float(np.mean(specs)) if len(specs) > 0 else 0.0

        return {
            'accuracy': acc, 'roc_auc': auc_score, 'f1': f1, 'recall': rec,
            'precision': prec, 'specificity': spec, 'balanced_acc': bal_acc
        }


def pretrain_ssl_encoder(dataset_info, ssl_epochs=8, lr=1e-3, seed=42):
    """
    Phase 1: Self-Supervised Masked Siamese Pretraining.
    Trains PhySSLNet backbone on unlabelled images using dual masked views.
    """
    set_seed(seed)
    in_channels = dataset_info['in_channels']
    ssl_loader = dataset_info['ssl_loader']
    
    encoder = PhySSLNet(in_channels=in_channels, num_classes=dataset_info['num_classes']).to(device)
    ssl_model = PhySSLSimSiam(encoder=encoder).to(device)
    
    optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ssl_epochs, eta_min=1e-6)
    
    print(f"\\n   >>> [Phase 1: SSL Pretraining] Pretraining PhySSLSimSiam ({ssl_epochs} epochs) with Seed {seed} on unlabelled data...")
    t0 = time.time()
    for ep in range(1, ssl_epochs + 1):
        ssl_model.train()
        total_loss, n_b = 0.0, 0
        for v1, v2 in ssl_loader:
            v1, v2 = v1.to(device), v2.to(device)
            optimizer.zero_grad()
            loss = ssl_model(v1, v2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ssl_model.parameters(), max_norm=5.0)
            if HAS_TPU: xm.optimizer_step(optimizer)
            else: optimizer.step()
            total_loss += loss.item()
            n_b += 1
        scheduler.step()
        if HAS_TPU: xm.mark_step()
        if ep % max(1, ssl_epochs // 2) == 0 or ep == ssl_epochs:
            print(f"       SSL Epoch [{ep:02d}/{ssl_epochs:02d}] - Negative Cosine Loss: {total_loss / max(1, n_b):.4f}")
    ssl_dur = time.time() - t0
    print(f"   >>> [Phase 1: SSL Pretraining] Finished in {ssl_dur:.1f}s.")
    return encoder.state_dict(), ssl_dur


def build_benchmark_model(model_name, in_channels, num_classes):
    m = model_name.lower()
    if m in ("physsl_net_ssl", "physsl_ssl"):
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, rbf_type="adaptive_spline")
    elif m in ("physsl_net", "physsl_asg_rbf"):
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, rbf_type="adaptive_spline")
    elif m == "physsl_intensity_rbf":
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, rbf_type="intensity_rbf")
    elif m == "no_curvature":
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, rbf_type="intensity_rbf", use_curvature=False)
    elif m == "no_token_mixer":
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, rbf_type="intensity_rbf", use_token_mixer=False)
    elif m == "no_bandpass":
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, rbf_type="intensity_rbf", use_bandpass=False)
    elif m == "vanilla_cnn_asg_rbf":
        return VanillaCNNRBF(in_channels=in_channels, num_classes=num_classes, rbf_type="adaptive_spline")
    elif m == "vanilla_cnn_rbf":
        return VanillaCNNRBF(in_channels=in_channels, num_classes=num_classes, rbf_type="gaussian")
    elif m == "vanilla_cnn":
        return VanillaCNN(in_channels=in_channels, num_classes=num_classes)
    elif m == "vgg16":
        return VGG16Small(in_channels=in_channels, num_classes=num_classes)
    elif m == "resnet50v2":
        return ResNet50V2Small(in_channels=in_channels, num_classes=num_classes)
    elif m == "convnext_femto":
        return ConvNeXtFemto(in_channels=in_channels, num_classes=num_classes)
    else:
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes)


def train_and_eval(model_name, dataset_info, epochs=15, lr=1e-3, pretrained_state_dict=None, seed=42, output_dir=None):
    set_seed(seed)
    in_channels = dataset_info['in_channels']
    num_classes = dataset_info['num_classes']
    is_multilabel = dataset_info['is_multilabel']

    model = build_benchmark_model(model_name, in_channels, num_classes).to(device)

    # Load SSL weights if physsl_net_ssl
    if "ssl" in model_name.lower() and pretrained_state_dict is not None:
        m_dict = model.state_dict()
        filtered = {k: v for k, v in pretrained_state_dict.items() if k in m_dict and v.shape == m_dict[k].shape}
        m_dict.update(filtered)
        model.load_state_dict(m_dict)
        print(f" (Transferred {len(filtered)}/{len(m_dict)} SSL Pretrained Tensors)", end="", flush=True)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = 0.0
    best_weights = None
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        for x, y in dataset_info['train_loader']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if HAS_TPU: xm.optimizer_step(optimizer)
            else: optimizer.step()
        scheduler.step()
        if HAS_TPU: xm.mark_step()

        # Validation Check
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in dataset_info['val_loader']:
                x = x.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy() if is_multilabel else F.softmax(logits, dim=1).cpu().numpy()
                val_preds.append(probs)
                val_targets.append(y.numpy())
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)

        val_m = compute_all_metrics(val_targets, val_preds, is_multiclass=(num_classes > 2), is_multilabel=is_multilabel)
        score = val_m['balanced_acc']
        if score > best_score or best_weights is None:
            best_score = score
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t0

    # Test Evaluation
    model.load_state_dict(best_weights)
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for x, y in dataset_info['test_loader']:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy() if is_multilabel else F.softmax(logits, dim=1).cpu().numpy()
            test_preds.append(probs)
            test_targets.append(y.numpy())
    test_preds = np.concatenate(test_preds, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)

    test_m = compute_all_metrics(test_targets, test_preds, is_multiclass=(num_classes > 2), is_multilabel=is_multilabel)

    res = {
        'model': model_name,
        'model_name': model_name,
        'dataset': dataset_info['name'],
        'seed': seed,
        'parameters': num_params,
        'num_parameters': num_params,
        'accuracy': round(test_m['accuracy'], 4),
        'test_accuracy': round(test_m['accuracy'] * 100, 2),
        'balanced_acc': round(test_m['balanced_acc'], 4),
        'test_balanced_acc': round(test_m['balanced_acc'] * 100, 2),
        'roc_auc': round(test_m['roc_auc'], 4),
        'test_roc_auc': round(test_m['roc_auc'], 4),
        'f1': round(test_m['f1'], 4),
        'test_f1': round(test_m['f1'], 4),
        'precision': round(test_m['precision'], 4),
        'test_precision': round(test_m['precision'] * 100, 2),
        'recall': round(test_m['recall'], 4),
        'test_recall': round(test_m['recall'] * 100, 2),
        'specificity': round(test_m['specificity'], 4),
        'test_specificity': round(test_m['specificity'] * 100, 2),
        'train_time_sec': round(train_time, 1),
        'tn': test_m.get('tn', 0),
        'fp': test_m.get('fp', 0),
        'fn': test_m.get('fn', 0),
        'tp': test_m.get('tp', 0)
    }

    if output_dir is not None:
        eval_dir = os.path.join(output_dir, "raw", "evaluation_logs")
        os.makedirs(eval_dir, exist_ok=True)
        eval_record = dict(res)
        eval_record["label_fraction"] = 1.0
        try:
            with open(os.path.join(eval_dir, f"{model_name}_{dataset_info['name']}_seed{seed}_eval.json"), "w") as fp:
                json.dump(eval_record, fp, indent=2)
        except Exception:
            pass

    return res
'''

# Now define analysis_pipeline_code directly from scripts/generate_research_visualizations.py
with open("scripts/generate_research_visualizations.py", "r", encoding="utf-8") as f:
    vis_code = f.read()

# Extract from imports onwards, excluding the main() runner so it can be called seamlessly
vis_pipeline_body = vis_code.split("def main():")[0].strip()

# Create benchmark runner code for kaggle_benchmark.py with multi-seed support
benchmark_runner_code = '''# =====================================================================
# 4. MULTI-SEED BENCHMARK EXECUTION & STATISTICAL AGGREGATION
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["pneumoniamnist", "dermamnist", "breastmnist"],
                        help="List of 2D MedMNIST datasets to run (or 'all')")
    parser.add_argument("--models", nargs="+", default=["physsl_net_ssl", "physsl_net", "convnext_femto", "vanilla_cnn", "vgg16", "resnet50v2"],
                        help="List of models to benchmark (or 'all')")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                        help="List of random seeds to evaluate (e.g. 42 123 456 for 3 seeds, or 42 123 456 789 999 for 5 seeds)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--ssl_epochs", type=int, default=8, help="Number of Self-Supervised pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (256 or 512 for TPU/GPU)")
    default_out = "/kaggle/working/results" if os.path.exists("/kaggle") else "./results"
    parser.add_argument("--output_dir", default=default_out, help="Output directory for results and figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_2d = [
        "pneumoniamnist", "dermamnist", "breastmnist", "bloodmnist",
        "retinamnist", "organcmnist", "organsmnist", "organamnist",
        "pathmnist", "octmnist", "tissuemnist"
    ]

    available_models = [
        "physsl_net_ssl", "physsl_net", "convnext_femto", "vanilla_cnn", "vgg16", "resnet50v2",
        "no_curvature", "no_token_mixer", "no_bandpass", "vanilla_cnn_rbf"
    ]

    selected_datasets = all_2d if "all" in args.datasets else args.datasets
    models = available_models if "all" in args.models else args.models
    seeds = args.seeds

    records = []
    print(f"\\n=================================================================")
    print(f" Starting Multi-Seed Benchmark: {len(selected_datasets)} Datasets x {len(models)} Models x {len(seeds)} Seeds on {device}")
    print(f" Models: {models}")
    print(f" Seeds: {seeds}")
    print(f"=================================================================\\n")

    for d_name in selected_datasets:
        print(f"\\n--> Loading Dataset: {d_name.upper()}...")
        d_info = load_medmnist_dataset(d_name, batch_size=args.batch_size)

        for s_idx, seed in enumerate(seeds):
            print(f"\\n  -------------------------------------------------------------")
            print(f"  >>> Running Seed: {seed} ({s_idx + 1}/{len(seeds)}) on {d_name.upper()}")
            print(f"  -------------------------------------------------------------")
            set_seed(seed)

            # Phase 1: Self-Supervised Pretraining for PhySSL-Net
            ssl_state_dict = None
            if any("ssl" in m for m in models):
                ssl_state_dict, ssl_sec = pretrain_ssl_encoder(d_info, ssl_epochs=args.ssl_epochs, lr=1e-3, seed=seed)

            for m_name in models:
                disp_label = "PHYSSL_NET (SSL PRETRAINED + FINE-TUNED)" if m_name == "physsl_net_ssl" else (
                              "PHYSSL_NET (TRAINED FROM SCRATCH)" if m_name == "physsl_net" else m_name.upper())
                print(f"    [+] Training & Evaluating {disp_label} (Seed {seed}, {args.epochs} epochs)...", end="", flush=True)
                res = train_and_eval(m_name, d_info, epochs=args.epochs, pretrained_state_dict=ssl_state_dict if "ssl" in m_name else None, seed=seed, output_dir=args.output_dir)
                print(f" Done! Acc: {res['test_accuracy']}% | Bal-Acc: {res['test_balanced_acc']}% | AUC: {res['test_roc_auc']} | Params: {res['parameters']:,}")
                records.append(res)

    df = pd.DataFrame(records)
    csv_path = os.path.join(args.output_dir, "medmnist_benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\\n[+] Raw Per-Seed Results successfully saved to: {csv_path}")

    # =====================================================================
    # MULTI-SEED STATISTICAL AGGREGATION (MEAN ± STD ACROSS SEEDS)
    # =====================================================================
    summary_records = []
    for (d_name, m_name), grp in df.groupby(['dataset', 'model']):
        n_s = len(grp)
        acc_m, acc_s = grp['test_accuracy'].mean(), grp['test_accuracy'].std() if n_s > 1 else 0.0
        bal_m, bal_s = grp['test_balanced_acc'].mean(), grp['test_balanced_acc'].std() if n_s > 1 else 0.0
        auc_m, auc_s = grp['test_roc_auc'].mean(), grp['test_roc_auc'].std() if n_s > 1 else 0.0
        f1_m, f1_s = grp['test_f1'].mean(), grp['test_f1'].std() if n_s > 1 else 0.0
        prec_m, prec_s = grp['test_precision'].mean(), grp['test_precision'].std() if n_s > 1 else 0.0
        rec_m, rec_s = grp['test_recall'].mean(), grp['test_recall'].std() if n_s > 1 else 0.0
        spec_m, spec_s = grp['test_specificity'].mean(), grp['test_specificity'].std() if n_s > 1 else 0.0
        params = grp['parameters'].iloc[0]

        summary_records.append({
            'dataset': d_name,
            'model': m_name,
            'num_seeds': n_s,
            'seeds': str(grp['seed'].tolist()),
            'parameters': f"{params:,}",
            'acc_mean': round(acc_m, 2),
            'acc_std': round(acc_s, 2),
            'accuracy_display': f"{acc_m:.2f} ± {acc_s:.2f}%" if n_s > 1 else f"{acc_m:.2f}%",
            'balanced_acc_display': f"{bal_m:.2f} ± {bal_s:.2f}%" if n_s > 1 else f"{bal_m:.2f}%",
            'roc_auc_display': f"{auc_m:.4f} ± {auc_s:.4f}" if n_s > 1 else f"{auc_m:.4f}",
            'f1_display': f"{f1_m:.4f} ± {f1_s:.4f}" if n_s > 1 else f"{f1_m:.4f}",
            'precision_display': f"{prec_m:.2f} ± {prec_s:.2f}%" if n_s > 1 else f"{prec_m:.2f}%",
            'recall_display': f"{rec_m:.2f} ± {rec_s:.2f}%" if n_s > 1 else f"{rec_m:.2f}%",
            'specificity_display': f"{spec_m:.2f} ± {spec_s:.2f}%" if n_s > 1 else f"{spec_m:.2f}%"
        })

    summary_df = pd.DataFrame(summary_records)
    summary_csv = os.path.join(args.output_dir, "multi_seed_benchmark_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[+] Multi-Seed Aggregated Summary saved to: {summary_csv}")

    disp_cols = ['dataset', 'model', 'num_seeds', 'parameters', 'accuracy_display', 'balanced_acc_display', 'roc_auc_display', 'f1_display']
    pretty_summary = summary_df[disp_cols].rename(columns={
        'dataset': 'Dataset',
        'model': 'Model',
        'num_seeds': 'Seeds',
        'parameters': 'Parameters',
        'accuracy_display': 'Test Accuracy (Mean ± SD)',
        'balanced_acc_display': 'Balanced Acc (Mean ± SD)',
        'roc_auc_display': 'ROC-AUC (Mean ± SD)',
        'f1_display': 'F1-Score (Mean ± SD)'
    })

    print(f"\\n=================================================================")
    print(f" MULTI-SEED STATISTICAL SUMMARY (MEAN ± SD ACROSS {len(seeds)} SEEDS)")
    print(f"=================================================================")
    print(pretty_summary.to_string(index=False))

    # Generate complete publication-grade figures, tables, and reports
    print(f"\\n=================================================================")
    print(f" Launching Automated Research Visualisation & Analysis Pipeline")
    print(f"=================================================================\\n")
    create_directory_structure(args.output_dir)
    run_environment_audit(args.output_dir)
    organize_raw_files(args.output_dir)
    generate_dataset_visualizations(os.path.join(args.output_dir, "data", "pneumoniamnist.npz") if os.path.exists(os.path.join(args.output_dir, "data", "pneumoniamnist.npz")) else "./data/pneumoniamnist.npz", args.output_dir)
    generate_training_visualizations(args.output_dir)
    bench_df, _ = load_and_consolidate_results(args.output_dir)
    if bench_df.empty and not df.empty:
        bench_df = df.copy()
        if 'model' in bench_df.columns and 'model_name' not in bench_df.columns:
            bench_df['model_name'] = bench_df['model']
        if 'accuracy' not in bench_df.columns and 'test_accuracy' in bench_df.columns:
            bench_df['accuracy'] = bench_df['test_accuracy'] / 100.0
        if 'roc_auc' not in bench_df.columns and 'test_roc_auc' in bench_df.columns:
            bench_df['roc_auc'] = bench_df['test_roc_auc']
    generate_benchmark_visualizations(bench_df, args.output_dir)
    generate_ablation_visualizations(bench_df, args.output_dir)
    generate_efficiency_visualizations(args.output_dir)
    generate_statistical_visualizations(args.output_dir)
    export_all_research_tables(args.output_dir)
    generate_reports(args.output_dir)

    # Package into downloadable archive
    zip_path = os.path.join(args.output_dir, "..", "physsl_net_research_results.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(args.output_dir):
            for file in files:
                f_p = os.path.join(root, file)
                zipf.write(f_p, os.path.relpath(f_p, os.path.dirname(args.output_dir)))
    print(f"\\n[✓] Research Package Created: {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")
'''

# Assemble full standalone python script
full_py = f"{header_code}\n\nimport zipfile\n\n{vis_pipeline_body}\n\n{models_code}\n\n{datasets_code}\n\n{metrics_training_code}\n\n{benchmark_runner_code}"

# Validate full python script via AST
ast.parse(full_py)
print("[+] Complete kaggle_benchmark.py AST syntax is 100% valid.")

with open("kaggle/kaggle_benchmark.py", "w", encoding="utf-8") as f:
    f.write(full_py)
print("[+] Saved kaggle/kaggle_benchmark.py")

# Assemble the single unified notebook
cells = []

# Cell 1: Header Markdown
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# PhySSL-Net: Physics- and Structure-Informed Self-Supervised Learning\n",
        "## Automated Benchmark Runner & Complete Scientific Research Visualisation Pipeline\n",
        "\n",
        "> **Project Title:** Physics- and Structure-Informed Neural Networks with Self-Supervised Learning for Medical Image Classification  \n",
        "> **Institution:** Delhi Technological University (DTU), Department of Computer Science & Engineering  \n",
        "> **Project Group:** `DTU/CSE/BTP/GP053` (AY 2026–27) | **Project Guide:** Dr. Ravin Ahuja  \n",
        "> **Authors:** Manish Singh, Shivam, Siddharth Kumar Maurya, Saurav Kumar  \n",
        "\n",
        "---\n",
        "### Key Architectural Innovations:\n",
        "1. **Ramanujan-Schoenberg Inspired Basis (`AdaptiveSplineGaussianRBF`):** Learnable Gaussian core modulated by truncated power splines with shape parameter $p_k \\in [1.2, 3.5]$ and exact Gaussian init $p = 2.0$.\n",
        "2. **Dual-Path Band-Pass Residual Block (`BandPassBlock`):** $7\\times 7$ depthwise convolution and absolute modulus for high-frequency edge isolation.\n",
        "3. **Level-Set Curvature Attention (`CurvatureAttention`):** Spatial attention derived from feature map mean curvature divergence $\\text{div}(\\nabla u / \\|\\nabla u\\|)$ with normalized Sobel kernels (/ 8.0).\n",
        "4. **Global Token Mixer (`GlobalTokenMixer`):** $7\\times 7$ spatial grid MLP token and channel mixer with LayerNorm and Dropout.\n",
        "5. **Self-Supervised Siamese Learning (`PhySSLSimSiam`):** Phase 1 unlabelled pretraining using dual stochastic masked patch views ($25\\%$ masking) and stop-gradient cosine loss.\n",
        "\n",
        "---\n",
        "### Multi-Seed Statistical Validation:\n",
        "- Automatically trains and evaluates across multiple deterministic random seeds (e.g. `[42, 123, 456]` for 3 seeds, or `[42, 123, 456, 789, 999]` for 5 seeds).\n",
        "- Aggregates and reports **Mean $\\pm$ Standard Deviation** for Accuracy, Balanced Accuracy, ROC-AUC, and F1-Score.\n",
        "\n",
        "---\n",
        "### Instructions for Running on Kaggle (TPU / GPU / CPU):\n",
        "1. In the right-hand sidebar under **Notebook options**:\n",
        "   - **Accelerator:** Select **GPU P100 / T4 x2** or **TPU VM v3-8 / TPU v5e**.\n",
        "   - **Internet:** Ensure it is toggled **ON** (required to download MedMNIST datasets).\n",
        "2. Click **Run All** (or `Shift + Enter` through each cell).\n",
        "3. The pipeline trains models, performs 2-phase SSL evaluation across seeds, computes statistical averages, generates 21 scientific publication figures (PNG @ 300 DPI, PDF, SVG), 10 research tables (CSV, Excel, LaTeX, Markdown), and packages everything into `/kaggle/working/physsl_net_research_results.zip` for instant download."
    ]
})

# Cell 2: Imports & Setup
c2_text = '''# Install scientific dependencies
!pip install -q medmnist scikit-learn seaborn matplotlib pandas openpyxl

''' + header_code + '''\nimport zipfile\nfrom IPython.display import display, Image, Markdown\n'''

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": to_cell_source(c2_text)
})

# Cell 3: Models
c3_text = models_code + "\n\nprint('[+] Model architectures (PhySSLNet, PhySSLSimSiam, Baselines) successfully defined.')\n"
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": to_cell_source(c3_text)
})

# Cell 4: Datasets
c4_text = datasets_code + "\n\nprint('[+] Datasets and SSL masked transform loaders ready.')\n"
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": to_cell_source(c4_text)
})

# Cell 5: Metrics & 2-Phase Engine
c5_text = metrics_training_code + "\n\nprint('[+] 2-Phase Training Pipeline (Phase 1 SSL Pretraining + Phase 2 Fine-Tuning with Multi-Seed support) ready.')\n"
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": to_cell_source(c5_text)
})

# Cell 6: Multi-Seed Benchmark Execution
c6_script = '''# =====================================================================
# 4. MULTI-SEED BENCHMARK EXECUTION (SSL PRETRAINED vs SCRATCH vs BASELINES)
# =====================================================================

SELECTED_DATASETS = ["pneumoniamnist", "dermamnist", "breastmnist"]

# To benchmark all 11 single-label 2D datasets, uncomment:
# SELECTED_DATASETS = [
#     "pneumoniamnist", "dermamnist", "breastmnist", "bloodmnist",
#     "retinamnist", "organcmnist", "organsmnist", "organamnist",
#     "pathmnist", "octmnist", "tissuemnist"
# ]

MODELS = ["physsl_net_ssl", "physsl_net", "convnext_femto", "vanilla_cnn", "vgg16", "resnet50v2"]
# To evaluate component ablations, add to MODELS:
# MODELS += ["no_curvature", "no_token_mixer", "no_bandpass", "vanilla_cnn_rbf"]

# Multi-Seed Configuration:
# Default: 3 seeds [42, 123, 456] for fast, reliable statistical averaging on Kaggle.
# Set to [42, 123, 456, 789] for 4 seeds, or [42, 123, 456, 789, 999] for full 5-seed publication evaluation.
SEEDS = [42, 123, 456]

EPOCHS = 15
SSL_EPOCHS = 8
BATCH_SIZE = 256  # 256 or 512 for optimal TPU / GPU hardware saturation
BASE_DIR = "/kaggle/working/results" if os.path.exists("/kaggle") else "./results"
os.makedirs(BASE_DIR, exist_ok=True)

records = []
print(f"\\n=================================================================")
print(f" Starting Multi-Seed Benchmark: {len(SELECTED_DATASETS)} Datasets x {len(MODELS)} Models x {len(SEEDS)} Seeds on {device}")
print(f" Models: {MODELS}")
print(f" Seeds: {SEEDS}")
print(f"=================================================================\\n")

for d_name in SELECTED_DATASETS:
    print(f"\\n--> Loading Dataset: {d_name.upper()}...")
    d_info = load_medmnist_dataset(d_name, batch_size=BATCH_SIZE)

    for s_idx, seed in enumerate(SEEDS):
        print(f"\\n  -------------------------------------------------------------")
        print(f"  >>> Running Seed: {seed} ({s_idx + 1}/{len(SEEDS)}) on {d_name.upper()}")
        print(f"  -------------------------------------------------------------")
        set_seed(seed)

        # Phase 1: Self-Supervised Pretraining for PhySSL-Net
        ssl_state_dict = None
        if any("ssl" in m for m in MODELS):
            ssl_state_dict, ssl_sec = pretrain_ssl_encoder(d_info, ssl_epochs=SSL_EPOCHS, lr=1e-3, seed=seed)

        for m_name in MODELS:
            disp_label = "PHYSSL_NET (SSL PRETRAINED + FINE-TUNED)" if m_name == "physsl_net_ssl" else (
                          "PHYSSL_NET (TRAINED FROM SCRATCH)" if m_name == "physsl_net" else m_name.upper())
            print(f"    [+] Training & Evaluating {disp_label} (Seed {seed}, {EPOCHS} epochs)...", end="", flush=True)
            res = train_and_eval(m_name, d_info, epochs=EPOCHS, pretrained_state_dict=ssl_state_dict if "ssl" in m_name else None, seed=seed, output_dir=BASE_DIR)
            print(f" Done! Acc: {res['test_accuracy']}% | Bal-Acc: {res['test_balanced_acc']}% | AUC: {res['test_roc_auc']} | Params: {res['parameters']:,}")
            records.append(res)

df = pd.DataFrame(records)
csv_path = os.path.join(BASE_DIR, "medmnist_benchmark_results.csv")
df.to_csv(csv_path, index=False)
print(f"\\n[+] Raw Per-Seed Results saved to: {csv_path}")

# =====================================================================
# MULTI-SEED STATISTICAL AGGREGATION (MEAN ± STD ACROSS SEEDS)
# =====================================================================
summary_records = []
for (d_name, m_name), grp in df.groupby(['dataset', 'model']):
    n_s = len(grp)
    acc_m, acc_s = grp['test_accuracy'].mean(), grp['test_accuracy'].std() if n_s > 1 else 0.0
    bal_m, bal_s = grp['test_balanced_acc'].mean(), grp['test_balanced_acc'].std() if n_s > 1 else 0.0
    auc_m, auc_s = grp['test_roc_auc'].mean(), grp['test_roc_auc'].std() if n_s > 1 else 0.0
    f1_m, f1_s = grp['test_f1'].mean(), grp['test_f1'].std() if n_s > 1 else 0.0
    prec_m, prec_s = grp['test_precision'].mean(), grp['test_precision'].std() if n_s > 1 else 0.0
    rec_m, rec_s = grp['test_recall'].mean(), grp['test_recall'].std() if n_s > 1 else 0.0
    spec_m, spec_s = grp['test_specificity'].mean(), grp['test_specificity'].std() if n_s > 1 else 0.0
    params = grp['parameters'].iloc[0]

    summary_records.append({
        'dataset': d_name,
        'model': m_name,
        'num_seeds': n_s,
        'seeds': str(grp['seed'].tolist()),
        'parameters': f"{params:,}",
        'acc_mean': round(acc_m, 2),
        'acc_std': round(acc_s, 2),
        'accuracy_display': f"{acc_m:.2f} ± {acc_s:.2f}%" if n_s > 1 else f"{acc_m:.2f}%",
        'balanced_acc_display': f"{bal_m:.2f} ± {bal_s:.2f}%" if n_s > 1 else f"{bal_m:.2f}%",
        'roc_auc_display': f"{auc_m:.4f} ± {auc_s:.4f}" if n_s > 1 else f"{auc_m:.4f}",
        'f1_display': f"{f1_m:.4f} ± {f1_s:.4f}" if n_s > 1 else f"{f1_m:.4f}",
        'precision_display': f"{prec_m:.2f} ± {prec_s:.2f}%" if n_s > 1 else f"{prec_m:.2f}%",
        'recall_display': f"{rec_m:.2f} ± {rec_s:.2f}%" if n_s > 1 else f"{rec_m:.2f}%",
        'specificity_display': f"{spec_m:.2f} ± {spec_s:.2f}%" if n_s > 1 else f"{spec_m:.2f}%"
    })

summary_df = pd.DataFrame(summary_records)
summary_csv = os.path.join(BASE_DIR, "multi_seed_benchmark_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"[+] Multi-Seed Aggregated Summary saved to: {summary_csv}")

disp_cols = ['dataset', 'model', 'num_seeds', 'parameters', 'accuracy_display', 'balanced_acc_display', 'roc_auc_display', 'f1_display']
pretty_summary = summary_df[disp_cols].rename(columns={
    'dataset': 'Dataset',
    'model': 'Model',
    'num_seeds': 'Seeds',
    'parameters': 'Parameters',
    'accuracy_display': 'Test Accuracy (Mean ± SD)',
    'balanced_acc_display': 'Balanced Acc (Mean ± SD)',
    'roc_auc_display': 'ROC-AUC (Mean ± SD)',
    'f1_display': 'F1-Score (Mean ± SD)'
})

print(f"\\n=================================================================")
print(f" MULTI-SEED STATISTICAL SUMMARY (MEAN ± SD ACROSS {len(SEEDS)} SEEDS)")
print(f"=================================================================")
display(pretty_summary) if "display" in globals() else print(pretty_summary.to_string(index=False))
'''
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": to_cell_source(c6_script)
})

# Cell 7: Markdown Section for Research Visualisations
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "# 5. Automated Research Visualisation, Statistical Analysis & Publication Results Pipeline\n",
        "The following cells automatically generate the complete, publication-grade scientific deliverables for the DTU B.Tech project report:\n",
        "- **21 Publication Figures** (in high-res 300 DPI PNG, vector PDF, and SVG)\n",
        "- **10 Comprehensive Research Tables** (in CSV, Excel XLSX, GitHub Markdown, and Academic LaTeX)\n",
        "- **Multi-Seed Statistical Hypothesis Testing** (McNemar Paired Chi-Square & Exact Binomial tests, 95% CIs)\n",
        "- **Pareto Frontier Efficiency Analysis**\n",
        "- **Automated Packaging** into a downloadable `.zip` file."
    ]
})

# Cell 8: Code for Research Pipeline Execution
c8_pipeline = vis_pipeline_body + '''

# Execute the complete research analysis pipeline
create_directory_structure(BASE_DIR)
run_environment_audit(BASE_DIR)
organize_raw_files(BASE_DIR)

# Dataset Visualisations (Figs 1-4, Tables 1-2)
pneumonia_npz = "./data/pneumoniamnist.npz"
if not os.path.exists(pneumonia_npz):
    pneumonia_npz = os.path.join(BASE_DIR, "data", "pneumoniamnist.npz")
generate_dataset_visualizations(pneumonia_npz, BASE_DIR)

# Training Convergence (Figs 5-9)
generate_training_visualizations(BASE_DIR)

# Main Benchmark & Ablation (Figs 10-17, Tables 4-6)
benchmark_df, _ = load_and_consolidate_results(BASE_DIR)
if benchmark_df.empty and 'df' in globals() and not df.empty:
    benchmark_df = df.copy()
    if 'model' in benchmark_df.columns and 'model_name' not in benchmark_df.columns:
        benchmark_df['model_name'] = benchmark_df['model']
    if 'accuracy' not in benchmark_df.columns and 'test_accuracy' in benchmark_df.columns:
        benchmark_df['accuracy'] = benchmark_df['test_accuracy'] / 100.0
    if 'roc_auc' not in benchmark_df.columns and 'test_roc_auc' in benchmark_df.columns:
        benchmark_df['roc_auc'] = bench_df['test_roc_auc']

generate_benchmark_visualizations(benchmark_df, BASE_DIR)
generate_ablation_visualizations(benchmark_df, BASE_DIR)

# Computational Efficiency & Pareto (Figs 18-20, Table 7)
generate_efficiency_visualizations(BASE_DIR)

# Statistical Significance (Fig 21, Table 8)
generate_statistical_visualizations(BASE_DIR)

# Export Full Tables 1-10 (CSV, Excel, LaTeX, Markdown)
export_all_research_tables(BASE_DIR)

# Generate Scientific Reports and Index
generate_reports(BASE_DIR)
print("\\n[✓] Research Analysis Pipeline Complete: All 21 figures and 10 tables generated.")
'''
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": to_cell_source(c8_pipeline)
})

# Cell 9: Code to Display Results Inline in Notebook
c9_display = '''# =====================================================================
# 6. INTERACTIVE SCIENTIFIC RESULTS INSPECTION & REPORT DISPLAY
# =====================================================================

print("=================================================================")
print(" 1. MULTI-SEED BENCHMARK PERFORMANCE COMPARISON (MEAN ± SD)")
print("=================================================================")
display(Image(filename=f"{BASE_DIR}/figures/benchmark/fig_10_model_accuracy_comparison.png"))
display(Image(filename=f"{BASE_DIR}/figures/benchmark/fig_11_roc_comparison.png"))
display(Image(filename=f"{BASE_DIR}/figures/benchmark/fig_12_metrics_grouped_comparison.png"))
display(Image(filename=f"{BASE_DIR}/figures/benchmark/fig_14_seed_variability.png"))

print("\\n--> MULTI-SEED STATISTICAL SUMMARY TABLE (Mean ± SD Across Seeds):")
multi_summary_file = f"{BASE_DIR}/multi_seed_benchmark_summary.csv"
if os.path.exists(multi_summary_file):
    display(pd.read_csv(multi_summary_file))
elif os.path.exists(f"{BASE_DIR}/tables/benchmark/table_05_five_seed_summary.csv"):
    display(pd.read_csv(f"{BASE_DIR}/tables/benchmark/table_05_five_seed_summary.csv"))

print("\\n--> TABLE 4: Per-Seed Breakdown Table:")
if os.path.exists(f"{BASE_DIR}/tables/benchmark/table_04_benchmark_per_seed.csv"):
    display(pd.read_csv(f"{BASE_DIR}/tables/benchmark/table_04_benchmark_per_seed.csv"))

print("\\n=================================================================")
print(" 2. ARCHITECTURAL ABLATION STUDY")
print("=================================================================")
display(Image(filename=f"{BASE_DIR}/figures/ablation/fig_15_ablation_comparison.png"))
display(Image(filename=f"{BASE_DIR}/figures/ablation/fig_17_component_removal_impact.png"))

print("\\n--> TABLE 6: Architectural Component Ablation Study:")
display(pd.read_csv(f"{BASE_DIR}/tables/ablation/table_06_ablation_study.csv"))

print("\\n=================================================================")
print(" 3. COMPUTATIONAL EFFICIENCY & PARETO FRONTIER")
print("=================================================================")
display(Image(filename=f"{BASE_DIR}/figures/efficiency/fig_20_pareto_frontier.png"))

print("\\n--> TABLE 7: Computational Efficiency Profile:")
display(pd.read_csv(f"{BASE_DIR}/tables/efficiency/table_07_computational_efficiency.csv"))

print("\\n=================================================================")
print(" 4. STATISTICAL SIGNIFICANCE & HYPOTHESIS TESTING")
print("=================================================================")
display(Image(filename=f"{BASE_DIR}/figures/statistics/fig_21_statistical_significance.png"))

print("\\n--> TABLE 8: Statistical Hypothesis Testing (McNemar Paired Tests):")
display(pd.read_csv(f"{BASE_DIR}/tables/statistics/table_08_statistical_significance.csv"))
'''
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": to_cell_source(c9_display)
})

# Cell 10: One-Click Packaging & Download Archive
c10_package = '''# =====================================================================
# 7. AUTOMATED PACKAGING & DOWNLOADABLE ARCHIVE GENERATION
# =====================================================================
zip_output_path = "/kaggle/working/physsl_net_research_results.zip" if os.path.exists("/kaggle") else "./physsl_net_research_results.zip"

print(f"[*] Packaging all research deliverables into {zip_output_path}...")
with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, os.path.dirname(BASE_DIR))
            zipf.write(file_path, arcname)

file_size_mb = os.path.getsize(zip_output_path) / (1024 * 1024)
print(f"\\n=================================================================")
print(f" [✓] COMPLETE RESEARCH DELIVERABLES ARCHIVE CREATED!")
print(f" Archive Location: {zip_output_path} ({file_size_mb:.2f} MB)")
print(f" Contents:")
print(f"   • Multi-Seed Statistical Aggregate Tables (Mean ± SD)")
print(f"   • 21 Publication Figures (PNG 300 DPI, vector PDF, SVG)")
print(f"   • 10 Comprehensive Research Tables (CSV, Excel, LaTeX, Markdown)")
print(f"   • Statistical Hypothesis Tests (McNemar Paired Tests, 95% CIs)")
print(f"   • Automated Academic Summary & Visualisation Index Reports")
print(f"=================================================================\\n")
'''
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": to_cell_source(c10_package)
})

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "TPU",
        "colab": {"provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Verify each code cell's syntax via Python AST
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        c_src = "".join([l for l in cell['source'] if not l.strip().startswith('!') and not l.strip().startswith('%')])
        try:
            ast.parse(c_src)
            print(f"[+] Cell {i+1} code syntax verified OK.")
        except SyntaxError as e:
            print(f"[!] SYNTAX ERROR in Cell {i+1}: {e}")
            raise e

# Save the final Kaggle notebook
with open("kaggle/kaggle_physsl_medmnist_benchmark.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print("[+] Saved kaggle/kaggle_physsl_medmnist_benchmark.ipynb successfully!")
