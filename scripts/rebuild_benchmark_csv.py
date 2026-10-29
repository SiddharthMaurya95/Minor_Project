"""
Rebuild results/benchmark_results.csv from individual *_eval.json records
ensuring consistent column schemas and zero row formatting errors.
"""
import os
import glob
import json
import csv

def rebuild_csv(results_dir="./results"):
    json_files = sorted(glob.glob(os.path.join(results_dir, "*_eval.json")))
    print(f"Found {len(json_files)} evaluation JSON records.")

    fieldnames = [
        'model_name', 'dataset', 'split', 'seed', 'label_fraction',
        'accuracy', 'roc_auc', 'f1', 'recall', 'specificity',
        'threshold', 'tta_applied', 'logit_adj_applied',
        'num_parameters', 'latency_ms_per_sample', 'throughput_fps',
        'checkpoint_path'
    ]

    records = []
    for f in json_files:
        with open(f, 'r') as fp:
            d = json.load(fp)
        
        row = {
            'model_name': d.get('model_name', 'N/A'),
            'dataset': d.get('dataset', 'pneumoniamnist'),
            'split': d.get('split', 'test'),
            'seed': d.get('seed', 42),
            'label_fraction': d.get('label_fraction', 1.0),
            'accuracy': d.get('accuracy', 0.0),
            'roc_auc': d.get('roc_auc', 0.0),
            'f1': d.get('f1', 0.0),
            'recall': d.get('recall', 0.0),
            'specificity': d.get('specificity', 0.0),
            'threshold': d.get('threshold', 0.5),
            'tta_applied': d.get('tta_applied', False),
            'logit_adj_applied': d.get('logit_adj_applied', False),
            'num_parameters': d.get('num_parameters', 0),
            'latency_ms_per_sample': d.get('latency_ms_per_sample', 0.0),
            'throughput_fps': d.get('throughput_fps', 0.0),
            'checkpoint_path': d.get('checkpoint_path', ''),
        }
        records.append(row)

    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    with open(csv_path, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[+] Successfully wrote {len(records)} consistent rows to {csv_path}")

if __name__ == "__main__":
    rebuild_csv()
