"""
Summarize pilot run results for Seed 42 across all architectures.
"""
import os
import glob
import json

def summarize():
    log_files = sorted(glob.glob("results/*_seed42_log.json"))
    print(f"{'Model/Log File':<42} | {'Epoch':<5} | {'Val AUC':<8} | {'Val Acc':<8} | {'Runtime (s)':<11} | {'Checkpoint'}")
    print("-" * 115)
    
    for f in log_files:
        basename = os.path.basename(f)
        with open(f, 'r') as fp:
            d = json.load(fp)
        
        best_ep = d.get("best_epoch", "N/A")
        best_auc = d.get("best_val_auc", 0.0)
        hist = d.get("history", {})
        val_accs = hist.get("val_acc", [])
        if isinstance(best_ep, int) and 0 <= best_ep - 1 < len(val_accs):
            best_acc = val_accs[best_ep - 1]
        elif val_accs:
            best_acc = val_accs[-1]
        else:
            best_acc = 0.0
            
        runtime = d.get("total_duration_sec", 0.0)
        ckpt = os.path.basename(d.get("checkpoint_path", "N/A"))
        print(f"{basename:<42} | {best_ep!s:<5} | {best_auc:<8.4f} | {best_acc:<8.4f} | {runtime:<11.2f} | {ckpt}")

if __name__ == "__main__":
    summarize()
