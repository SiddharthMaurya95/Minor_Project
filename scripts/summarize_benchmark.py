"""
Script to format and display complete aggregated benchmark, ablation, and statistical test results.
"""
import json
import os

def display_summary():
    agg_path = "results/benchmark_aggregated_summary.json"
    if os.path.exists(agg_path):
        with open(agg_path, 'r') as f:
            agg = json.load(f)
        
        print("\n" + "=" * 105)
        print(" FINAL BENCHMARK RESULTS ACROSS 5 SEEDS (42, 123, 456, 789, 999) ON TEST SET (624 SAMPLES)")
        print("=" * 105)
        print(f"{'Model':<22} | {'Accuracy (%)':<16} | {'ROC-AUC':<16} | {'Recall/Sens (%)':<17} | {'Specificity (%)':<17} | {'F1-Score':<10}")
        print("-" * 105)
        for k, v in agg.items():
            acc_m = v['accuracy']['mean'] * 100
            acc_s = v['accuracy']['std'] * 100
            auc_m = v['roc_auc']['mean']
            auc_s = v['roc_auc']['std']
            rec_m = v['recall']['mean'] * 100
            rec_s = v['recall']['std'] * 100
            spc_m = v['specificity']['mean'] * 100
            spc_s = v['specificity']['std'] * 100
            f1_m = v['f1']['mean']
            f1_s = v['f1']['std']
            print(f"{k:<22} | {acc_m:5.2f} ± {acc_s:4.2f}% | {auc_m:.4f} ± {auc_s:.4f} | {rec_m:5.2f} ± {rec_s:4.2f}% | {spc_m:5.2f} ± {spc_s:4.2f}% | {f1_m:.4f} ± {f1_s:.4f}")

    ab_path = "results/ablation_summary.json"
    if os.path.exists(ab_path):
        with open(ab_path, 'r') as f:
            ab = json.load(f)
        print("\n" + "=" * 105)
        print(" COMPONENT ABLATION STUDY RESULTS (SEED 42, HELD-OUT TEST SET)")
        print("=" * 105)
        print(f"{'Ablation Variant':<28} | {'Test Accuracy (%)':<18} | {'ROC-AUC':<12} | {'Sensitivity (%)':<17} | {'Specificity (%)':<17}")
        print("-" * 105)
        for k, v in ab.items():
            acc = v['mean'] * 100
            auc = v['roc_auc']
            rec = v['recall'] * 100
            spc = v['specificity'] * 100
            print(f"{k:<28} | {acc:5.2f}%            | {auc:.4f}     | {rec:5.2f}%           | {spc:5.2f}%")

    frac_path = "results/label_fraction_results.json"
    if os.path.exists(frac_path):
        with open(frac_path, 'r') as f:
            frac = json.load(f)
        print("\n" + "=" * 105)
        print(" LABEL FRACTION DATA SCARCITY STUDY (SEED 42, HELD-OUT TEST SET)")
        print("=" * 105)
        print(f"{'Label Fraction':<16} | {'PhySSL-Net Accuracy':<22} | {'Vanilla CNN Accuracy':<22} | {'PhySSL-Net Advantage':<20}")
        print("-" * 105)
        for k in ["100%", "50%", "25%", "10%"]:
            p_acc = frac['physsl_net_ssl'][k]['accuracy'] * 100
            c_acc = frac['vanilla_cnn'][k]['accuracy'] * 100
            diff = p_acc - c_acc
            diff_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
            print(f"{k:<16} | {p_acc:5.2f}%                | {c_acc:5.2f}%                | {diff_str}")

    mcn_path = "results/mcnemar_tests.json"
    if os.path.exists(mcn_path):
        with open(mcn_path, 'r') as f:
            mcn = json.load(f)
        print("\n" + "=" * 105)
        print(" MCNEMAR'S STATISTICAL SIGNIFICANCE TESTS (PAIRED ERROR DISCORDANCE ON TEST SET)")
        print("=" * 105)
        for k, v in mcn.items():
            print(f" {k:<28} -> Chi2 = {v['statistic']:6.2f}, p-value = {v['p_value']:.4e} ({v['interpretation']})")

if __name__ == "__main__":
    display_summary()
