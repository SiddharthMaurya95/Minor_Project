"""
Builds the unified kaggle_benchmark.py and kaggle_physsl_medmnist_benchmark.ipynb
with the complete Research Visualisation, Statistical Analysis, and Results Generation
pipeline embedded directly into the single Kaggle benchmark notebook and script!
"""

import os
import json
import ast

os.makedirs("kaggle", exist_ok=True)

# Helper to split text into lines with keepends=True for clean nbformat
def to_cell_source(text):
    return text.splitlines(keepends=True)

# Read generate_unified_kaggle.py to reuse the pristine models, datasets, metrics, header code
with open("C:/Users/maury/.gemini/antigravity/brain/6b6e073f-62c2-4fcd-a6ff-86f3e24925a6/scratch/generate_unified_kaggle.py", "r", encoding="utf-8") as f:
    orig_gen = f.read()

# Extract header_code, models_code, datasets_code, metrics_training_code
header_code = orig_gen.split("models_code = '''")[0].split("header_code = '''")[1].rstrip("'''\n")
models_code = orig_gen.split("models_code = '''")[1].split("datasets_code = '''")[0].rstrip("'''\n")
datasets_code = orig_gen.split("datasets_code = '''")[1].split("metrics_training_code = '''")[0].rstrip("'''\n")
metrics_training_code = orig_gen.split("metrics_training_code = '''")[1].split("benchmark_runner_code = '''")[0].rstrip("'''\n")

print(f"[+] Loaded existing validated blocks:")
print(f"    header_code: {len(header_code)} chars")
print(f"    models_code: {len(models_code)} chars")
print(f"    datasets_code: {len(datasets_code)} chars")
print(f"    metrics_training_code: {len(metrics_training_code)} chars")

# Now define analysis_pipeline_code directly from scripts/generate_research_visualizations.py
with open("scripts/generate_research_visualizations.py", "r", encoding="utf-8") as f:
    vis_code = f.read()

# Extract from imports onwards, excluding the main() runner so it can be called seamlessly
vis_pipeline_body = vis_code.split("def main():")[0].strip()

# Create benchmark runner code for kaggle_benchmark.py
benchmark_runner_code = '''# =====================================================================
# 4. BENCHMARK EXECUTION (SSL PRETRAINED vs SCRATCH vs BASELINES)
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["pneumoniamnist", "dermamnist", "breastmnist"],
                        help="List of 2D MedMNIST datasets to run (or 'all')")
    parser.add_argument("--models", nargs="+", default=["physsl_net_ssl", "physsl_net", "convnext_femto", "vanilla_cnn", "vgg16", "resnet50v2"],
                        help="List of models to benchmark (or 'all')")
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

    records = []
    print(f"\\n=================================================================")
    print(f" Starting Benchmark: {len(selected_datasets)} Datasets x {len(models)} Models on {device}")
    print(f" Models: {models}")
    print(f"=================================================================\\n")

    for d_name in selected_datasets:
        print(f"\\n--> Loading Dataset: {d_name.upper()}...")
        d_info = load_medmnist_dataset(d_name, batch_size=args.batch_size)

        # Phase 1: Self-Supervised Pretraining for PhySSL-Net
        ssl_state_dict = None
        if any("ssl" in m for m in models):
            ssl_state_dict, ssl_sec = pretrain_ssl_encoder(d_info, ssl_epochs=args.ssl_epochs, lr=1e-3)

        for m_name in models:
            disp_label = "PHYSSL_NET (SSL PRETRAINED + FINE-TUNED)" if m_name == "physsl_net_ssl" else (
                          "PHYSSL_NET (TRAINED FROM SCRATCH)" if m_name == "physsl_net" else m_name.upper())
            print(f"  [+] Training & Evaluating {disp_label} on {d_name} ({args.epochs} epochs)...", end="", flush=True)
            res = train_and_eval(m_name, d_info, epochs=args.epochs, pretrained_state_dict=ssl_state_dict if "ssl" in m_name else None)
            print(f" Done! Acc: {res['test_accuracy']}% | Bal-Acc: {res['test_balanced_acc']}% | AUC: {res['test_roc_auc']} | Params: {res['parameters']:,}")
            records.append(res)

    df = pd.DataFrame(records)
    csv_path = os.path.join(args.output_dir, "medmnist_benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\\n[+] Results successfully saved to: {csv_path}")
    print(df.to_string())

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
        "### Instructions for Running on Kaggle (TPU / GPU / CPU):\n",
        "1. In the right-hand sidebar under **Notebook options**:\n",
        "   - **Accelerator:** Select **GPU P100 / T4 x2** or **TPU VM v3-8 / TPU v5e**.\n",
        "   - **Internet:** Ensure it is toggled **ON** (required to download MedMNIST datasets).\n",
        "2. Click **Run All** (or `Shift + Enter` through each cell).\n",
        "3. The pipeline trains models, performs 2-phase SSL evaluation, generates 21 scientific publication figures (PNG @ 300 DPI, PDF, SVG), 10 research tables (CSV, Excel, LaTeX, Markdown), and packages everything into `/kaggle/working/physsl_net_research_results.zip` for instant download."
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
c5_text = metrics_training_code + "\n\nprint('[+] 2-Phase Training Pipeline (Phase 1 SSL Pretraining + Phase 2 Fine-Tuning) ready.')\n"
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": to_cell_source(c5_text)
})

# Cell 6: Benchmark Execution
c6_script = '''# =====================================================================
# 4. BENCHMARK EXECUTION (SSL PRETRAINED vs SCRATCH vs BASELINES)
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

EPOCHS = 15
SSL_EPOCHS = 8
BATCH_SIZE = 256  # 256 or 512 for optimal TPU / GPU hardware saturation
BASE_DIR = "/kaggle/working/results" if os.path.exists("/kaggle") else "./results"
os.makedirs(BASE_DIR, exist_ok=True)

records = []
print(f"\\n=================================================================")
print(f" Starting Benchmark: {len(SELECTED_DATASETS)} Datasets x {len(MODELS)} Models on {device}")
print(f" Models: {MODELS}")
print(f"=================================================================\\n")

for d_name in SELECTED_DATASETS:
    print(f"\\n--> Loading Dataset: {d_name.upper()}...")
    d_info = load_medmnist_dataset(d_name, batch_size=BATCH_SIZE)

    # Phase 1: Self-Supervised Pretraining for PhySSL-Net
    ssl_state_dict = None
    if any("ssl" in m for m in MODELS):
        ssl_state_dict, ssl_sec = pretrain_ssl_encoder(d_info, ssl_epochs=SSL_EPOCHS, lr=1e-3)

    for m_name in MODELS:
        disp_label = "PHYSSL_NET (SSL PRETRAINED + FINE-TUNED)" if m_name == "physsl_net_ssl" else (
                      "PHYSSL_NET (TRAINED FROM SCRATCH)" if m_name == "physsl_net" else m_name.upper())
        print(f"  [+] Training & Evaluating {disp_label} on {d_name} ({EPOCHS} epochs)...", end="", flush=True)
        res = train_and_eval(m_name, d_info, epochs=EPOCHS, pretrained_state_dict=ssl_state_dict if "ssl" in m_name else None)
        print(f" Done! Acc: {res['test_accuracy']}% | Bal-Acc: {res['test_balanced_acc']}% | AUC: {res['test_roc_auc']} | Params: {res['parameters']:,}")
        records.append(res)

df = pd.DataFrame(records)
csv_path = os.path.join(BASE_DIR, "medmnist_benchmark_results.csv")
df.to_csv(csv_path, index=False)
print(f"\\n[+] Results successfully saved to: {csv_path}")
display(df) if "display" in globals() else print(df.to_string())
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
        "- **Statistical Hypothesis Testing** (McNemar Paired Chi-Square & Exact Binomial tests, 95% CIs)\n",
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
        benchmark_df['roc_auc'] = benchmark_df['test_roc_auc']

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
print(" 1. MAIN BENCHMARK PERFORMANCE COMPARISON (5-SEED MEAN ± SD)")
print("=================================================================")
display(Image(filename=f"{BASE_DIR}/figures/benchmark/fig_10_model_accuracy_comparison.png"))
display(Image(filename=f"{BASE_DIR}/figures/benchmark/fig_11_roc_comparison.png"))
display(Image(filename=f"{BASE_DIR}/figures/benchmark/fig_12_metrics_grouped_comparison.png"))

print("\\n--> TABLE 5: Five-Seed Performance Summary:")
display(pd.read_csv(f"{BASE_DIR}/tables/benchmark/table_05_five_seed_summary.csv"))

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

