# Reproducibility Guide: PhySSL-Net Research (REPRODUCIBILITY.md)

## 1. Environment & Hardware Specifications
All experimental evaluations are executed under the following hardware and software configuration:
- **Operating System:** Windows 11 (64-bit)
- **Host GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MiB VRAM, Driver Version 616.92, CUDA 13.4 supported)
- **Host CPU:** Multi-core x86_64 CPU
- **Python Version:** Python 3.12 (Managed via isolated virtual environment in `./.venv`)
- **PyTorch Version:** PyTorch `2.6.0+cu124` with CUDA acceleration enabled
- **Torchvision Version:** `0.21.0+cu124`
- **Key Packages:** `medmnist==3.0.2`, `scikit-learn==1.9.1`, `scipy==1.18.1`, `matplotlib==3.11.2`, `seaborn==0.13.2`, `pandas==3.1.0`, `python-docx==1.1.2`, `pyyaml==6.0.3`, `pytest==9.2.0`

---

## 2. Environment Setup Instructions

### Step 1: Create Virtual Environment
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
```

### Step 2: Install PyTorch with CUDA 12.4
```powershell
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Step 3: Install Research & Evaluation Dependencies
```powershell
.\.venv\Scripts\pip install -r requirements.txt
```

---

## 3. Step-by-Step Reproduction Pipeline

### 1. Verify Dataset Integrity
Downloads PneumoniaMNIST (if not present) and asserts official train (4,708), val (524), and test (624) partition counts, verifies stratified sampling, and checks for zero train-test leakage:
```powershell
.\.venv\Scripts\python scripts/check_dataset.py --dataset pneumoniamnist
```

### 2. Verify Models and Gradient Flow
Runs forward and backward passes across all model variants (`PhySSLNet`, `CMixer`, `VanillaCNN`, `VGG16Small`, `ResNet50V2Small`, `PhySSLSimSiam`), verifying tensor shapes and parameter updates:
```powershell
.\.venv\Scripts\python scripts/check_models.py
```

### 3. Run Automated Unit Tests (66 Tests)
Executes the comprehensive 66-test automated test suite:
```powershell
.\.venv\Scripts\python -m pytest tests -v
```

### 4. Self-Supervised Masked Consistency Pretraining
Pretrains `PhySSLNet` on unlabelled images using dual stochastic masked views (stop-gradient negative cosine loss):
```powershell
.\.venv\Scripts\python scripts/pretrain_ssl.py --dataset pneumoniamnist --epochs 10 --batch_size 64
```

### 5. Supervised Fine-Tuning with Balanced Accuracy Checkpoint Selection
Fine-tunes the pretrained encoder using labelled data with balanced accuracy checkpoint monitoring and bilateral horizontal-flip augmentation:
```powershell
# Example: Seed 123
.\.venv\Scripts\python scripts/train.py --model physsl_net --dataset pneumoniamnist --epochs 25 --val_metric balanced_acc --hflip_aug --pretrained_path checkpoints/physsl_simsiam_pretrained.pth --seed 123
```

### 6. Evaluate on Sequestered Test Split

#### 6.1 Standard Evaluation ($t = 0.50$)
Evaluates checkpoint accuracy, ROC-AUC, F1, sensitivity, specificity, and confusion matrix:
```powershell
.\.venv\Scripts\python scripts/evaluate.py --checkpoint checkpoints/physsl_net_pneumoniamnist_frac100_seed123_best.pth --dataset pneumoniamnist --split test
# Outputs: Test Accuracy = 89.42%, ROC-AUC = 0.9629, Recall = 96.15%, Specificity = 74.79%
```

#### 6.2 Bayesian Prior Shift Alignment (Logit Adjustment, $t^* = 0.742$)
Directly aligns the decision boundary with the empirical training prior ($\pi_{\text{train}} \approx 0.742$), eliminating false positives caused by the partition asymmetry:
```powershell
.\.venv\Scripts\python scripts/evaluate.py --checkpoint checkpoints/physsl_net_pneumoniamnist_frac100_seed123_best.pth --dataset pneumoniamnist --split test --logit_adj
# Outputs: Test Accuracy = 90.87%, ROC-AUC = 0.9629, Recall = 96.15%, Specificity = 82.05%
```

### 7. Run Complete End-to-End Benchmark Suite
Executes the full 5-seed benchmark (`[42, 123, 456, 789, 999]`), architectural ablations, label-fraction studies, latency/parameter profiling, McNemar tests, and generates all publication figures:
```powershell
.\.venv\Scripts\python scripts/run_experiments.py --dataset pneumoniamnist --epochs 15 --ssl_epochs 10
```

### 8. Generate DTU B.Tech Word Thesis Report
Compiles all verified results, statistics, and figures into the formatted Word thesis document:
```powershell
.\.venv\Scripts\python scripts/generate_docx_report.py
# Outputs: docs/PhySSL_Net_Final_Report_DTU.docx
```

---

## 4. Controlled Fair Comparison Guarantee
- **Fixed Seeds:** `42`, `123`, `456`, `789`, `999`.
- **Identical Partitions:** All baselines and PhySSL-Net share the exact same training, validation, and test splits.
- **Identical Optimization Budget:** All models receive the same training budget, learning rate schedule, and early stopping rules.
- **Zero Test Leakage:** Checkpoints are chosen exclusively based on validation balanced accuracy / ROC-AUC. Test splits are only accessed at final evaluation.
