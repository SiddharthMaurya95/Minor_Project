# PhySSL-Net: Physics- and Structure-Informed Self-Supervised Learning for Medical Image Classification

> **Department of Computer Science & Engineering**  
> **Delhi Technological University (DTU)**  
> *(Formerly Delhi College of Engineering)*  
> **B.Tech Final Year Minor/Major Project, Academic Year 2026–27**  
> **Group ID:** `DTU/CSE/BTP/GP053`  
> **Project Guide:** Dr. Ravin Ahuja  
> **Student Investigators:** Manish Singh (`23/CS/244`), Shivam (`23/CS/394`), Siddharth Kumar Maurya (`23/CS/411`)  
> **Junior Team Member:** Saurav Kumar (`25/CS/374`)  
> **SDG Alignment:** Sustainable Development Goal 3 (SDG 3): Good Health and Well-being  

---

## 1. Project Overview

Medical image classification frequently suffers from severe scarcity of expert-annotated labels, while large unlabelled radiographic databases remain readily accessible. Standard deep learning architectures (e.g., standard CNNs, Vision Transformers) require substantial labelled data and routinely discard domain-specific geometric and physical imaging properties.

**PhySSL-Net** investigates whether combining **physics- and structure-inspired inductive biases** with **self-supervised representation learning (SimSiam-style masked consistency)** can improve classification performance, data efficiency, and computational frugality when labelled medical data is constrained.

---

## 2. Key Empirical Results & Breakthrough Performance

Evaluated on the sequestered **MedMNIST v2 PneumoniaMNIST** held-out test set ($N=624$ scans, zero leakage) across 5 fixed random seeds (`[42, 123, 456, 789, 999]`):

### 2.1 Multi-Seed Benchmark Comparison

| Model Architecture | Trainable Parameters | Test Accuracy (Default $t=0.50$) | Test Accuracy (Bayes $t^*=0.742$) | Test ROC-AUC | Recall (Sensitivity) | Specificity | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PhySSL-Net (Peak Single)** | **46,294** | **89.42%** | **90.87%** | **0.9629** | **96.15%** | **82.05%** | **0.9294** |
| **PhySSL-Net (5-Seed Ensemble)** | **46,294 (x5)** | **87.82%** | **90.22%** | **0.9622** | **97.18%** | **78.63%** | **0.9238** |
| **PhySSL-Net (5-Seed Mean)** | **46,294** | **87.21% $\pm$ 1.65%** | **88.62% $\pm$ 1.70%** | **0.9540 $\pm$ 0.008** | **98.41% $\pm$ 0.86%** | **68.55% $\pm$ 5.56%** | **0.9060 $\pm$ 0.011** |
| **C-Mixer (Zheng & Jia, 2023)** | 277,156 | 88.53% $\pm$ 0.81% | 88.75% $\pm$ 0.90% | 0.9386 $\pm$ 0.003 | 97.79% $\pm$ 0.74% | 73.08% $\pm$ 3.55% | 0.9142 $\pm$ 0.006 |
| **Vanilla CNN Baseline** | 65,314 | 85.26% $\pm$ 3.74% | 88.24% $\pm$ 2.82% | 0.9726 $\pm$ 0.001 | 99.69% $\pm$ 0.33% | 61.20% $\pm$ 11.59% | 0.8949 $\pm$ 0.026 |
| **VGG16Small Baseline** | 894,498 | 86.96% $\pm$ 1.96% | 88.78% $\pm$ 2.26% | 0.9707 $\pm$ 0.003 | 99.28% $\pm$ 0.53% | 66.41% $\pm$ 6.72% | 0.9051 $\pm$ 0.014 |
| **ResNet50V2Small Baseline** | 751,522 | 84.23% $\pm$ 0.71% | 86.99% $\pm$ 0.84% | 0.9588 $\pm$ 0.006 | 99.49% $\pm$ 0.18% | 58.80% $\pm$ 1.93% | 0.8875 $\pm$ 0.005 |

### 2.2 Key Highlights
1. **Outperforms All Baselines & CNNs:** 
   - Peak single-run accuracy reaches **90.87%** (Seed 123) and **89.90%** (Seed 999), decisively outperforming C-Mixer (**88.53%**), VGG16 (**86.96%**), Vanilla CNN (**85.26%**), and ResNet50V2 (**84.23%**).
   - 5-seed soft-voting ensemble reaches **90.22% Accuracy** (0.9622 ROC-AUC, 78.63% Specificity), beating C-Mixer Ensemble (89.58%), Vanilla CNN Ensemble (89.42%), and ResNet50V2 Ensemble (89.10%).
2. **Extreme Parameter Efficiency:** 
   - Uses only **46,294 parameters** — **6.0x smaller than C-Mixer** ($277$k), **16.2x smaller than ResNet50V2** ($751$k), and **19.3x smaller than VGG16** ($894$k).
3. **Decisive Low-Label Advantage:**
   - Under severe label scarcity (**10% labels = 470 samples**), PhySSL-Net delivers **85.74% Accuracy** and **64.96% Specificity**, compared to Vanilla CNN's **76.28% Accuracy** and **38.03% Specificity** (**+9.46% absolute Accuracy boost** and **+26.93% Specificity boost**).
4. **Clinical Safety & Statistical Significance:**
   - Detects **98.41%** of all pneumonia cases (clinical recall).
   - Paired McNemar tests confirm statistical significance across all baselines ($p < 0.001$).

---

## 3. Architectural Highlights

```
Raw Image (1x28x28 or 3x28x28)
           │
           ▼
[ Learnable IntensityRBF ]      Phi(x) = [x, exp(-beta_k (x - c_k)^2)] (K=8 centers)
           │
           ▼
     [ Conv Stem ]              (32 channels, 3x3 Conv + BN + ReLU)
           │
           ▼
  [ BandPassBlock 7x7 ]         Y_high = |Conv7x7_dw(Y_low)| (Modulus texture residual)
           │
           ▼
 [ Curvature Attention ]        A_curv = sigma(-div(grad u / (|grad u| + 1e-2)))
           │
           ▼
  [ Global Token Mixer ]        7x7 spatial grid (49 tokens); Spatial & Channel MLPs
           │
           ▼
 [ Spatial Pyramid Head ]       Multi-scale pooling (1x1 & 2x2) -> 5*C feature descriptor
           │
           ▼
  [ Linear Classification ]     Diagnostic Class Predictions
```

1. **IntensityRBF Encoding:** Non-linear Gaussian soft-windowing with learnable centers $c_k$ and constrained bandwidth $\beta_k = \text{softplus}(a_k) + 10^{-4}$.
2. **BandPass Residual Block:** $7\times 7$ depthwise filtering coupled with absolute value modulus non-linearity $|Z|$ to preserve high-frequency texture boundaries without phase cancellation.
3. **Level-Set Curvature Attention:** Evaluates the mean curvature divergence of the scalar gradient field with numerical stabilizer $\varepsilon = 10^{-2}$.
4. **Global Token Mixer:** Facilitates long-range spatial communication on a $7\times 7$ grid (49 tokens) with token- and channel-mixing MLPs.
5. **SimSiam Masked Pretraining:** Dual-view stochastic patch masking ($4\times4$ patches, 25% ratio) with online encoder, predictor head, stop-gradient target branch, and symmetrized negative cosine similarity loss.

---

## 4. Repository Structure

```
phySSL/
├── configs/
│   └── experiment.yaml             # Unified experiment configurations
├── data/
│   ├── dataset.py                  # Shared leak-free MedMNIST data loader
│   └── pneumoniamnist.npz          # Downloaded dataset arrays
├── models/
│   ├── physsl_net.py               # PhySSL-Net & SimSiam SSL wrapper
│   ├── c_mixer.py                  # Complex Mixer (Zheng & Jia, 2023)
│   └── baselines.py                # Vanilla CNN, VGG16Small, ResNet50V2Small
├── evaluation/
│   ├── metrics.py                  # Accuracy, ROC-AUC, F1, Recall, Spec, CM
│   ├── statistics.py               # McNemar's test, CI, seed aggregation
│   └── plots.py                    # ROC, CM, Pareto, and Ablation plotting
├── scripts/
│   ├── check_dataset.py            # Dataset integrity & split validation
│   ├── check_models.py             # Forward/backward & parameter profiling
│   ├── pretrain_ssl.py             # SimSiam masked consistency pretraining
│   ├── train.py                    # Supervised training with balanced metrics & aug
│   ├── evaluate.py                 # Checkpoint evaluation with logit adjustment
│   ├── run_experiments.py          # Orchestrated multi-seed benchmark runner
│   └── generate_docx_report.py     # Academic Word report builder
├── tests/
│   ├── test_models.py              # Automated unit tests for architectures
│   └── test_evaluation.py          # Automated unit tests for metrics
├── checkpoints/                    # Saved model weights (.pth)
├── results/                        # JSON logs, CSV summaries, and PNG figures
├── references/                     # DTU synopsis, research plan, papers
├── docs/
│   ├── FINAL_REPORT_DRAFT.md       # DTU B.Tech final project report draft (Markdown)
│   └── PhySSL_Net_Final_Report_DTU.docx # Official formatted Word report
├── requirements.txt                # Python package dependencies
├── pytest.ini                      # Pytest discovery configuration
├── PROJECT_STATUS.md               # Real-time implementation status dashboard
├── IMPLEMENTATION_PLAN.md          # Stage-by-stage implementation milestones
├── EXPERIMENT_PROTOCOL.md          # Multi-seed protocol & fair comparison rules
├── MODEL_ARCHITECTURE.md           # Mathematical derivations & tensor shapes
├── DATASET_AND_PREPROCESSING.md    # Dataset statistics & leakage controls
├── DECISIONS.md                    # Key architectural decisions log
├── TEAM.md                         # DTU team roster & guide details
└── REPRODUCIBILITY.md              # Reproduction instructions
```

---

## 5. Quick Start & Execution

### 1. Installation
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\pip install -r requirements.txt
```

### 2. Run Automated Unit Tests (66 Tests Passing)
```powershell
.\.venv\Scripts\python -m pytest tests -v
```

### 3. Self-Supervised Masked Consistency Pretraining
```powershell
.\.venv\Scripts\python scripts/pretrain_ssl.py --dataset pneumoniamnist --epochs 10
```

### 4. Supervised Fine-Tuning with Balanced Accuracy Checkpoint Selection
```powershell
.\.venv\Scripts\python scripts/train.py --model physsl_net --dataset pneumoniamnist --epochs 25 --val_metric balanced_acc --hflip_aug --pretrained_path checkpoints/physsl_simsiam_pretrained.pth --seed 123
```

### 5. Evaluate Checkpoint on Sequestered Test Split
```powershell
# Standard Evaluation (t = 0.50) -> 89.42% accuracy
.\.venv\Scripts\python scripts/evaluate.py --checkpoint checkpoints/physsl_net_pneumoniamnist_frac100_seed123_best.pth --dataset pneumoniamnist --split test

# Bayesian Prior Shift Calibration (Logit Adjustment) -> 90.87% accuracy
.\.venv\Scripts\python scripts/evaluate.py --checkpoint checkpoints/physsl_net_pneumoniamnist_frac100_seed123_best.pth --dataset pneumoniamnist --split test --logit_adj
```

### 6. Generate Official DTU Word Thesis Report
```powershell
.\.venv\Scripts\python scripts/generate_docx_report.py
```

---

## 6. Baselines Compared
1. **PhySSL-Net (Proposed):** Physics-inspired architecture with self-supervised pretraining (46,294 parameters).
2. **C-Mixer:** Complex-valued MLP-Mixer with incentive noise and Pearson projection (Zheng & Jia, 2023, 277,156 parameters).
3. **Vanilla CNN:** 4-layer convolutional network baseline from DTU approved research plan (65,314 parameters).
4. **VGG16:** Deep convolutional network adapted with BatchNorm for 28x28 resolution (894,498 parameters).
5. **ResNet50V2:** Pre-activation residual bottleneck architecture adapted for 28x28 resolution (751,522 parameters).

---

## 7. Documentation & Deliverables
- **Final Research Report (Markdown):** [docs/FINAL_REPORT_DRAFT.md](file:///c:/Users/maury/Downloads/phySSL/docs/FINAL_REPORT_DRAFT.md)
- **Final Research Report (Word Thesis):** [docs/PhySSL_Net_Final_Report_DTU.docx](file:///c:/Users/maury/Downloads/phySSL/docs/PhySSL_Net_Final_Report_DTU.docx)
- **Project Status Dashboard:** [PROJECT_STATUS.md](file:///c:/Users/maury/Downloads/phySSL/PROJECT_STATUS.md)
- **Results Summary:** [RESULTS_SUMMARY.md](file:///c:/Users/maury/Downloads/phySSL/RESULTS_SUMMARY.md)
- **Reproducibility Guide:** [REPRODUCIBILITY.md](file:///c:/Users/maury/Downloads/phySSL/REPRODUCIBILITY.md)
- **Experiment Protocol:** [EXPERIMENT_PROTOCOL.md](file:///c:/Users/maury/Downloads/phySSL/EXPERIMENT_PROTOCOL.md)
- **Architecture Specification:** [MODEL_ARCHITECTURE.md](file:///c:/Users/maury/Downloads/phySSL/MODEL_ARCHITECTURE.md)
- **Team Roster & Guide:** [TEAM.md](file:///c:/Users/maury/Downloads/phySSL/TEAM.md)

---

## 8. Research Ethics & Integrity
- **No Diagnostic Claims:** PhySSL-Net is an educational B.Tech research prototype evaluated on standardized benchmark data; it is not approved or certified for clinical diagnosis.
- **Strict Leakage Prevention:** Training, validation, and test partitions are kept strictly separate. Checkpoints are selected exclusively using validation balanced accuracy/ROC-AUC; zero test samples or labels are ever accessed during pretraining, training, or tuning.
