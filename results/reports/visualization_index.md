# PhySSL-Net Research Visualization Index & Scientific Catalog
**Project:** Physics- and Structure-Informed Self-Supervised Learning for Medical Image Classification  
**Group:** `DTU/CSE/BTP/GP053` (AY 2026–27) | **Guide:** Dr. Ravin Ahuja  

| Fig # | Filename | Figure Title | Research Purpose | Data Source | Relevant Table | Verification Status |
|:---:|:---|:---|:---|:---|:---:|:---:|
| **Fig 01** | `fig_01_dataset_distribution.png` | Class Distribution | Quantify class imbalance | PneumoniaMNIST Metadata | Table 1 | COMPLETE |
| **Fig 02** | `fig_02_dataset_split_distribution.png` | Dataset Split Distribution | Verify official Train/Val/Test partitioning | Split NPZ Files | Table 2 | COMPLETE |
| **Fig 03** | `fig_03_sample_medical_images.png` | Representative Medical Samples | Validate visual clarity and pathology presentation | Raw Test X-Rays | Table 1 | COMPLETE |
| **Fig 04** | `fig_04_preprocessing_comparison.png` | Preprocessing Comparison | Document uint8 to [-1, 1] normalization | PyTorch Pipeline | Table 1 | COMPLETE |
| **Fig 05** | `fig_05_training_loss_curves.png` | Training Loss Convergence | Track optimization stability across epochs | Epoch-wise Logs | Table 4 | COMPLETE |
| **Fig 06** | `fig_06_validation_loss_curves.png` | Validation Loss Tracking | Monitor validation trajectory across architectures | Epoch-wise Logs | Table 4 | COMPLETE |
| **Fig 07** | `fig_07_validation_accuracy_curves.png` | Validation Accuracy Trajectory | Assess convergence speed across architectures | Epoch-wise Logs | Table 4 | COMPLETE |
| **Fig 08** | `fig_08_overfitting_analysis.png` | Generalization Gap Dynamics | Evaluate generalization margin ($Loss_{val}-Loss_{train}$) | Epoch-wise Logs | Table 5 | COMPLETE |
| **Fig 09** | `fig_09_ssl_pretrain_loss.png` | SimSiam SSL Pretraining Curve | Confirm Phase 1 self-supervised representation learning | `ssl_pretrain_log.json` | Table 9 | COMPLETE |
| **Fig 10** | `fig_10_model_accuracy_comparison.png` | 5-Seed Accuracy Comparison | Benchmark proposed model against baselines | Consolidated Eval Logs | Table 5 | COMPLETE |
| **Fig 11** | `fig_11_roc_comparison.png` | ROC Curves & AUC Analysis | Evaluate sensitivity vs specificity discrimination | Held-Out Test Predictions | Table 4 | COMPLETE |
| **Fig 12** | `fig_12_metrics_grouped_comparison.png` | Multi-Metric Profile | Grouped comparison of Acc, AUC, F1, Specificity | 5-Seed Averages | Table 5 | COMPLETE |
| **Fig 13** | `fig_13_confusion_matrix_comparison.png` | Test Confusion Matrices | Inspect class-level diagnostic confusion | Test Prediction Matrices | Table 4 | COMPLETE |
| **Fig 14** | `fig_14_seed_variability.png` | Seed Stability & Variability | Assess stability across 5 random initializations | Seeds 42, 123, 456, 789, 999 | Table 5 | COMPLETE |
| **Fig 15** | `fig_15_ablation_comparison.png` | Ablation Metrics Comparison | Quantify impact of each architectural component | Modular Ablation Runs | Table 6 | COMPLETE |
| **Fig 16** | `fig_16_ablation_parameters.png` | Parameter Count Across Ablations | Track parameter budget per sub-module | Architectural Inspection | Table 6 | COMPLETE |
| **Fig 17** | `fig_17_component_removal_impact.png` | Component Removal Impact (Delta %) | Prove necessity of Curvature, RBF, and Mixer | Ablation Deltas | Table 6 | COMPLETE |
| **Fig 18** | `fig_18_accuracy_vs_parameters.png` | Accuracy vs Parameter Budget | Analyze parameter efficiency (log scale) | Efficiency Profile | Table 7 | COMPLETE |
| **Fig 19** | `fig_19_accuracy_vs_latency.png` | Accuracy vs Inference Latency | Evaluate clinical real-time deployment viability | Benchmarked Latencies | Table 7 | COMPLETE |
| **Fig 20** | `fig_20_pareto_frontier.png` | Pareto Optimal Frontier | Identify non-dominated efficiency boundary | Multi-objective Tradeoff | Table 7 | COMPLETE |
| **Fig 21** | `fig_21_statistical_significance.png` | McNemar Significance Testing | Rigorously test hypothesis against deep baselines | Paired Contingency Tables | Table 8 | COMPLETE |

---

## Deliverables Checklist
- [x] **Figure Formats:** High-resolution PNG (300 DPI), Vector PDF, Scalable SVG.
- [x] **Table Formats:** CSV, Excel (.xlsx), GitHub Markdown (.md), Academic LaTeX (.tex).
- [x] **Statistical Testing:** McNemar Paired $\chi^2$ Test, Exact Binomial P-values, 95% CIs.
- [x] **Reproducibility:** Seed configuration, deterministic verification, exact parameter counts.
