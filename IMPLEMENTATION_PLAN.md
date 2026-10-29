# Implementation Plan: PhySSL-Net Research (IMPLEMENTATION_PLAN.md)

## Stage A: Repository and Resource Inspection [COMPLETED]
- [x] Scan workspace and parse reference documents:
  - `PhySSL_Architecture_Specification.docx`
  - `DTU_Final_Year_Minor_Project_Detailed_Research_Plan.pdf`
  - `C_Mixer_Reference_Paper.pdf`
  - `NEW DOCS FILE BTech minor project print new.docx`
- [x] Extract DTU academic project metadata (Group `DTU/CSE/BTP/GP053`, Guide Dr. Ravin Ahuja, Students Manish Singh, Shivam, Siddharth Kumar Maurya, Saurav Kumar).
- [x] Profile local hardware and software environment (NVIDIA RTX 3050 4GB GPU, Python 3.12).
- [x] Establish core project governance docs: `TEAM.md`, `DECISIONS.md`, `EXPERIMENT_PROTOCOL.md`, `PROJECT_STATUS.md`.

## Stage B: Environment, Configuration & Data Pipeline [IN PROGRESS]
- [ ] Finalize Python virtual environment `.venv` with PyTorch (CUDA enabled), Torchvision, MedMNIST, Scikit-learn, Matplotlib, PyYAML, SciPy.
- [ ] Create `configs/experiment.yaml` encapsulating:
  - Dataset parameters (PneumoniaMNIST, DermaMNIST, PathMNIST).
  - Model architectures and block switches.
  - SSL SimSiam training hyperparameters.
  - Fine-tuning and baseline hyperparameters.
  - Evaluation thresholds, metrics, and seeds (`[42, 123, 456, 789, 999]`).
- [ ] Implement `data/dataset.py`:
  - Deterministic MedMNIST loading and tensor conversion.
  - Strict preservation of official train/val/test splits.
  - Unlabelled split creation (stripping labels for SSL pretraining).
  - Label fraction generator (100%, 50%, 25%, 10%) with stratified subsampling.
  - Dual-view stochastic masking transforms for SSL.
- [ ] Implement `scripts/check_dataset.py` to assert data integrity, shape sanity, label mappings, and class distributions.

## Stage C: PhySSL-Net Architecture Implementation
- [ ] Implement `models/physsl_net.py`:
  - `IntensityRBF`: Learnable Gaussian RBF encoding with softplus width constraints.
  - `BandPassBlock`: 7x7 depthwise convolution with modulus residual extraction.
  - `CurvatureAttention`: Level-set divergence of normalized gradient with $\varepsilon = 10^{-2}$.
  - `GlobalTokenMixer`: 7x7 spatial token mixing and channel mixing with residual connections.
  - `SpatialPyramidHead`: Multi-scale spatial pooling ($1\times1, 2\times2$) and classification projection.
  - `PhySSLSimSiam`: Self-supervised dual-branch wrapper with online encoder, predictor, and stop-gradient.
  - Model parameter count and forward pass shape checks.

## Stage D: Baseline Model Implementations
- [ ] Implement `models/c_mixer.py`:
  - Complex affine transformations ($W = A + Bi$).
  - CReLU activation ($\text{ReLU}(a) + i\text{ReLU}(b)$).
  - Incentive learning Gaussian noise generation.
  - Generalized Pearson curve projection back to real space.
- [ ] Implement `models/baselines.py`:
  - `VanillaCNN`: 4-layer convolutional network.
  - `VGG16Small`: Adapted VGG16 with BatchNorm for 28x28 inputs.
  - `ResNet50V2Small`: Adapted Pre-activation ResNet for 28x28 inputs.

## Stage E: Training, Pretraining & Evaluation Engines
- [ ] Implement `evaluation/metrics.py`:
  - Accuracy, ROC-AUC, Macro-Precision, Macro-Recall, Macro-F1, Specificity, Confusion Matrix.
  - Perfect synthetic score checks and missing class handling.
- [ ] Implement `evaluation/statistics.py`:
  - Mean $\pm$ Std across seeds.
  - McNemar's test for paired classification discordance on test set.
  - 95% Confidence Intervals.
- [ ] Implement `evaluation/plots.py`:
  - Loss curves (train/val).
  - ROC curves with AUC annotations.
  - Confusion matrix heatmaps.
  - Pareto parameter count vs. performance plots.
- [ ] Implement `scripts/train.py` (Supervised training engine with validation-based checkpointing).
- [ ] Implement `scripts/pretrain_ssl.py` (Self-supervised pretraining engine).
- [ ] Implement `scripts/evaluate.py` (Evaluation engine for held-out test splits).

## Stage F: Unit Tests & Smoke Pilot Validation
- [ ] Implement `tests/test_models.py` (forward pass, backward pass, gradient flow, parameter counts).
- [ ] Implement `tests/test_evaluation.py` (metric calculations, edge cases, deterministic evaluation).
- [ ] Execute `pytest` test suite.
- [ ] Run end-to-end smoke pilot (1 epoch SSL pretrain, 1 epoch fine-tuning, evaluation).

## Stage G: Full Benchmark & Ablation Study Execution
- [ ] Run benchmark on PneumoniaMNIST across 5 seeds (`[42, 123, 456, 789, 999]`):
  - PhySSL-Net (Pretrained + Fine-tuned)
  - PhySSL-Net (Scratch / No pretraining)
  - C-Mixer
  - Vanilla CNN
  - VGG16
  - ResNet50V2
- [ ] Run architectural ablations on PhySSL-Net:
  - w/o IntensityRBF
  - w/o BandPassBlock
  - w/o Curvature Attention
  - w/o Global Token Mixer
- [ ] Run label fraction study (10%, 25%, 50%, 100% labelled data).

## Stage H: Computational Efficiency & Analysis
- [ ] Profile Parameter count, FLOPs/MACs, CPU inference latency, GPU inference latency, memory footprint.
- [ ] Compute McNemar's test p-values against baselines.
- [ ] Generate all publication-quality figures and tables in `results/`.

## Stage I: Final Documentation and DTU B.Tech Report
- [ ] Update `PROJECT_STATUS.md` with verified experimental outputs.
- [ ] Create `MODEL_ARCHITECTURE.md`, `DATASET_AND_PREPROCESSING.md`, `REPRODUCIBILITY.md`, `RESULTS_SUMMARY.md`.
- [ ] Author comprehensive `docs/FINAL_REPORT_DRAFT.md` following DTU Department of CSE requirements.
- [ ] Update `README.md` with complete usage instructions.
