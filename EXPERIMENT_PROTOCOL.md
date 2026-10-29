# Experiment Protocol: PhySSL-Net Research (EXPERIMENT_PROTOCOL.md)

## 1. Experimental Objectives
Evaluate the representation learning and downstream classification performance of **PhySSL-Net** against established baselines on standardized medical image classification benchmarks under:
1. Full labelled supervision.
2. Label-fraction regimes (100%, 50%, 25%, 10% labelled data).
3. Self-supervised pretraining vs. random initialization.
4. Component ablation studies isolating each proposed module.
5. Computational efficiency profiling (Parameters, MACs/FLOPs, Latency).
6. Statistical validation across multiple random seeds.

## 2. Dataset and Partitioning Protocol
- **Benchmark Family:** MedMNIST v2
- **Primary Benchmark:** **PneumoniaMNIST** (binary classification: Normal vs Pneumonia, resolution 1x28x28).
  - Training Set: 4,708 samples (used for unlabelled pretraining and supervised fine-tuning)
  - Validation Set: 524 samples (strictly for early stopping, hyperparameter checks, checkpoint selection)
  - Test Set: 624 samples (held-out for final inference only)
- **Extended Benchmarks:** **DermaMNIST** (3x28x28, 7 classes) & **PathMNIST** (3x28x28, 9 classes).
- **Leakage Prevention:**
  - Zero test image is ever passed to pretraining, normalization fitting, or hyperparameter selection.
  - Normalization statistics $(\mu, \sigma)$ are computed strictly on the training partition.
  - Fixed official MedMNIST partition IDs ensure perfect reproducibility and cross-model fairness.

## 3. Models Compared
1. **PhySSL-Net:** Physics-inspired neural network with IntensityRBF, 7x7 BandPassBlock, Level-Set Curvature Attention, 7x7 Global Token Mixer, and Spatial Pyramid Head.
2. **C-Mixer:** Complex-valued MLP-Mixer with incentive noise and Pearson projection (Zheng & Jia, 2023).
3. **VGG16:** Deep convolutional network with batch normalization adapted for 28x28 medical images.
4. **ResNet50V2:** Pre-activation residual network adapted for 28x28 medical images.
5. **Vanilla CNN:** Standard 4-layer convolutional baseline as approved in DTU research plan.

## 4. Training and Evaluation Regimes
### Phase 1: Self-Supervised Pretraining (PhySSL-Net & C-Mixer)
- **Data:** 100% of the training split, completely unlabelled (class labels discarded).
- **Transformation:** Dual stochastic masking with random cutout and medical-safe augmentations.
- **Objective:** Symmetrized negative cosine similarity loss with stop-gradient predictor.
- **Optimizer:** AdamW, learning rate $1\times 10^{-3}$, weight decay $1\times 10^{-4}$.
- **Batch Size:** 64 or 128 (hardware adapted).
- **Budget:** 10 epochs for pilot validation; 30-50 epochs for benchmark.

### Phase 2: Supervised Fine-Tuning / Baseline Training
- **Data:** Training split under specified label fraction (100%, 50%, 25%, 10%).
- **Objective:** Cross-Entropy loss (or BCEWithLogits for binary tasks).
- **Optimizer:** AdamW ($1\times 10^{-3}$) / SGD with momentum 0.9.
- **Scheduler:** Cosine annealing / MultiStepLR.
- **Checkpoint Selection:** Checkpoint maximizing validation balanced accuracy ($\frac{1}{2}(\text{Sensitivity} + \text{Specificity})$) or validation ROC-AUC.
- **Data Augmentation:** Random crop/resize and medical bilateral horizontal flipping (`--hflip_aug`).
- **Budget:** 25 epochs per run.

## 5. Random Seeds
To guarantee statistical significance and experimental reproducibility, all benchmark and ablation runs use 5 fixed seeds:
- `42`
- `123`
- `456`
- `789`
- `999`

## 6. Evaluation Metrics & Decision Calibration
- **Accuracy (ACC):** Fraction of correct predictions at default threshold $t=0.50$.
- **Bayesian Prior-Shift Adjusted Accuracy:** Fraction of correct predictions at threshold $t^* = \pi_{\text{train}} \approx 0.742$ (aligning training/testing class prevalence).
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve (threshold-independent).
- **Macro-Precision:** Unweighted average precision across all classes.
- **Macro-Recall (Sensitivity):** Unweighted average recall across all classes ($\frac{TP}{TP + FN}$).
- **Specificity:** True negative rate ($\frac{TN}{TN + FP}$).
- **Macro-F1:** Harmonic mean of macro-precision and macro-recall.
- **Confusion Matrix:** True vs predicted class frequencies.
- **Multi-Seed Ensembling:** Soft-voting probability averaging across the 5 independent seeds.

## 7. Ablation Plan
Evaluate the contribution of each architectural innovation in PhySSL-Net on the primary benchmark:
1. **Full PhySSL-Net** (All components enabled).
2. **w/o IntensityRBF** (Standard convolutional input stem).
3. **w/o BandPassBlock** (Standard 3x3 conv block without modulus high-pass filtering).
4. **w/o Curvature Attention** (Uniform spatial weighting without differential geometry).
5. **w/o Global Token Mixer** (Standard convolutional pooling without spatial MLP mixing).
6. **w/o SSL Pretraining** (Trained from scratch with random initialization).

## 8. Statistical Rigor
- Report **Mean $\pm$ Standard Deviation** across the 5 seeds for all primary metrics.
- Perform **McNemar's test** for paired classification error discordance on the test partition.
- Compute 95% Confidence Intervals for test accuracy and ROC-AUC.
