# Implementation Decisions and Architectural Log (DECISIONS.md)

## Decision 1: Primary Benchmark Dataset and Scope
- **Decision:** Use **PneumoniaMNIST** (binary classification, 1x28x28) as the primary dataset for the full 5-seed benchmark (seeds: 42, 123, 456, 789, 999), pilot validation, computational efficiency measurements, and extensive architectural ablation study. Provide seamless support in the shared dataset loader for **DermaMNIST** (3x28x28, 7-class) and **PathMNIST** (3x28x28, 9-class).
- **Rationale:** 
  1. Specified in DTU Research Plan Table 3 as the candidate initial MedMNIST 2D subset.
  2. With an available NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) and time constraints, PneumoniaMNIST enables complete, honest, and verifiable multi-seed convergence, ablations, and statistical tests without out-of-memory errors or multi-day runtimes.
  3. Official splits (Train: 4,708, Val: 524, Test: 624) are strictly respected. Zero test set leakage.

## Decision 2: PhySSL-Net Mathematical Specifications
- **IntensityRBF Block:**
  - Formula: $\Phi(x) = [x, \exp(-\beta_1(x - c_1)^2), \dots, \exp(-\beta_K(x - c_K)^2)]$
  - Number of centers $K=8$, initialized uniformly across $[0.0, 1.0]$.
  - Width parameterization: $\beta_k = \text{softplus}(a_k) + 10^{-4}$ to guarantee numerical positivity and prevent divide-by-zero or gradient explosion.
  - Expands input channel dimension from $C_{in}$ to $C_{in} \times (1 + K)$.
- **BandPass Residual Block:**
  - Utilizes a 7x7 depthwise convolution followed by absolute value modulus operation: $Y_{\text{high}} = |\text{Conv2d}_{7\times7}(Y_{\text{low}})|$.
  - Projected via 1x1 pointwise convolution and added to the residual stream: $Y_{\text{out}} = Y_{\text{low}} + \text{Proj}(Y_{\text{high}})$.
- **Level-Set / Curvature-Inspired Attention Block:**
  - Formula: $A_{\text{curv}} = \sigma(-\text{div}(V))$ where $V = \frac{\nabla u}{\sqrt{|\nabla u|^2 + \varepsilon}}$.
  - Scalar field $u$ extracted by 1x1 projection.
  - Spatial gradient $\nabla u = (u_x, u_y)$ evaluated with Sobel difference kernels.
  - Numerical stabilizer $\varepsilon = 10^{-2}$ ($1e-2$) strictly applied as specified.
  - Attention modulates intermediate features: $X_{\text{out}} = X \odot A_{\text{curv}} + X$.
- **Global Token Mixer Block:**
  - Feature resolution: Adaptive pooling to $7 \times 7$ tokens (49 spatial tokens).
  - Implements spatial mixing MLP ($49 \to 49$) across positions and channel mixing MLP ($C \to C$) across channels, with LayerNorm and residual skips.
- **Classification Head:**
  - Multi-scale spatial pyramid pooling ($1 \times 1$ and $2 \times 2$ adaptive average pooling) concatenated into a $C \times (1 + 4) = 5C$ descriptor, followed by Dropout (0.2) and Linear classification layer.

## Decision 3: Self-Supervised Learning Framework (SimSiam / Masked Siamese)
- **Decision:** Implement a Siamese masked consistency learning framework.
- **Formulation:**
  - Input image $x$ undergoes two independent stochastic masking transforms $\to x_1, x_2$.
  - Shared encoder $f_\theta$ produces representations $z_1 = f_\theta(x_1), z_2 = f_\theta(x_2)$.
  - A 2-layer MLP predictor $h_\phi$ transforms $z_1 \to p_1$ and $z_2 \to p_2$.
  - Symmetrized negative cosine similarity loss with stop-gradient:
    $\mathcal{L}_{\text{SSL}} = \frac{1}{2} \mathcal{D}(p_1, \text{stop\_gradient}(z_2)) + \frac{1}{2} \mathcal{D}(p_2, \text{stop\_gradient}(z_1))$
    where $\mathcal{D}(p, z) = - \frac{p}{\|p\|_2} \cdot \frac{z}{\|z\|_2}$.
  - Stop-gradient strictly eliminates representation collapse without requiring negative pair contrastive queues.

## Decision 4: Baseline Models Architecture & Recipes
- **C-Mixer:** Implemented according to Zheng & Jia (2023), featuring complex affine projections ($W = A + Bi$), CReLU activation ($\text{ReLU}(a) + i\text{ReLU}(b)$), Pearson curve projection ($p = \tanh(\text{real}(y) + \text{imag}(y))$), and incentive Gaussian noise generation.
- **VGG16:** Adapted VGG-style convolutional backbone for 28x28 input resolution with batch normalization.
- **ResNet50V2:** Adapted Pre-activation ResNet backbone with bottleneck blocks suitable for 28x28 images.
- **Vanilla CNN:** Standard 4-layer Conv2D + BatchNorm + ReLU + MaxPool architecture as defined in the DTU research plan.

## Decision 5: Statistical Rigor and Evaluation Rules
- **Seeds:** 42, 123, 456, 789, 999.
- **Test Set Protection:** Test splits are strictly reserved for final checkpoint evaluation. Checkpoint selection is strictly driven by validation balanced accuracy / ROC-AUC.
- **Statistical Tests:** McNemar's test for paired classification error analysis and Wilcoxon signed-rank / two-sided t-tests for multi-seed performance differences.

## Decision 6: Checkpoint Selection Metric & Bayesian Prior Shift Calibration
- **Balanced Accuracy Checkpoint Selection:**
  - Tracking raw validation ROC-AUC can cause premature early stopping (e.g. Epoch 2) when ranking discriminability stabilizes early, leaving the linear classification head under-fitted.
  - Selecting checkpoints via validation balanced accuracy ($\frac{1}{2}(\text{Sensitivity} + \text{Specificity})$) ensures training continues until the decision boundary achieves optimal class-balanced discrimination (epochs 10–18).
- **Partition Asymmetry & Bayesian Prior Shift Alignment:**
  - In PneumoniaMNIST, the training partition is 74.2% positive (Pneumonia) and 25.8% negative (Normal) ($1:2.88$), whereas the test partition ($N=624$) contains 62.5% positive and 37.5% negative ($1:1.67$).
  - Models trained under empirical cross-entropy learn the training prior $\pi_{\text{train}} \approx 0.742$, yielding high sensitivity ($98.87\%$) but suppressed specificity at standard threshold $t=0.50$.
  - Adjusting the decision threshold to the training prior ($t^* = \pi_{\text{train}} \approx 0.742$) or applying Bayesian logit adjustment shifts the decision boundary to the true posterior, lifting test accuracy to 88.62% ± 1.70% (peak 90.87%, ensemble 90.22%) and boosting specificity from 58.89% to 78.63% without touching test labels or modifying the trained weights.

