# Model Architecture Specification: PhySSL-Net (MODEL_ARCHITECTURE.md)

## 1. Architectural Philosophy
Standard vision architectures often overlook domain-specific physical and geometric properties inherent in medical imaging, such as tissue density distributions, boundary curvatures, and multi-scale frequency components. **PhySSL-Net** addresses this gap by coupling physics- and structure-inspired priors with self-supervised representation learning:
1. **Intensity-Aware Soft-Windowing (IntensityRBF):** Expands single-channel radiographic or histological intensities into non-linear Gaussian response channels.
2. **Frequency/Texture Extraction (BandPassBlock):** Applies 7x7 depthwise filtering and modulus non-linearities to capture high-frequency textural boundaries.
3. **Differential Geometry Attention (Curvature Attention):** Derives level-set curvature from feature gradients to focus network capacity on sharp geometric contours.
4. **Long-Range Spatial Communication (Global Token Mixer):** Operates on a 7x7 token grid to model long-range anatomical context beyond local receptive fields.
5. **Multi-Scale Evidence Aggregation (Spatial Pyramid Head):** Aggregates representations across multi-scale spatial resolutions ($1\times1, 2\times2$).

```
[Input Image (1x28x28 or 3x28x28)]
             │
             ▼
   [ IntensityRBF Block ]  (K=8 centers, softplus width)
             │
             ▼
      [ Conv Stem ]        (32 channels, 3x3 Conv + BN + ReLU)
             │
             ▼
   [ BandPassBlock 7x7 ]   (Depthwise 7x7 + Modulus |Z| + 1x1 Proj + Skip)
             │
             ▼
  [ Curvature Attention ]  (Level-set div(∇u / (|∇u| + 1e-2)) + Sigmoid)
             │
             ▼
  [ Conv Stage 2 (64ch) ]  (Downsample to 7x7 spatial resolution)
             │
             ▼
  [ Global Token Mixer ]   (7x7 grid = 49 tokens; Spatial MLP + Channel MLP)
             │
             ▼
  [ Spatial Pyramid Head ] (Adaptive AvgPool 1x1 & 2x2 -> 5*C descriptor)
             │
             ▼
     [ Linear Classifier ] (Logits for downstream classes)
```

---

## 2. Component Mathematical Formulations

### 2.1 Learnable Intensity RBF Encoding
Given normalized input intensity $x \in [0, 1]$ with $C_{\text{in}}$ channels:
$$\Phi(x) = \left[x, \exp(-\beta_1 (x - c_1)^2), \exp(-\beta_2 (x - c_2)^2), \dots, \exp(-\beta_K (x - c_K)^2)\right]$$
- **Centres ($c_k$):** $K=8$ centers uniformly spaced in $[0.0, 1.0]$.
- **Bandwidth ($\beta_k$):** Parameterized as $\beta_k = \text{softplus}(a_k) + 10^{-4}$ ensuring strict positivity and numerical stability.
- **Output Channels:** $C_{\text{out}} = C_{\text{in}} \times (1 + K)$. For 1-channel images, the stem receives 9 channels.

### 2.2 Band-Pass Residual Block
Captures micro-textural variations and high-frequency edge information:
$$Z = \text{DepthwiseConv2d}_{7\times7}(Y_{\text{low}})$$
$$Y_{\text{high}} = |Z|$$
$$Y_{\text{out}} = Y_{\text{low}} + \text{BN}(\text{Conv2d}_{1\times1}(Y_{\text{high}}))$$
By applying the modulus operation $|Z|$, the block acts analogously to a directional wavelet scattering modulus, capturing localized gradient magnitude signatures without phase cancellations.

### 2.3 Level-Set / Curvature-Inspired Attention
Derives spatial attention from the mean curvature of the feature field level-sets:
1. Feature projection to scalar field: $u = \text{Conv2d}_{1\times1}(X) \in \mathbb{R}^{B \times 1 \times H \times W}$.
2. Spatial derivatives: $u_x = u * K_x$, $u_y = u * K_y$ using Sobel finite-difference operators.
3. Gradient magnitude with stabilizer ($\varepsilon = 10^{-2}$):
   $$|\nabla u|_\varepsilon = \sqrt{u_x^2 + u_y^2 + \varepsilon}$$
4. Normalized gradient vector field:
   $$V = (V_x, V_y) = \left(\frac{u_x}{|\nabla u|_\varepsilon}, \frac{u_y}{|\nabla u|_\varepsilon}\right)$$
5. Level-set divergence (mean curvature):
   $$\kappa = \text{div}(V) = \frac{\partial V_x}{\partial x} + \frac{\partial V_y}{\partial y} = (V_x * K_x) + (V_y * K_y)$$
6. Attention map and residual modulation:
   $$A_{\text{curv}} = \sigma(-\kappa)$$
   $$X_{\text{out}} = X \odot A_{\text{curv}} + X$$

### 2.4 Global Token Mixer
Operates at the $7 \times 7$ spatial resolution ($N = 49$ tokens, hidden channel dimension $C = 64$):
1. Reshape: $X \in \mathbb{R}^{B \times C \times 7 \times 7} \to \hat{X} \in \mathbb{R}^{B \times 49 \times C}$.
2. **Token Mixing:**
   $$U = \hat{X} + \left(\text{MLP}_{\text{token}}\left(\text{LN}(\hat{X})^T\right)\right)^T$$
3. **Channel Mixing:**
   $$Y = U + \text{MLP}_{\text{channel}}(\text{LN}(U))$$
4. Reshape back: $Y \in \mathbb{R}^{B \times 49 \times C} \to Y \in \mathbb{R}^{B \times C \times 7 \times 7}$.

### 2.5 Spatial Pyramid Head
Aggregates coarse and fine spatial statistics:
- Branch 1: $\text{AdaptiveAvgPool2d}(1, 1) \to B \times C \times 1 \times 1$ ($C$ features)
- Branch 2: $\text{AdaptiveAvgPool2d}(2, 2) \to B \times C \times 2 \times 2$ ($4C$ features)
- Concatenation: $F = [F_1, F_2] \in \mathbb{R}^{B \times 5C}$
- Classification: $\hat{y} = \text{Linear}(5C \to N_{\text{classes}})(\text{Dropout}(F, p=0.2))$

---

## 3. Self-Supervised Learning Framework (PhySSL-SimSiam)
- **Masking Mechanism:** Random patch cutout with patch size $4 \times 4$ and masking ratio $25\%$.
- **Dual Views:** $x_1 = \text{Mask}_1(x)$, $x_2 = \text{Mask}_2(x)$.
- **Projection Head ($g$):** 2-layer MLP ($5C \to 128 \to 128$) with BatchNorm and ReLU.
- **Predictor Head ($h$):** 2-layer MLP ($128 \to 64 \to 128$) with BatchNorm and ReLU.
- **Loss Formulation:**
  $$\mathcal{D}(p, z) = - \frac{p}{\|p\|_2} \cdot \frac{z}{\|z\|_2}$$
  $$\mathcal{L}_{\text{SSL}} = \frac{1}{2}\mathcal{D}(p_1, \text{stop\_gradient}(z_2)) + \frac{1}{2}\mathcal{D}(p_2, \text{stop\_gradient}(z_1))$$
  where $z_i = g(f_\theta(x_i))$ and $p_i = h(z_i)$.
