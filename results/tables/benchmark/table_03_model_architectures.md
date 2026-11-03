### TABLE 3: Model Architecture and Parameters

| Architecture | Design Philosophy | Stem / Encoding | High-Frequency Block | Attention Mechanism | Token Mixing | Head | Parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PhySSL-Net (Proposed) | Physics-Informed + Self-Supervised | AdaptiveSplineGaussianRBF ([-1, 1]) | 7x7 DW-Conv + Modulus |Z| | Mean Curvature Level-Set (Sobel / 8.0) | Spatial 7x7 Grid MLP | Spatial Pyramid Pooling | 46,318 |
| ConvNeXt-Femto | Modern Pure CNN (CVPR 2022) | Standard 3x3 Conv Stem | Inverted Bottleneck 7x7 DW | None (GELU + LayerNorm) | Channels-first 1x1 MLP | Global Average Pooling | 109,890 |
| Vanilla CNN | Standard 4-layer Convolutional | Standard 3x3 Conv Stem | None (Standard 3x3) | None | None | Global Average Pooling | 65,314 |
| VGG16 (Small) | Deep Homogeneous Conv (3x3 with BN) | Standard 3x3 Conv Stem | None | None | None | Global Average Pooling + MLP | 894,498 |
| ResNet50V2 (Small) | Pre-activation Residual Bottleneck | Standard 3x3 Conv Stem | None | None | None | Global Average Pooling | 751,522 |