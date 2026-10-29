"""
Baseline Architectures for Medical Image Classification Benchmark:
1. Vanilla CNN (4-layer Convolutional Network)
2. VGG16 (Adapted with BatchNorm for 28x28 medical images)
3. ResNet50V2 (Adapted Pre-activation ResNet for 28x28 medical images)
4. ConvNeXt-Femto (CVPR 2022 modern pure-CNN baseline adapted for 28x28 medical images)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Vanilla CNN Baseline
# ==============================================================================
class VanillaCNN(nn.Module):
    """
    Standard 4-layer Convolutional Neural Network baseline.
    Input: (B, C_in, 28, 28)
    Stem -> Conv Block 1 -> Pool -> Conv Block 2 -> Pool -> GAP -> Linear
    """
    def __init__(self, in_channels=1, num_classes=2, base_channels=32, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: 28x28 -> 14x14
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 2: 14x14 -> 7x7
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base_channels * 2, num_classes),
        )

    def extract_features(self, x):
        feat = self.features(x)
        return self.gap(feat).flatten(1)

    def forward(self, x):
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits


# ==============================================================================
# 2. VGG16 Adapted Baseline
# ==============================================================================
class VGG16Small(nn.Module):
    """
    VGG16 architecture with Batch Normalization adapted for 28x28 resolution:
    - Replaces large 7x7 first conv with 3x3 conv preserving spatial dimensions.
    - Uses 5 convolutional stages with [2, 2, 3, 3, 3] layers.
    - Global average pooling followed by linear classification head.
    """
    def __init__(self, in_channels=1, num_classes=2, dropout=0.2):
        super().__init__()
        cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M']
        layers = []
        c_in = in_channels
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                    nn.Conv2d(c_in, v, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ])
                c_in = v
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def extract_features(self, x):
        feat = self.features(x)
        return self.gap(feat).flatten(1)

    def forward(self, x):
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits


# ==============================================================================
# 3. ResNet50V2 Adapted Baseline
# ==============================================================================
class PreActBottleneck(nn.Module):
    """
    Pre-activation bottleneck block for ResNetV2:
    BN -> ReLU -> 1x1 Conv -> BN -> ReLU -> 3x3 Conv -> 1x1 Conv + Skip
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)

        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet50V2Small(nn.Module):
    """
    ResNet50V2 adapted for 28x28 medical images:
    Bottleneck stages [3, 4, 6, 3] with base planes=16 (to manage capacity for 28x28 inputs).
    """
    def __init__(self, in_channels=1, num_classes=2, dropout=0.2):
        super().__init__()
        self.in_planes = 32
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)

        # 4 stages with bottleneck blocks
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 4, stride=2)
        self.layer3 = self._make_layer(64, 6, stride=2)
        self.layer4 = self._make_layer(64, 3, stride=1)

        self.bn = nn.BatchNorm2d(64 * PreActBottleneck.expansion)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(64 * PreActBottleneck.expansion, num_classes),
        )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(PreActBottleneck(self.in_planes, planes, s))
            self.in_planes = planes * PreActBottleneck.expansion
        return nn.Sequential(*layers)

    def extract_features(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        return self.gap(out).flatten(1)

    def forward(self, x):
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits

# ==============================================================================
# 4. ConvNeXt-Femto Baseline (Liu et al., CVPR 2022)
# ==============================================================================
class LayerNormChannelsFirst(nn.Module):
    """
    LayerNorm that supports (B, C, H, W) channels-first format.
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block (Liu et al., CVPR 2022).
    Structure:
    1. 7x7 Depthwise Conv
    2. LayerNorm (channels-first)
    3. 1x1 Pointwise Conv (inverted bottleneck expansion 4x)
    4. GELU
    5. 1x1 Pointwise Conv (projection back to C)
    6. LayerScale (gamma parameter)
    7. Residual shortcut
    """
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNormChannelsFirst(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return input + x


class ConvNeXtFemto(nn.Module):
    """
    ConvNeXt-Femto (CVPR 2022 baseline adapted for 28x28 medical images).
    - Lightweight, modern pure convolutional architecture (~109k parameters).
    - Depths: [2, 2, 2] across 3 stages
    - Channel dimensions: [16, 32, 64]
    - Inverted bottleneck with 7x7 depthwise convolutions and LayerScale.
    """
    def __init__(self, in_channels=1, num_classes=2, depths=[2, 2, 2], dims=[16, 32, 64], dropout=0.2):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        # Stem: 28x28 -> 28x28 (stride 1 for small 28x28 inputs)
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1),
            LayerNormChannelsFirst(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)

        # Downsample stages: 28x28 -> 14x14 -> 7x7
        for i in range(len(dims) - 1):
            downsample_layer = nn.Sequential(
                LayerNormChannelsFirst(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Stages with ConvNeXt blocks
        self.stages = nn.ModuleList()
        for i in range(len(dims)):
            stage = nn.Sequential(*[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)

        self.norm = LayerNormChannelsFirst(dims[-1], eps=1e-6)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(dims[-1], num_classes)
        )

    def extract_features(self, x):
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x)
        return self.gap(x).flatten(1)

    def forward(self, x):
        feat = self.extract_features(x)
        return self.classifier(feat)


__all__ = ['VanillaCNN', 'VGG16Small', 'ResNet50V2Small', 'ConvNeXtFemto', 'ConvNeXtBlock', 'LayerNormChannelsFirst']

