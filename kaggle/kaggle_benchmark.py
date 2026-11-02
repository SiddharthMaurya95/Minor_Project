"""
PhySSL-Net: Physics- and Structure-Informed Self-Supervised Learning
Automated Multi-Dataset Benchmark Runner for MedMNIST v2 on Kaggle (TPU v5e-8 / GPU / CPU).

DTU B.Tech Research Project (AY 2026-27, Group DTU/CSE/BTP/GP053)
Project Guide: Dr. Ravin Ahuja
Team Members: Manish Singh, Shivam, Siddharth Kumar Maurya, Saurav Kumar
"""

import os
import sys
import time
import math
import random
import urllib.request
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_curve,
    auc,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# HARDWARE ACCELERATOR DETECTION: Google Cloud TPU v5e-8 / GPU / CPU
# =====================================================================
HAS_TPU = False
xm = None
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    HAS_TPU = True
    print(f"[*] Active Hardware Backend: Google Cloud TPU ({device}) via PyTorch/XLA")
except Exception:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[*] TPU not active. Active Hardware: {device} ({gpu_name})")

try:
    import medmnist
    from medmnist import INFO
    MEDMNIST_AVAILABLE = True
except ImportError:
    MEDMNIST_AVAILABLE = False
    print("[!] MedMNIST not installed via pip. Using direct NPZ dataset downloader fallback.")


# =====================================================================
# 1. MODEL ARCHITECTURES (100% IDENTICAL TO LOCAL REPO SPECIFICATION)
# =====================================================================

class AdaptiveSplineGaussianRBF(nn.Module):
    """
    Unified Ramanujan-Schoenberg Inspired Basis:
    - Acts as pure Gaussian RBF when gamma -> 0 and p -> 2 (best for small/unbalanced data)
    - Morphs into Asymmetric Rational/Polynomial Spline when data is abundant (PathMNIST/BloodMNIST)
    """
    def __init__(self, in_channels, num_centers=8, eps=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.K = num_centers
        self.eps = eps

        # 1. Learnable Centers c_k evenly initialized in [0, 1]
        centers = torch.linspace(0.0, 1.0, num_centers).view(1, 1, num_centers, 1, 1)
        self.centers = nn.Parameter(centers.repeat(1, in_channels, 1, 1, 1))

        # 2. Learnable Widths beta_k (constrained positive via softplus)
        self.raw_beta = nn.Parameter(torch.ones(1, in_channels, num_centers, 1, 1) * 2.0)

        # 3. Learnable Shape/Order Parameter p_k (Initialized so p_k = 2.0 -> Gaussian)
        self.raw_p = nn.Parameter(torch.zeros(1, in_channels, num_centers, 1, 1))

        # 4. Spline Polynomial Modulation Coefficients (Initialized to 0 -> starts as pure Gaussian!)
        self.gamma_linear = nn.Parameter(torch.zeros(1, in_channels, num_centers, 1, 1))
        self.gamma_spline = nn.Parameter(torch.zeros(1, in_channels, num_centers, 1, 1))

    def forward(self, x):
        x_exp = x.unsqueeze(2)
        diff = x_exp - self.centers

        beta = F.softplus(self.raw_beta) + self.eps
        p = 1.2 + 2.3 * torch.sigmoid(self.raw_p)

        gaussian_core = torch.exp(-beta * torch.pow(torch.abs(diff) + self.eps, p))
        spline_mod = 1.0 + self.gamma_linear * diff + self.gamma_spline * (F.relu(diff) ** 2)

        phi = gaussian_core * spline_mod
        B, C, K, H, W = phi.shape
        phi_flat = phi.view(B, C * K, H, W)
        return torch.cat([x, phi_flat], dim=1)


class IntensityRBF(nn.Module):
    """Original learnable Gaussian RBF intensity projection layer."""
    def __init__(self, in_channels, num_centers=8):
        super().__init__()
        self.in_channels = in_channels
        self.num_centers = num_centers
        centers = torch.linspace(0.0, 1.0, num_centers).view(1, 1, 1, 1, num_centers)
        self.register_buffer('centers', centers)
        self.log_bandwidth = nn.Parameter(torch.zeros(1, in_channels, 1, 1, num_centers))

    def forward(self, x):
        B, C, H, W = x.shape
        x_exp = x.unsqueeze(-1)
        beta = F.softplus(self.log_bandwidth) + 1e-4
        rbf = torch.exp(-beta * (x_exp - self.centers) ** 2)
        rbf = rbf.view(B, C * self.num_centers, H, W)
        return torch.cat([x, rbf], dim=1)


class BandPassBlock(nn.Module):
    """
    Dual-Path Band-Pass Residual Block:
    Extracts high-frequency spatial gradients via 7x7 depthwise convolution and absolute modulus:
        Y = ReLU(X + Conv1x1(|DepthwiseConv7x7(X)|))
    """
    def __init__(self, channels=32, kernel_size=7, padding=3):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        dw = self.depthwise(x)
        high = torch.abs(dw)
        proj = self.pointwise(high)
        proj = self.bn(proj)
        out = self.act(x + proj)
        return out


class CurvatureAttention(nn.Module):
    """
    Level-Set / Curvature-Inspired Spatial Attention:
    Derives mean curvature kappa from scalar feature field u = Conv1x1(X):
        kappa = div(nabla u / sqrt(|nabla u|^2 + eps))
    where eps = 1e-2 and normalized Sobel kernels (/ 8.0) are utilized.
    Attention map A_curv = sigmoid(-kappa).
    Residual bypass: X_out = X * A_curv + X.
    """
    def __init__(self, in_channels=32, eps=1e-2):
        super().__init__()
        self.eps = eps
        self.proj = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]], dtype=torch.float32) / 8.0
        ky = torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]], dtype=torch.float32) / 8.0

        self.register_buffer('kx', kx.view(1, 1, 3, 3))
        self.register_buffer('ky', ky.view(1, 1, 3, 3))

    def forward(self, x):
        u = self.proj(x)
        ux = F.conv2d(u, self.kx, padding=1)
        uy = F.conv2d(u, self.ky, padding=1)
        grad_norm = torch.sqrt(ux ** 2 + uy ** 2 + self.eps)
        vx = ux / grad_norm
        vy = uy / grad_norm
        div_x = F.conv2d(vx, self.kx, padding=1)
        div_y = F.conv2d(vy, self.ky, padding=1)
        kappa = div_x + div_y
        a_curv = torch.sigmoid(-kappa)
        return x * a_curv + x


class GlobalTokenMixer(nn.Module):
    """
    Global Token Mixer operating on a 7x7 spatial token grid (49 tokens):
    1. Spatial token mixing MLP (49 -> 49) with LayerNorm and residual connection.
    2. Channel mixing MLP (C -> C) with LayerNorm and residual connection.
    """
    def __init__(self, channels=64, num_tokens=49, token_hidden_dim=49, channel_hidden_dim=128, dropout=0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.channels = channels

        self.ln1 = nn.LayerNorm(channels)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden_dim, num_tokens),
            nn.Dropout(dropout),
        )

        self.ln2 = nn.LayerNorm(channels)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channel_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden_dim, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        norm1 = self.ln1(tokens)
        mixed_tokens = self.token_mlp(norm1.transpose(1, 2)).transpose(1, 2)
        tokens = tokens + mixed_tokens
        norm2 = self.ln2(tokens)
        mixed_channels = self.channel_mlp(norm2)
        tokens = tokens + mixed_channels
        return tokens.transpose(1, 2).view(B, C, H, W)


class SpatialPyramidHead(nn.Module):
    """
    Spatial Pyramid Classification Head:
    Multi-scale pooling via 1x1 and 2x2 adaptive avg pooling:
        Output = [Pool_1x1, Pool_2x2] -> 5C feature vector (320-dim).
    """
    def __init__(self, in_channels=64, num_classes=2, dropout=0.2):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        total_dim = in_channels * (1 + 4)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(total_dim, num_classes),
        )

    def extract_features(self, x):
        p1 = self.pool1(x).flatten(1)
        p2 = self.pool2(x).flatten(1)
        return torch.cat([p1, p2], dim=1)

    def forward(self, x):
        feat = self.extract_features(x)
        return self.classifier(feat)


class PhySSLNet(nn.Module):
    """
    PhySSL-Net: Physics- and Structure-Informed Neural Network (46,318 parameters).
    Complete modular architecture with exact stem, downsampling, and ablation pass-throughs.
    """
    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        rbf_centers=8,
        base_channels=32,
        token_channels=64,
        use_rbf=True,
        rbf_type="adaptive_spline",
        use_bandpass=True,
        use_curvature=True,
        use_token_mixer=True,
        dropout=0.2,
    ):
        super().__init__()
        self.use_rbf = use_rbf
        self.rbf_type = rbf_type
        self.use_bandpass = use_bandpass
        self.use_curvature = use_curvature
        self.use_token_mixer = use_token_mixer

        # 1. RBF Module
        if use_rbf:
            if rbf_type == "intensity_rbf":
                self.rbf = IntensityRBF(in_channels=in_channels, num_centers=rbf_centers)
            else:
                self.rbf = AdaptiveSplineGaussianRBF(in_channels=in_channels, num_centers=rbf_centers)
            stem_in_channels = in_channels * (1 + rbf_centers)
        else:
            self.rbf = nn.Identity()
            stem_in_channels = in_channels

        # 2. Convolutional Stem (28x28 -> 14x14)
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 3. Band-Pass Block
        self.bandpass = BandPassBlock(base_channels, kernel_size=7, padding=3) if use_bandpass else nn.Identity()

        # 4. Curvature Attention
        self.curvature = CurvatureAttention(base_channels, eps=1e-2) if use_curvature else nn.Identity()

        # 5. Transition Stage (14x14 -> 7x7)
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_channels, token_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(token_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 6. Global Token Mixer at 7x7
        self.token_mixer = GlobalTokenMixer(channels=token_channels, num_tokens=49) if use_token_mixer else nn.Identity()

        # 7. Spatial Pyramid Head
        self.head = SpatialPyramidHead(in_channels=token_channels, num_classes=num_classes, dropout=dropout)

    def extract_features(self, x):
        x = self.rbf(x)
        x = self.stem(x)
        x = self.bandpass(x)
        x = self.curvature(x)
        x = self.stage2(x)
        x = self.token_mixer(x)
        return self.head.extract_features(x)

    def forward(self, x):
        feat = self.extract_features(x)
        return self.head.classifier(feat)


class PhySSLSimSiam(nn.Module):
    """
    Self-Supervised Learning Framework (PhySSL-SimSiam):
    Masked Siamese Consistency with Stop-Gradient and Symmetrized Negative Cosine Distance.
    Pretrains the PhySSLNet backbone on unlabelled medical images using dual stochastic masked views.
    """
    def __init__(self, encoder: PhySSLNet, feature_dim=320, proj_dim=128, pred_dim=64):
        super().__init__()
        self.encoder = encoder

        # Projection MLP: maps 320-dim feature representation -> 128-dim latent space
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False),
        )

        # Predictor MLP: predicts representation of opposing branch (128 -> 64 -> 128)
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, proj_dim),
        )

    @staticmethod
    def cosine_loss(p, z):
        z = z.detach() # stop-gradient
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()

    def forward(self, x1, x2):
        feat1 = self.encoder.extract_features(x1)
        feat2 = self.encoder.extract_features(x2)

        z1 = self.projector(feat1)
        z2 = self.projector(feat2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        loss = 0.5 * self.cosine_loss(p1, z2) + 0.5 * self.cosine_loss(p2, z1)
        return loss


# =====================================================================
# BASELINES (100% IDENTICAL TO models/baselines.py)
# =====================================================================

class VanillaCNN(nn.Module):
    """Standard 4-layer Convolutional Neural Network baseline."""
    def __init__(self, in_channels=1, num_classes=2, base_channels=32, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

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
        return self.gap(self.features(x)).flatten(1)

    def forward(self, x):
        return self.classifier(self.extract_features(x))


class VanillaCNNRBF(nn.Module):
    """Vanilla CNN with prepended Gaussian or Adaptive Spline RBF layer."""
    def __init__(self, in_channels=1, num_classes=2, base_channels=32, num_centers=8, dropout=0.2, rbf_type="gaussian"):
        super().__init__()
        if rbf_type == "adaptive_spline":
            self.rbf = AdaptiveSplineGaussianRBF(in_channels=in_channels, num_centers=num_centers)
        else:
            self.rbf = IntensityRBF(in_channels=in_channels, num_centers=num_centers)
        stem_in = in_channels * (1 + num_centers)
        self.cnn = VanillaCNN(in_channels=stem_in, num_classes=num_classes, base_channels=base_channels, dropout=dropout)

    def extract_features(self, x):
        return self.cnn.extract_features(self.rbf(x))

    def forward(self, x):
        return self.cnn(self.rbf(x))


class VGG16Small(nn.Module):
    """VGG16 architecture with Batch Normalization adapted for 28x28 resolution."""
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
        return self.gap(self.features(x)).flatten(1)

    def forward(self, x):
        return self.classifier(self.extract_features(x))


class PreActBottleneck(nn.Module):
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
    """ResNet50V2 adapted for 28x28 medical images."""
    def __init__(self, in_channels=1, num_classes=2, dropout=0.2):
        super().__init__()
        self.in_planes = 32
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
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
        return self.classifier(self.extract_features(x))


class LayerNormChannelsFirst(nn.Module):
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
    """ConvNeXt-Femto baseline (~109k parameters)."""
    def __init__(self, in_channels=1, num_classes=2, depths=[2, 2, 2], dims=[16, 32, 64], dropout=0.2):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1),
            LayerNormChannelsFirst(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)
        for i in range(len(dims) - 1):
            downsample_layer = nn.Sequential(
                LayerNormChannelsFirst(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

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
        return self.classifier(self.extract_features(x))


# =====================================================================
# 2. DATASETS & SSL MASKED TRANSFORMS (100% IDENTICAL TO data/dataset.py)
# =====================================================================

class MaskedPairTransform:
    """Generates two stochastic masked views of the same image for SimSiam SSL pretraining."""
    def __init__(self, img_size=28, patch_size=4, mask_ratio=0.25):
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches_1d = img_size // patch_size
        self.total_patches = self.num_patches_1d * self.num_patches_1d
        self.num_masked = int(round(self.total_patches * mask_ratio))

    def _apply_mask(self, x):
        masked_x = x.clone()
        mask_indices = random.sample(range(self.total_patches), self.num_masked)
        for idx in mask_indices:
            r = (idx // self.num_patches_1d) * self.patch_size
            c = (idx % self.num_patches_1d) * self.patch_size
            masked_x[:, r:r+self.patch_size, c:c+self.patch_size] = 0.0
        return masked_x

    def __call__(self, x):
        x_flip = torch.flip(x, dims=[-1]) if random.random() > 0.5 else x
        return self._apply_mask(x_flip), self._apply_mask(x_flip)


class SSLMedMNISTDataset(Dataset):
    """Unlabelled SSL dataset returning dual stochastic masked views."""
    def __init__(self, images, patch_size=4, mask_ratio=0.25):
        self.images = images
        self.mask_tf = MaskedPairTransform(img_size=28, patch_size=patch_size, mask_ratio=mask_ratio)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.ndim == 2:
            img = img[:, :, None]
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        x = (x - 0.5) / 0.5
        return self.mask_tf(x)


class MedMNISTDataset(Dataset):
    """In-memory Supervised Dataset for MedMNIST."""
    def __init__(self, images, labels, is_multilabel=False, hflip_aug=False):
        self.images = images
        self.labels = labels
        self.is_multilabel = is_multilabel
        self.hflip_aug = hflip_aug

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.ndim == 2:
            img = img[:, :, None]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = (img - 0.5) / 0.5
        if self.hflip_aug and random.random() > 0.5:
            img = torch.flip(img, dims=[-1])
        label = self.labels[idx]
        if self.is_multilabel:
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = torch.tensor(label.item(), dtype=torch.long)
        return img, label


def download_medmnist_file(dataset_name, data_root):
    os.makedirs(data_root, exist_ok=True)
    info = INFO[dataset_name] if MEDMNIST_AVAILABLE else {'url': None}
    flag = dataset_name
    npz_path = os.path.join(data_root, f"{flag}.npz")
    
    if os.path.exists(npz_path) and os.path.getsize(npz_path) > 1024:
        try:
            with open(npz_path, 'rb') as f:
                d = np.load(f)
                if 'train_images' in d:
                    d.close()
                    return npz_path
        except Exception:
            pass

    candidate_urls = [
        info.get('url') if MEDMNIST_AVAILABLE else None,
        f"https://zenodo.org/records/10519652/files/{flag}.npz?download=1",
        f"https://zenodo.org/records/6496656/files/{flag}.npz?download=1",
    ]
    candidate_urls = [u for u in candidate_urls if u]
    headers_list = [{'User-Agent': 'curl/8.4.0'}, {'User-Agent': 'MedMNIST-Downloader/1.0'}, {}]

    print(f"[*] Downloading {flag}.npz to {data_root}...")
    tmp_path = npz_path + ".tmp"
    success = False

    for url in candidate_urls:
        for headers in headers_list:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp, open(tmp_path, "wb") as out_f:
                    while chunk := resp.read(65536):
                        out_f.write(chunk)
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1024:
                    with open(tmp_path, 'rb') as f:
                        d = np.load(f)
                        valid = 'train_images' in d
                        d.close()
                    if valid:
                        if os.path.exists(npz_path):
                            os.remove(npz_path)
                        os.replace(tmp_path, npz_path)
                        success = True
                        break
            except Exception:
                if os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except Exception: pass
            if success: break
        if success: break

    if not success:
        raise RuntimeError(f"[ERROR] Could not download {dataset_name}.npz. Please ensure internet access is enabled.")
    return npz_path


def load_medmnist_dataset(name="pneumoniamnist", batch_size=256, data_root="/kaggle/working/data"):
    data_root = os.path.abspath(data_root)
    os.makedirs(data_root, exist_ok=True)
    npz_path = download_medmnist_file(name, data_root)

    data = np.load(npz_path)
    train_imgs, train_labels = data['train_images'], data['train_labels']
    val_imgs, val_labels = data['val_images'], data['val_labels']
    test_imgs, test_labels = data['test_images'], data['test_labels']

    if MEDMNIST_AVAILABLE:
        info = INFO[name]
        in_channels = info['n_channels']
        num_classes = len(info['label'])
        is_multilabel = (info['task'] == 'multi-label, binary-class')
    else:
        in_channels = 1 if train_imgs.ndim == 3 or train_imgs.shape[-1] == 1 else 3
        num_classes = len(np.unique(train_labels))
        is_multilabel = False

    train_loader = DataLoader(MedMNISTDataset(train_imgs, train_labels, is_multilabel, hflip_aug=True), batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(MedMNISTDataset(val_imgs, val_labels, is_multilabel, hflip_aug=False), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(MedMNISTDataset(test_imgs, test_labels, is_multilabel, hflip_aug=False), batch_size=batch_size, shuffle=False)
    
    # SSL Pretraining loader
    ssl_dataset = SSLMedMNISTDataset(train_imgs, patch_size=4, mask_ratio=0.25)
    ssl_loader = DataLoader(ssl_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'ssl_loader': ssl_loader,
        'train_raw_imgs': train_imgs,
        'in_channels': in_channels,
        'num_classes': num_classes,
        'is_multilabel': is_multilabel,
        'name': name,
        'total_train': len(train_imgs),
        'total_test': len(test_imgs)
    }


# =====================================================================
# 3. METRICS EVALUATOR (100% IDENTICAL TO evaluation/metrics.py)
# =====================================================================

def compute_all_metrics(y_true, y_pred_prob, is_multiclass=False, is_multilabel=False):
    y_true = np.asarray(y_true).squeeze()
    y_pred_prob = np.asarray(y_pred_prob)

    if is_multilabel:
        try:
            auc_score = roc_auc_score(y_true, y_pred_prob, average='macro')
        except Exception:
            auc_score = 0.5
        preds = (y_pred_prob >= 0.5).astype(int)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average='macro', zero_division=0)
        rec = recall_score(y_true, preds, average='macro', zero_division=0)
        prec = precision_score(y_true, preds, average='macro', zero_division=0)
        return {
            'accuracy': float(acc), 'roc_auc': float(auc_score), 'f1': float(f1),
            'recall': float(rec), 'precision': float(prec), 'specificity': 0.0,
            'balanced_acc': float(acc)
        }

    if not is_multiclass:
        scores = y_pred_prob[:, 1] if y_pred_prob.ndim == 2 else y_pred_prob
        y_pred = (scores >= 0.5).astype(int)
        acc = float(accuracy_score(y_true, y_pred))
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc_score = float(auc(fpr, tpr))
        except Exception:
            auc_score = 0.5
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        bal_acc = float(0.5 * (rec + spec))
        return {
            'accuracy': acc, 'roc_auc': auc_score, 'f1': f1, 'recall': rec,
            'precision': prec, 'specificity': spec, 'balanced_acc': bal_acc
        }
    else:
        y_pred = np.argmax(y_pred_prob, axis=1)
        acc = float(accuracy_score(y_true, y_pred))
        try:
            auc_score = float(roc_auc_score(y_true, y_pred_prob, multi_class='ovr', average='macro'))
        except Exception:
            auc_score = 0.5
        f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        rec = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        prec = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        return {
            'accuracy': acc, 'roc_auc': auc_score, 'f1': f1, 'recall': rec,
            'precision': prec, 'specificity': 0.0, 'balanced_acc': acc
        }


# =====================================================================
# 4. 2-PHASE TRAINING PIPELINE (PRETRAIN SSL + SUPERVISED FINE-TUNING)
# =====================================================================

def pretrain_ssl_encoder(dataset_info, ssl_epochs=8, lr=1e-3):
    """
    Phase 1: Self-Supervised Masked Siamese Pretraining.
    Trains PhySSLNet backbone on unlabelled images using dual masked views.
    """
    in_channels = dataset_info['in_channels']
    ssl_loader = dataset_info['ssl_loader']
    
    encoder = PhySSLNet(in_channels=in_channels, num_classes=dataset_info['num_classes']).to(device)
    ssl_model = PhySSLSimSiam(encoder=encoder).to(device)
    
    optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ssl_epochs, eta_min=1e-6)
    
    print(f"\n   >>> [Phase 1: SSL Pretraining] Pretraining PhySSLSimSiam ({ssl_epochs} epochs) on unlabelled data...")
    t0 = time.time()
    for ep in range(1, ssl_epochs + 1):
        ssl_model.train()
        total_loss, n_b = 0.0, 0
        for v1, v2 in ssl_loader:
            v1, v2 = v1.to(device), v2.to(device)
            optimizer.zero_grad()
            loss = ssl_model(v1, v2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ssl_model.parameters(), max_norm=5.0)
            if HAS_TPU: xm.optimizer_step(optimizer)
            else: optimizer.step()
            total_loss += loss.item()
            n_b += 1
        scheduler.step()
        if HAS_TPU: xm.mark_step()
        if ep % max(1, ssl_epochs // 2) == 0 or ep == ssl_epochs:
            print(f"       SSL Epoch [{ep:02d}/{ssl_epochs:02d}] - Negative Cosine Loss: {total_loss / max(1, n_b):.4f}")
    ssl_dur = time.time() - t0
    print(f"   >>> [Phase 1: SSL Pretraining] Finished in {ssl_dur:.1f}s.")
    return encoder.state_dict(), ssl_dur


def build_benchmark_model(model_name, in_channels, num_classes):
    m = model_name.lower()
    if m in ("physsl_net_ssl", "physsl_ssl"):
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, rbf_type="adaptive_spline")
    elif m in ("physsl_net", "physsl_asg_rbf"):
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, rbf_type="adaptive_spline")
    elif m == "physsl_intensity_rbf":
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, rbf_type="intensity_rbf")
    elif m == "no_curvature":
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, rbf_type="intensity_rbf", use_curvature=False)
    elif m == "no_token_mixer":
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, rbf_type="intensity_rbf", use_token_mixer=False)
    elif m == "no_bandpass":
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes, rbf_type="intensity_rbf", use_bandpass=False)
    elif m == "vanilla_cnn_asg_rbf":
        return VanillaCNNRBF(in_channels=in_channels, num_classes=num_classes, rbf_type="adaptive_spline")
    elif m == "vanilla_cnn_rbf":
        return VanillaCNNRBF(in_channels=in_channels, num_classes=num_classes, rbf_type="gaussian")
    elif m == "vanilla_cnn":
        return VanillaCNN(in_channels=in_channels, num_classes=num_classes)
    elif m == "vgg16":
        return VGG16Small(in_channels=in_channels, num_classes=num_classes)
    elif m == "resnet50v2":
        return ResNet50V2Small(in_channels=in_channels, num_classes=num_classes)
    elif m == "convnext_femto":
        return ConvNeXtFemto(in_channels=in_channels, num_classes=num_classes)
    else:
        return PhySSLNet(in_channels=in_channels, num_classes=num_classes)


def train_and_eval(model_name, dataset_info, epochs=15, lr=1e-3, pretrained_state_dict=None):
    in_channels = dataset_info['in_channels']
    num_classes = dataset_info['num_classes']
    is_multilabel = dataset_info['is_multilabel']

    model = build_benchmark_model(model_name, in_channels, num_classes).to(device)

    # Load SSL weights if physsl_net_ssl
    if "ssl" in model_name.lower() and pretrained_state_dict is not None:
        m_dict = model.state_dict()
        filtered = {k: v for k, v in pretrained_state_dict.items() if k in m_dict and v.shape == m_dict[k].shape}
        m_dict.update(filtered)
        model.load_state_dict(m_dict)
        print(f" (Transferred {len(filtered)}/{len(m_dict)} SSL Pretrained Tensors)", end="", flush=True)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = 0.0
    best_weights = None
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        for x, y in dataset_info['train_loader']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if HAS_TPU: xm.optimizer_step(optimizer)
            else: optimizer.step()
        scheduler.step()
        if HAS_TPU: xm.mark_step()

        # Validation Check
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in dataset_info['val_loader']:
                x = x.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy() if is_multilabel else F.softmax(logits, dim=1).cpu().numpy()
                val_preds.append(probs)
                val_targets.append(y.numpy())
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)

        val_m = compute_all_metrics(val_targets, val_preds, is_multiclass=(num_classes > 2), is_multilabel=is_multilabel)
        score = val_m['balanced_acc']
        if score > best_score or best_weights is None:
            best_score = score
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t0

    # Test Evaluation
    model.load_state_dict(best_weights)
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for x, y in dataset_info['test_loader']:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy() if is_multilabel else F.softmax(logits, dim=1).cpu().numpy()
            test_preds.append(probs)
            test_targets.append(y.numpy())
    test_preds = np.concatenate(test_preds, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)

    test_m = compute_all_metrics(test_targets, test_preds, is_multiclass=(num_classes > 2), is_multilabel=is_multilabel)

    return {
        'model': model_name,
        'dataset': dataset_info['name'],
        'parameters': num_params,
        'test_accuracy': round(test_m['accuracy'] * 100, 2),
        'test_roc_auc': round(test_m['roc_auc'], 4),
        'test_f1': round(test_m['f1'], 4),
        'test_recall': round(test_m['recall'] * 100, 2),
        'test_specificity': round(test_m['specificity'] * 100, 2),
        'train_time_sec': round(train_time, 1)
    }


# =====================================================================
# 5. BENCHMARK EXECUTION
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["pneumoniamnist", "dermamnist", "breastmnist"],
                        help="List of 2D MedMNIST datasets to run (or 'all')")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--ssl_epochs", type=int, default=8, help="Number of Self-Supervised pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (256 or 512 for TPU/GPU)")
    parser.add_argument("--output_dir", default="/kaggle/working")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_2d = [
        "pneumoniamnist", "dermamnist", "breastmnist", "bloodmnist",
        "retinamnist", "organcmnist", "organsmnist", "organamnist",
        "pathmnist", "octmnist", "tissuemnist"
    ]

    selected_datasets = all_2d if "all" in args.datasets else args.datasets
    models = ["physsl_net_ssl", "physsl_net", "convnext_femto", "vanilla_cnn", "vgg16", "resnet50v2"]

    records = []
    print(f"\n=================================================================")
    print(f" Starting Benchmark: {len(selected_datasets)} Datasets x {len(models)} Models")
    print(f" Models: {models}")
    print(f"=================================================================\n")

    for d_name in selected_datasets:
        print(f"\n--> Loading Dataset: {d_name.upper()}...")
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
            print(f" Done! Acc: {res['test_accuracy']}% | AUC: {res['test_roc_auc']} | Params: {res['parameters']:,}")
            records.append(res)

    df = pd.DataFrame(records)
    csv_path = os.path.join(args.output_dir, "medmnist_benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Results successfully saved to: {csv_path}")
    print(df.to_string())

    # Plot Comparison
    plt.figure(figsize=(12, 6), dpi=300)
    sns.barplot(data=df, x="dataset", y="test_accuracy", hue="model", palette="Set2")
    plt.title("MedMNIST Multi-Dataset Benchmark: PhySSL-Net (SSL vs Scratch) vs Baselines", fontsize=14, fontweight="bold")
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.xlabel("Dataset", fontsize=12)
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "medmnist_benchmark_plot.png")
    plt.savefig(plot_path)
    print(f"[+] Comparison plot saved to: {plot_path}")
