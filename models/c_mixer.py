"""
C-Mixer: Complex Mixer for MedMNIST Classification
Reference Implementation based on:
Zhuoran Zheng and Xiuyi Jia, "Complex Mixer for MedMNIST Classification Decathlon", arXiv:2304.10054 (2023).

Key components:
1. Incentive learning: generates parameterized Gaussian noise as an imaginary channel.
2. Complex affine transformation: Wh = (Aa - Bb) + i(Ba + Ab).
3. CReLU activation: CReLU(a + bi) = ReLU(a) + i ReLU(b).
4. Generalized Pearson projection: p = tanh(real(y) + imag(y)).
5. Spatial and Channel complex token mixing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexLinear(nn.Module):
    """
    Complex Linear layer implementing affine transformation in C:
        W = A + Bi
        h = a + bi
        Wh = (Aa - Bb) + i(Ba + Ab)
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias_real = nn.Parameter(torch.empty(out_features))
            self.bias_imag = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_real, -bound, bound)
        nn.init.uniform_(self.weight_imag, -bound, bound)
        if self.bias_real is not None:
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)

    def forward(self, real, imag):
        # real: (..., in_features), imag: (..., in_features)
        out_real = F.linear(real, self.weight_real) - F.linear(imag, self.weight_imag)
        out_imag = F.linear(real, self.weight_imag) + F.linear(imag, self.weight_real)

        if self.bias_real is not None:
            out_real = out_real + self.bias_real
            out_imag = out_imag + self.bias_imag

        return out_real, out_imag


class ComplexMLP(nn.Module):
    """
    Two-layer MLP in the complex domain with CReLU activation.
    """
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = ComplexLinear(dim, hidden_dim)
        self.fc2 = ComplexLinear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, real, imag):
        r, i = self.fc1(real, imag)
        # CReLU = ReLU(real) + i ReLU(imag)
        r = F.relu(r)
        i = F.relu(i)
        r = self.dropout(r)
        i = self.dropout(i)
        r, i = self.fc2(r, i)
        r = self.dropout(r)
        i = self.dropout(i)
        return r, i


class ComplexMixerBlock(nn.Module):
    """
    Complex Mixer layer:
    1. Token-mixing complex MLP.
    2. Channel-mixing complex MLP.
    """
    def __init__(self, num_tokens=49, hidden_dim=64, token_mlp_dim=49, channel_mlp_dim=128, dropout=0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

        self.ln_token_r = nn.LayerNorm(hidden_dim)
        self.ln_token_i = nn.LayerNorm(hidden_dim)
        self.token_mlp = ComplexMLP(num_tokens, token_mlp_dim, dropout=dropout)

        self.ln_channel_r = nn.LayerNorm(hidden_dim)
        self.ln_channel_i = nn.LayerNorm(hidden_dim)
        self.channel_mlp = ComplexMLP(hidden_dim, channel_mlp_dim, dropout=dropout)

    def forward(self, real, imag):
        # real, imag: (B, num_tokens, hidden_dim)
        # Token mixing
        norm_r = self.ln_token_r(real).transpose(1, 2) # (B, hidden_dim, num_tokens)
        norm_i = self.ln_token_i(imag).transpose(1, 2)
        mixed_r, mixed_i = self.token_mlp(norm_r, norm_i)
        real = real + mixed_r.transpose(1, 2)
        imag = imag + mixed_i.transpose(1, 2)

        # Channel mixing
        norm_r = self.ln_channel_r(real)
        norm_i = self.ln_channel_i(imag)
        mixed_r, mixed_i = self.channel_mlp(norm_r, norm_i)
        real = real + mixed_r
        imag = imag + mixed_i

        return real, imag


class CMixer(nn.Module):
    """
    Complex Mixer architecture for 28x28 medical images.
    Implements incentive learning generator, patch embedding,
    stacked complex mixer blocks, and Pearson curve projection.
    """
    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        img_size=28,
        patch_size=4,
        hidden_dim=64,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_tokens = (img_size // patch_size) ** 2 # 7x7 = 49 tokens
        self.hidden_dim = hidden_dim

        # 1. Incentive learning parameter generator: MLP(Flatten(I)) -> mu, sigma
        flatten_dim = in_channels * img_size * img_size
        self.incentive_net = nn.Sequential(
            nn.Linear(flatten_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2), # mu, log_var
        )

        # 2. Patch embedding projections (linear projection to hidden_dim)
        patch_dim = in_channels * patch_size * patch_size
        self.patch_proj_r = nn.Linear(patch_dim, hidden_dim)
        self.patch_proj_i = nn.Linear(patch_dim, hidden_dim)

        # 3. Stack of Complex Mixer Blocks
        self.layers = nn.ModuleList([
            ComplexMixerBlock(
                num_tokens=self.num_tokens,
                hidden_dim=hidden_dim,
                token_mlp_dim=self.num_tokens,
                channel_mlp_dim=hidden_dim * 2,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # 4. LayerNorm prior to projection
        self.final_ln_r = nn.LayerNorm(hidden_dim)
        self.final_ln_i = nn.LayerNorm(hidden_dim)

        # 5. Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _generate_incentive_noise(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        flat = x.flatten(1)
        params = self.incentive_net(flat) # (B, 2)
        mu = torch.tanh(params[:, 0:1]).view(B, 1, 1, 1)
        log_var = params[:, 1:2].view(B, 1, 1, 1)
        std = torch.exp(0.5 * log_var)

        # Sample Gaussian imaginary matrix x* during training; use deterministic mean during eval
        if self.training:
            eps = torch.randn_like(x)
            x_imag = mu + std * eps
        else:
            x_imag = mu.expand_as(x)
        return x_imag

    def _extract_patches(self, img):
        # img: (B, C, H, W) -> (B, num_tokens, patch_dim)
        B, C, H, W = img.shape
        p = self.patch_size
        # Unfold into patches
        patches = img.unfold(2, p, p).unfold(3, p, p) # (B, C, num_p_h, num_p_w, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous() # (B, num_p_h, num_p_w, C, p, p)
        patches = patches.view(B, self.num_tokens, -1)
        return patches

    def extract_features(self, x):
        # 1. Incentive learning: generate imaginary component
        x_imag = self._generate_incentive_noise(x)

        # 2. Extract real and imaginary patches
        patches_r = self._extract_patches(x)
        patches_i = self._extract_patches(x_imag)

        # Linear projection to complex tokens
        tokens_r = self.patch_proj_r(patches_r)
        tokens_i = self.patch_proj_i(patches_i)

        # 3. Pass through complex mixer layers
        for layer in self.layers:
            tokens_r, tokens_i = layer(tokens_r, tokens_i)

        tokens_r = self.final_ln_r(tokens_r)
        tokens_i = self.final_ln_i(tokens_i)

        # 4. Generalized Pearson projection back to real domain: p = tanh(real(y) + imag(y))
        y_real = tokens_r.mean(dim=1) # (B, hidden_dim)
        y_imag = tokens_i.mean(dim=1)
        features = torch.tanh(y_real + y_imag)
        return features

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits
