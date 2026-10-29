"""
PhySSL-Net: Physics- and Structure-Informed Self-Supervised Learning Architecture
Complete PyTorch implementation adhering strictly to the DTU approved specification:
- Learnable IntensityRBF Encoding.
- BandPass Residual Block (Depthwise 7x7 Conv + Modulus |Z|).
- Level-Set / Curvature-Inspired Attention (eps = 1e-2).
- Global Token Mixer (7x7 spatial tokens = 49 positions).
- Spatial Pyramid Classification Head (1x1 and 2x2 pooling).
- Masked SimSiam SSL Pretraining Wrapper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class IntensityRBF(nn.Module):
    """
    Learnable Intensity Radial Basis Function (RBF) Encoding.
    Enriches raw pixel intensity x in [0, 1] with K smooth Gaussian kernel responses:
        Phi(x) = [x, exp(-beta_1 (x - c_1)^2), ..., exp(-beta_K (x - c_K)^2)]
    Width is constrained to be strictly positive via beta_k = softplus(a_k) + eps.
    """
    def __init__(self, in_channels=1, num_centers=8, init_range=(0.0, 1.0), eps=1e-4):
        super().__init__()
        self.in_channels = in_channels
        self.num_centers = num_centers
        self.eps = eps

        # Initialize centers uniformly across the specified range
        centers_init = torch.linspace(init_range[0], init_range[1], num_centers)
        self.centers = nn.Parameter(centers_init.view(1, 1, num_centers, 1, 1))

        # Initialize width parameters (a_k initialized such that softplus(a) ~ 10.0)
        # softplus(x) = ln(1 + exp(x))
        init_a = math.log(math.exp(10.0) - 1.0)
        self.a = nn.Parameter(torch.full((1, 1, num_centers, 1, 1), init_a))

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        x_exp = x.unsqueeze(2) # (B, C, 1, H, W)

        beta = F.softplus(self.a) + self.eps # (1, 1, K, 1, 1)
        diff_sq = (x_exp - self.centers) ** 2 # (B, C, K, H, W)
        rbf = torch.exp(-beta * diff_sq) # (B, C, K, H, W)

        # Reshape rbf to (B, C*K, H, W)
        rbf = rbf.view(B, C * self.num_centers, H, W)

        # Concatenate original intensity x with RBF responses
        out = torch.cat([x, rbf], dim=1) # (B, C * (1 + K), H, W)
        return out


class BandPassBlock(nn.Module):
    """
    Multi-Scale Band-Pass / High-Frequency Residual Block.
    Employs a 7x7 depthwise convolution followed by an absolute value modulus nonlinearity:
        Y_high = |Conv7x7_dw(Y_low)|
    Projected back via 1x1 convolution and added to the residual path:
        Y_out = Y_low + BN(Conv1x1(Y_high))
    """
    def __init__(self, channels, kernel_size=7, padding=3):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # High-pass texture extraction via depthwise convolution + modulus
        dw = self.depthwise(x)
        high = torch.abs(dw)
        proj = self.pointwise(high)
        proj = self.bn(proj)
        out = self.act(x + proj)
        return out


class CurvatureAttention(nn.Module):
    """
    Level-Set / Curvature-Inspired Spatial Attention.
    Derives mean curvature kappa from the scalar feature field u = Conv1x1(X):
        kappa = div(nabla u / sqrt(|nabla u|^2 + eps))
    where eps = 1e-2 as mandated by the specification.
    Generates spatial attention map A_curv = sigmoid(-kappa).
    Modulates features via residual attention: X_out = X * A_curv + X.
    """
    def __init__(self, in_channels, eps=1e-2):
        super().__init__()
        self.eps = eps
        # Project channels down to scalar field u
        self.proj = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

        # Fixed 3x3 Sobel kernels for spatial gradients
        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]], dtype=torch.float32) / 8.0
        ky = torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]], dtype=torch.float32) / 8.0

        self.register_buffer('kx', kx.view(1, 1, 3, 3))
        self.register_buffer('ky', ky.view(1, 1, 3, 3))

    def forward(self, x):
        # x: (B, C, H, W)
        u = self.proj(x) # (B, 1, H, W)

        # Gradient nabla u = (u_x, u_y)
        ux = F.conv2d(u, self.kx, padding=1)
        uy = F.conv2d(u, self.ky, padding=1)

        # Gradient magnitude with eps = 1e-2
        grad_norm = torch.sqrt(ux ** 2 + uy ** 2 + self.eps)

        # Normalized gradient vector field
        vx = ux / grad_norm
        vy = uy / grad_norm

        # Divergence: div(V) = d(vx)/dx + d(vy)/dy
        div_x = F.conv2d(vx, self.kx, padding=1)
        div_y = F.conv2d(vy, self.ky, padding=1)
        kappa = div_x + div_y # (B, 1, H, W)

        # Spatial attention map
        a_curv = torch.sigmoid(-kappa)

        # Modulate feature map with residual bypass
        out = x * a_curv + x
        return out


class GlobalTokenMixer(nn.Module):
    """
    Global Token Mixer operating on a 7x7 spatial token grid (49 tokens).
    Implements:
    1. Spatial token mixing MLP (49 -> 49) with LayerNorm and residual connection.
    2. Channel mixing MLP (C -> C) with LayerNorm and residual connection.
    """
    def __init__(self, channels=64, num_tokens=49, token_hidden_dim=49, channel_hidden_dim=128, dropout=0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.channels = channels

        # Token mixing branch
        self.ln1 = nn.LayerNorm(channels)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden_dim, num_tokens),
            nn.Dropout(dropout),
        )

        # Channel mixing branch
        self.ln2 = nn.LayerNorm(channels)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channel_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden_dim, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, C, 7, 7)
        B, C, H, W = x.shape
        assert H * W == self.num_tokens, f"Expected {self.num_tokens} tokens, got {H}x{W}={H*W}"

        # (B, C, H, W) -> (B, N, C) where N = 49
        tokens = x.flatten(2).transpose(1, 2)

        # Token mixing across spatial positions
        norm1 = self.ln1(tokens) # (B, N, C)
        # Transpose to (B, C, N) to mix along tokens
        mixed_tokens = self.token_mlp(norm1.transpose(1, 2)).transpose(1, 2)
        tokens = tokens + mixed_tokens

        # Channel mixing across features
        norm2 = self.ln2(tokens)
        mixed_channels = self.channel_mlp(norm2)
        tokens = tokens + mixed_channels

        # Reshape back to (B, C, 7, 7)
        out = tokens.transpose(1, 2).view(B, C, H, W)
        return out


class SpatialPyramidHead(nn.Module):
    """
    Spatial Pyramid Classification Head.
    Aggregates multi-scale spatial features using 1x1 and 2x2 adaptive average pooling:
        Output representation = [Pool_1x1, Pool_2x2] -> C * (1 + 4) = 5C features.
    Projected to num_classes via Dropout and Linear layer.
    """
    def __init__(self, in_channels=64, num_classes=2, dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        total_dim = in_channels * (1 + 4) # 5 * in_channels

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(total_dim, num_classes),
        )

    def extract_features(self, x):
        # x: (B, C, H, W)
        p1 = self.pool1(x).flatten(1) # (B, C)
        p2 = self.pool2(x).flatten(1) # (B, 4C)
        features = torch.cat([p1, p2], dim=1) # (B, 5C)
        return features

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits


class PhySSLNet(nn.Module):
    """
    Complete PhySSL-Net Backbone & Classifier.
    Supports controlled ablation switches for all proposed modules:
    - use_rbf: bool
    - use_bandpass: bool
    - use_curvature: bool
    - use_token_mixer: bool
    """
    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        rbf_centers=8,
        base_channels=32,
        token_channels=64,
        use_rbf=True,
        use_bandpass=True,
        use_curvature=True,
        use_token_mixer=True,
        dropout=0.2,
    ):
        super().__init__()
        self.use_rbf = use_rbf
        self.use_bandpass = use_bandpass
        self.use_curvature = use_curvature
        self.use_token_mixer = use_token_mixer

        # 1. Intensity RBF
        if use_rbf:
            self.rbf = IntensityRBF(in_channels=in_channels, num_centers=rbf_centers)
            stem_in_channels = in_channels * (1 + rbf_centers)
        else:
            self.rbf = nn.Identity()
            stem_in_channels = in_channels

        # 2. Convolutional Stem (28x28 -> 14x14)
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, base_channels, 14, 14)
        )

        # 3. Band-Pass Residual Block
        if use_bandpass:
            self.bandpass = BandPassBlock(base_channels, kernel_size=7, padding=3)
        else:
            self.bandpass = nn.Sequential(
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
            )

        # 4. Curvature-Inspired Attention
        if use_curvature:
            self.curvature = CurvatureAttention(base_channels, eps=1e-2)
        else:
            self.curvature = nn.Identity()

        # 5. Transition to 7x7 Stage (14x14 -> 7x7)
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_channels, token_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(token_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, token_channels, 7, 7)
        )

        # 6. Global Token Mixer at 7x7
        if use_token_mixer:
            self.token_mixer = GlobalTokenMixer(channels=token_channels, num_tokens=49)
        else:
            self.token_mixer = nn.Identity()

        # 7. Spatial Pyramid Classification Head
        self.head = SpatialPyramidHead(in_channels=token_channels, num_classes=num_classes, dropout=dropout)

    def extract_features(self, x):
        """Returns the pooled 5C feature representation prior to the classification linear layer."""
        x = self.rbf(x)
        x = self.stem(x)
        x = self.bandpass(x)
        x = self.curvature(x)
        x = self.stage2(x)
        x = self.token_mixer(x)
        feat = self.head.extract_features(x)
        return feat

    def forward(self, x, return_features=False):
        x = self.rbf(x)
        x = self.stem(x)
        x = self.bandpass(x)
        x = self.curvature(x)
        x = self.stage2(x)
        x = self.token_mixer(x)

        features = self.head.extract_features(x)
        logits = self.head.classifier(features)

        if return_features:
            return logits, features
        return logits


class PhySSLSimSiam(nn.Module):
    """
    Self-Supervised Learning Wrapper with Masked Siamese (SimSiam) Consistency.
    - Online encoder f_theta extracts representations z1, z2 for two masked views.
    - Projection head g maps representations to latent embeddings.
    - Predictor head h maps embeddings of one branch to predict the other.
    - Symmetrized negative cosine similarity loss with stop-gradient on the target branch.
    """
    def __init__(self, encoder: PhySSLNet, feature_dim=320, proj_dim=128, pred_dim=64):
        super().__init__()
        self.encoder = encoder

        # Projection head: 2-layer MLP (feature_dim -> proj_dim -> proj_dim)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False),
        )

        # Predictor head: 2-layer MLP (proj_dim -> pred_dim -> proj_dim)
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, proj_dim),
        )

    @staticmethod
    def cosine_loss(p, z):
        # Stop gradient on z
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()

    def forward(self, x1, x2):
        # Dual masked views -> feature representation
        feat1 = self.encoder.extract_features(x1)
        feat2 = self.encoder.extract_features(x2)

        # Projection embeddings
        z1 = self.projector(feat1)
        z2 = self.projector(feat2)

        # Predictions
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Symmetrized stop-gradient loss
        loss = 0.5 * self.cosine_loss(p1, z2) + 0.5 * self.cosine_loss(p2, z1)
        return loss
