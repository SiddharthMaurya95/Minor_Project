"""
Automated Unit Tests for Model Architectures:
- Model output dimensions
- Forward and backward pass gradient flow
- Checkpoint saving and reloading
- Loss finiteness
- Ablation switches functionality
"""

import os
import tempfile
import torch
import torch.nn as nn
import pytest

from models.physsl_net import PhySSLNet, PhySSLSimSiam, IntensityRBF, BandPassBlock, CurvatureAttention, GlobalTokenMixer
from models.c_mixer import CMixer
from models.baselines import VanillaCNN, VGG16Small, ResNet50V2Small


def test_intensity_rbf():
    x = torch.rand(2, 1, 28, 28)
    rbf = IntensityRBF(in_channels=1, num_centers=8)
    out = rbf(x)
    # Output channels should be in_channels * (1 + num_centers) = 1 * 9 = 9
    assert out.shape == (2, 9, 28, 28)
    assert torch.isfinite(out).all()


def test_bandpass_block():
    x = torch.randn(2, 32, 14, 14)
    block = BandPassBlock(channels=32, kernel_size=7, padding=3)
    out = block(x)
    assert out.shape == (2, 32, 14, 14)
    assert torch.isfinite(out).all()


def test_curvature_attention():
    x = torch.randn(2, 32, 14, 14)
    attn = CurvatureAttention(in_channels=32, eps=1e-2)
    out = attn(x)
    assert out.shape == (2, 32, 14, 14)
    assert torch.isfinite(out).all()


def test_global_token_mixer():
    x = torch.randn(2, 64, 7, 7)
    mixer = GlobalTokenMixer(channels=64, num_tokens=49)
    out = mixer(x)
    assert out.shape == (2, 64, 7, 7)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("use_rbf", [True, False])
@pytest.mark.parametrize("use_bandpass", [True, False])
@pytest.mark.parametrize("use_curvature", [True, False])
@pytest.mark.parametrize("use_token_mixer", [True, False])
def test_physsl_net_forward_backward(use_rbf, use_bandpass, use_curvature, use_token_mixer):
    model = PhySSLNet(
        in_channels=1,
        num_classes=2,
        use_rbf=use_rbf,
        use_bandpass=use_bandpass,
        use_curvature=use_curvature,
        use_token_mixer=use_token_mixer,
    )
    x = torch.randn(2, 1, 28, 28, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 2)
    loss = nn.CrossEntropyLoss()(out, torch.tensor([0, 1]))
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_simsiam_ssl_loss():
    encoder = PhySSLNet(in_channels=1, num_classes=2)
    ssl_model = PhySSLSimSiam(encoder=encoder)
    x1 = torch.randn(4, 1, 28, 28)
    x2 = torch.randn(4, 1, 28, 28)
    loss = ssl_model(x1, x2)
    assert torch.isfinite(loss)
    loss.backward()

    # Verify gradients exist in encoder parameters
    has_grad = any(p.grad is not None and torch.any(p.grad != 0) for p in encoder.parameters())
    assert has_grad, "Encoder did not receive gradients during SSL backward pass!"


@pytest.mark.parametrize("model_cls", [CMixer, VanillaCNN, VGG16Small, ResNet50V2Small])
def test_baseline_models(model_cls):
    model = model_cls(in_channels=1, num_classes=2)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    assert out.shape == (2, 2)
    loss = nn.CrossEntropyLoss()(out, torch.tensor([0, 1]))
    assert torch.isfinite(loss)
    loss.backward()


def test_checkpoint_saving_reloading():
    model = PhySSLNet(in_channels=1, num_classes=2)
    model.eval()
    x = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        out_orig = model(x)

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        ckpt_path = f.name

    try:
        torch.save({'state_dict': model.state_dict()}, ckpt_path)
        model_reloaded = PhySSLNet(in_channels=1, num_classes=2)
        model_reloaded.eval()
        ckpt = torch.load(ckpt_path)
        model_reloaded.load_state_dict(ckpt['state_dict'])
        with torch.no_grad():
            out_reloaded = model_reloaded(x)
        assert torch.allclose(out_orig, out_reloaded, atol=1e-6)
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


def test_intermediate_tensor_shapes():
    """Verify exact tensor shapes across all intermediate stages."""
    model = PhySSLNet(in_channels=1, num_classes=2, rbf_centers=8, base_channels=32, token_channels=64)
    model.eval()
    B, C, H, W = 3, 1, 28, 28
    x = torch.rand(B, C, H, W)

    # 1. RBF
    x_rbf = model.rbf(x)
    assert x_rbf.shape == (B, 1 * (1 + 8), 28, 28) # (3, 9, 28, 28)

    # 2. Stem
    x_stem = model.stem(x_rbf)
    assert x_stem.shape == (B, 32, 14, 14)

    # 3. BandPass
    x_bp = model.bandpass(x_stem)
    assert x_bp.shape == (B, 32, 14, 14)

    # 4. Curvature
    x_curv = model.curvature(x_bp)
    assert x_curv.shape == (B, 32, 14, 14)

    # 5. Stage 2 (Transition to 7x7)
    x_stage2 = model.stage2(x_curv)
    assert x_stage2.shape == (B, 64, 7, 7)

    # 6. Global Token Mixer
    x_mix = model.token_mixer(x_stage2)
    assert x_mix.shape == (B, 64, 7, 7)

    # 7. Spatial Pyramid Feature Extraction
    feat = model.head.extract_features(x_mix)
    assert feat.shape == (B, 64 * (1 + 4)) # (3, 320)

    # 8. Classification Logits
    logits = model.head.classifier(feat)
    assert logits.shape == (B, 2)


def test_parameter_registration_and_gradient_flow():
    """Verify that all trainable parameters are registered and receive valid non-zero gradients."""
    model = PhySSLNet(in_channels=1, num_classes=2)
    model.train()
    x = torch.rand(4, 1, 28, 28)
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 0, 1]))
    loss.backward()

    for name, param in model.named_parameters():
        assert param.requires_grad, f"Parameter {name} should have requires_grad=True"
        assert param.grad is not None, f"Parameter {name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"Parameter {name} has NaN/Inf gradients"
        assert not torch.all(param.grad == 0), f"Parameter {name} gradient is completely zero"


def test_numerical_stability_extreme_inputs():
    """Verify numerical stability under zero, constant, and extreme inputs."""
    model = PhySSLNet(in_channels=1, num_classes=2)
    model.eval()

    inputs = [
        torch.zeros(2, 1, 28, 28),
        torch.ones(2, 1, 28, 28),
        torch.full((2, 1, 28, 28), 0.5),
        torch.rand(2, 1, 28, 28) * 100.0, # large values
        torch.rand(2, 1, 28, 28) * 1e-4,  # near-zero values
    ]

    for inp in inputs:
        with torch.no_grad():
            out = model(inp)
        assert torch.isfinite(out).all(), f"Non-finite output for input range [{inp.min()}, {inp.max()}]"


def test_multichannel_and_multiclass_support():
    """Verify PhySSLNet adapts seamlessly to 3 channels and multi-class settings (e.g. DermaMNIST, PathMNIST)."""
    # 3 channels, 7 classes (DermaMNIST)
    model_derma = PhySSLNet(in_channels=3, num_classes=7)
    x_derma = torch.rand(2, 3, 28, 28)
    out_derma = model_derma(x_derma)
    assert out_derma.shape == (2, 7)
    assert torch.isfinite(out_derma).all()

    # 3 channels, 9 classes (PathMNIST)
    model_path = PhySSLNet(in_channels=3, num_classes=9)
    x_path = torch.rand(2, 3, 28, 28)
    out_path = model_path(x_path)
    assert out_path.shape == (2, 9)
    assert torch.isfinite(out_path).all()


def test_exact_parameter_count():
    """Verify exact parameter count for PhySSLNet (46,294)."""
    model = PhySSLNet(in_channels=1, num_classes=2)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params == 46294, f"Expected 46,294 parameters, got {num_params}"


def test_rbf_width_positivity():
    """Verify beta in IntensityRBF is strictly positive via softplus(a) + eps."""
    rbf = IntensityRBF(in_channels=1, num_centers=8, eps=1e-4)
    beta = nn.functional.softplus(rbf.a) + rbf.eps
    assert (beta > 0).all()
    assert (beta >= 1e-4).all()


def test_curvature_attention_constant_input():
    """Verify CurvatureAttention handles flat/constant inputs gracefully without division by zero."""
    attn = CurvatureAttention(in_channels=32, eps=1e-2)
    x = torch.zeros(2, 32, 14, 14)
    out = attn(x)
    assert out.shape == (2, 32, 14, 14)
    assert torch.isfinite(out).all()
    # Gradient test on zero input
    x_grad = torch.zeros(2, 32, 14, 14, requires_grad=True)
    out_grad = attn(x_grad)
    loss = out_grad.sum()
    loss.backward()
    assert torch.isfinite(x_grad.grad).all()


@pytest.mark.parametrize("model_cls,expected_params", [
    (CMixer, 277156),
    (VanillaCNN, 65314),
    (VGG16Small, 894498),
    (ResNet50V2Small, 751522),
])
def test_baseline_exact_parameter_counts(model_cls, expected_params):
    """Verify exact parameter counts for all four baseline architectures."""
    model = model_cls(in_channels=1, num_classes=2)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert params == expected_params, f"{model_cls.__name__} expected {expected_params} params, got {params}"


@pytest.mark.parametrize("model_cls", [CMixer, VanillaCNN, VGG16Small, ResNet50V2Small])
def test_baseline_checkpoint_saving_reloading(model_cls):
    """Verify state_dict checkpoint saving, reloading, and numerical consistency for all baselines."""
    model = model_cls(in_channels=1, num_classes=2)
    model.eval()
    x = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        out_orig = model(x)

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        ckpt_path = f.name

    try:
        torch.save({'state_dict': model.state_dict()}, ckpt_path)
        model_reloaded = model_cls(in_channels=1, num_classes=2)
        model_reloaded.eval()
        ckpt = torch.load(ckpt_path)
        model_reloaded.load_state_dict(ckpt['state_dict'])
        with torch.no_grad():
            out_reloaded = model_reloaded(x)
        assert torch.allclose(out_orig, out_reloaded, atol=1e-5), f"Reloaded checkpoint outputs mismatch in {model_cls.__name__}"
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


@pytest.mark.parametrize("model_cls,expected_dim", [
    (CMixer, 64),
    (VanillaCNN, 64),
    (VGG16Small, 128),
    (ResNet50V2Small, 256),
])
def test_baseline_feature_extraction(model_cls, expected_dim):
    """Verify extract_features produces correct latent representations across all baselines."""
    model = model_cls(in_channels=1, num_classes=2)
    model.eval()
    x = torch.randn(3, 1, 28, 28)
    with torch.no_grad():
        feats = model.extract_features(x)
    assert feats.shape == (3, expected_dim), f"{model_cls.__name__} expected feature shape (3, {expected_dim}), got {feats.shape}"
    assert torch.isfinite(feats).all()


@pytest.mark.parametrize("model_cls", [CMixer, VanillaCNN, VGG16Small, ResNet50V2Small])
def test_baseline_multichannel_and_multiclass(model_cls):
    """Verify baselines support 3 input channels and multiclass settings (DermaMNIST 7 classes, PathMNIST 9 classes)."""
    # DermaMNIST: 3 channels, 7 classes
    model_derma = model_cls(in_channels=3, num_classes=7)
    x_derma = torch.rand(2, 3, 28, 28)
    out_derma = model_derma(x_derma)
    assert out_derma.shape == (2, 7)
    assert torch.isfinite(out_derma).all()

    # PathMNIST: 3 channels, 9 classes
    model_path = model_cls(in_channels=3, num_classes=9)
    x_path = torch.rand(2, 3, 28, 28)
    out_path = model_path(x_path)
    assert out_path.shape == (2, 9)
    assert torch.isfinite(out_path).all()


@pytest.mark.parametrize("model_cls", [CMixer, VanillaCNN, VGG16Small, ResNet50V2Small])
def test_baseline_gradient_flow(model_cls):
    """Verify that every trainable parameter in each baseline receives valid non-zero gradients."""
    model = model_cls(in_channels=1, num_classes=2)
    model.train()
    x = torch.rand(2, 1, 28, 28)
    out = model(x)
    loss = nn.CrossEntropyLoss()(out, torch.tensor([0, 1]))
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} in {model_cls.__name__} received no gradient"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} in {model_cls.__name__} has NaN/Inf gradients"
            assert not torch.all(param.grad == 0), f"Parameter {name} in {model_cls.__name__} has zero gradient"


def test_cmixer_incentive_noise_generation():
    """Verify C-Mixer incentive learning noise generator."""
    model = CMixer(in_channels=1, num_classes=2)
    x = torch.rand(2, 1, 28, 28)
    imag = model._generate_incentive_noise(x)
    assert imag.shape == x.shape
    assert torch.isfinite(imag).all()
    # Gradient flow through incentive network
    loss = imag.sum()
    loss.backward()
    for name, param in model.incentive_net.named_parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()

