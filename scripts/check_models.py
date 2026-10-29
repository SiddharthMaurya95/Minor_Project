"""
Script to verify model architectures, parameter counts, forward/backward passes, and gradient flow.
Tests:
1. PhySSLNet (Full and all ablation variants)
2. CMixer (Complex Mixer)
3. VanillaCNN
4. VGG16Small
5. ResNet50V2Small
6. PhySSLSimSiam SSL Wrapper
"""

import os
import sys
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.physsl_net import PhySSLNet, PhySSLSimSiam
from models.c_mixer import CMixer
from models.baselines import VanillaCNN, VGG16Small, ResNet50V2Small


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_forward_backward(name, model, x, num_classes=2, is_ssl=False):
    print(f"\n--- Testing {name} ---")
    num_params = count_parameters(model)
    print(f"    Trainable Parameters: {num_params:,}")

    model.train()
    if is_ssl:
        x1 = x
        x2 = x + 0.01 * torch.randn_like(x)
        loss = model(x1, x2)
        print(f"    SSL Loss: {loss.item():.4f}")
        assert torch.isfinite(loss), f"Loss is non-finite in {name}"
        loss.backward()
    else:
        out = model(x)
        assert out.shape == (x.shape[0], num_classes), f"Output shape mismatch: {out.shape}"
        loss = F_loss = nn.CrossEntropyLoss()(out, torch.zeros(x.shape[0], dtype=torch.long, device=x.device))
        assert torch.isfinite(loss), f"Loss is non-finite in {name}"
        loss.backward()

    # Check gradient flow
    zero_grad_params = []
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None or torch.all(param.grad == 0):
                zero_grad_params.append(param_name)

    if zero_grad_params:
        print(f"    [WARN] No/zero gradients in: {zero_grad_params[:5]} (total {len(zero_grad_params)})")
    else:
        print(f"    [OK] All trainable parameters received non-zero gradients.")

    print(f"    [PASSED] Forward and backward pass verified for {name}")
    return num_params


def run_model_checks(device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Running model checks on device: {device.upper()}")
    B, C, H, W = 4, 1, 28, 28
    x = torch.randn(B, C, H, W, device=device)

    # 1. Full PhySSLNet
    physsl = PhySSLNet(in_channels=C, num_classes=2).to(device)
    test_model_forward_backward("PhySSLNet (Full)", physsl, x)

    # 2. PhySSLNet Ablations
    ablations = {
        "PhySSLNet (w/o RBF)": PhySSLNet(in_channels=C, num_classes=2, use_rbf=False),
        "PhySSLNet (w/o BandPass)": PhySSLNet(in_channels=C, num_classes=2, use_bandpass=False),
        "PhySSLNet (w/o Curvature)": PhySSLNet(in_channels=C, num_classes=2, use_curvature=False),
        "PhySSLNet (w/o TokenMixer)": PhySSLNet(in_channels=C, num_classes=2, use_token_mixer=False),
    }
    for name, ablated_model in ablations.items():
        test_model_forward_backward(name, ablated_model.to(device), x)

    # 3. PhySSLSimSiam Wrapper
    ssl_wrapper = PhySSLSimSiam(encoder=PhySSLNet(in_channels=C, num_classes=2)).to(device)
    test_model_forward_backward("PhySSLSimSiam SSL", ssl_wrapper, x, is_ssl=True)

    # 4. Baselines
    c_mixer = CMixer(in_channels=C, num_classes=2).to(device)
    test_model_forward_backward("C-Mixer Baseline", c_mixer, x)

    vanilla_cnn = VanillaCNN(in_channels=C, num_classes=2).to(device)
    test_model_forward_backward("Vanilla CNN Baseline", vanilla_cnn, x)

    vgg16 = VGG16Small(in_channels=C, num_classes=2).to(device)
    test_model_forward_backward("VGG16Small Baseline", vgg16, x)

    resnet50 = ResNet50V2Small(in_channels=C, num_classes=2).to(device)
    test_model_forward_backward("ResNet50V2Small Baseline", resnet50, x)

    print("\n=======================================================")
    print("[PASSED] ALL MODEL ARCHITECTURES PASSED INTEGRITY TESTS")
    print("=======================================================")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_model_checks(device=device)
