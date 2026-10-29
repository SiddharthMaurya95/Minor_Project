"""
PhySSL-Net Grad-CAM Interpretability & Visual Explanations
Generates class-activation maps for representative test cases (Normal vs. Pneumonia)
using verified model checkpoints from the PneumoniaMNIST test set.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.physsl_net import PhySSLNet
from data.dataset import build_dataloaders


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        score = output[0, target_class]
        score.backward()
        
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True) # (1, C, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # (1, 1, H, W)
        cam = F.relu(cam)
        
        # Resize to input dimensions (28, 28)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        probs = F.softmax(output, dim=1).detach().squeeze().cpu().numpy()
        return cam, target_class, probs


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for PhySSL-Net")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/physsl_net_pneumoniamnist_frac100_seed789_best.pth",
                        help="Path to trained PhySSLNet checkpoint")
    parser.add_argument("--output_path", type=str, default="results/gradcam_explanations.png",
                        help="Path to save output visualization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Grad-CAM generation on {device} using checkpoint: {args.checkpoint}")

    # Build model
    model = PhySSLNet(in_channels=1, num_classes=2).to(device)
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print("Checkpoint successfully loaded.")
    else:
        # Fallback to seed 42 checkpoint if seed 789 is missing
        fallback = "checkpoints/physsl_net_pneumoniamnist_frac100_seed42_best.pth"
        print(f"Target checkpoint not found. Falling back to: {fallback}")
        checkpoint = torch.load(fallback, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

    # Hook target layer (stage2: transition conv before token mixer)
    target_layer = model.stage2

    grad_cam = GradCAM(model, target_layer)

    # Load test split
    loaders = build_dataloaders(
        dataset_name="pneumoniamnist",
        batch_size=1,
        seed=42,
    )
    test_loader = loaders['test_loader']

    class_names = ["Normal (Healthy)", "Pneumonia (Infected)"]

    # Select representative samples:
    # 1. True Negative (Normal predicted Normal)
    # 2. True Positive (Pneumonia predicted Pneumonia)
    # 3. Severe Pneumonia (High confidence Pneumonia)
    # 4. Borderline / Challenging sample
    selected_cases = []
    
    for idx, (img, target) in enumerate(test_loader):
        img_dev = img.to(device)
        target_val = int(target.item()) if target.ndim == 1 else int(target.squeeze().item())
        
        with torch.no_grad():
            logits = model(img_dev)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_val = int(np.argmax(probs))

        # Check conditions
        if len(selected_cases) == 0 and target_val == 0 and pred_val == 0 and probs[0] > 0.85:
            # High confidence Normal
            selected_cases.append((img, target_val, "True Normal (TN)", idx))
        elif len(selected_cases) == 1 and target_val == 1 and pred_val == 1 and probs[1] > 0.95:
            # High confidence Pneumonia
            selected_cases.append((img, target_val, "Typical Pneumonia (TP)", idx))
        elif len(selected_cases) == 2 and target_val == 1 and pred_val == 1 and 0.80 < probs[1] < 0.92:
            # Moderate Pneumonia
            selected_cases.append((img, target_val, "Focal Pneumonia (TP)", idx))
        elif len(selected_cases) == 3 and target_val == 0 and pred_val == 0 and 0.60 < probs[0] < 0.80:
            # Challenging Normal
            selected_cases.append((img, target_val, "Borderline Normal (TN)", idx))

        if len(selected_cases) == 4:
            break

    # If any slot is empty, fill with available items
    if len(selected_cases) < 4:
        for idx, (img, target) in enumerate(test_loader):
            target_val = int(target.item()) if target.ndim == 1 else int(target.squeeze().item())
            if len(selected_cases) < 4:
                selected_cases.append((img, target_val, f"Test Case #{idx}", idx))

    # Generate heatmaps and plot
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=300)

    for col, (img, true_label, case_desc, orig_idx) in enumerate(selected_cases):
        img_dev = img.to(device)
        cam, pred_class, probs = grad_cam.generate_cam(img_dev, target_class=None)
        
        # Raw image in [0, 1] for visualization
        raw_img = img.squeeze().cpu().numpy()
        # Undo standard normalization [-1, 1] -> [0, 1]
        disp_img = (raw_img * 0.5) + 0.5
        disp_img = np.clip(disp_img, 0.0, 1.0)

        # Plot 1: Original Image
        axes[0, col].imshow(disp_img, cmap='gray')
        axes[0, col].set_title(f"{case_desc}\nTrue: {class_names[true_label]}", fontsize=10, fontweight='bold')
        axes[0, col].axis('off')

        # Plot 2: Grad-CAM Overlay
        axes[1, col].imshow(disp_img, cmap='gray')
        im = axes[1, col].imshow(cam, cmap='jet', alpha=0.55)
        conf_str = f"P(Normal)={probs[0]:.2f}\nP(Pneumo)={probs[1]:.2f}"
        pred_label_str = class_names[pred_class]
        is_correct = (pred_class == true_label)
        status_color = "green" if is_correct else "red"
        
        axes[1, col].set_title(f"Pred: {pred_label_str}\n{conf_str}", fontsize=9, color=status_color, fontweight='bold')
        axes[1, col].axis('off')

    plt.suptitle("PhySSL-Net Visual Explanations via Grad-CAM (PneumoniaMNIST Held-Out Test Set)\n"
                 "Top: Input Chest Radiographs | Bottom: Grad-CAM Saliency Overlays on Stage-2 Feature Maps",
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path, bbox_inches='tight')
    plt.close()
    print(f"Grad-CAM explanation maps successfully saved to: {args.output_path}")


if __name__ == "__main__":
    main()
