"""
Visualisation Utilities for PhySSL-Net Research:
- Training and Validation Loss Curves.
- Multi-Model ROC Curves.
- Confusion Matrix Heatmaps.
- Pareto Trade-off Curves (Parameter Count vs. Test Accuracy).
- Architectural Component Ablation Plots.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc


def setup_style():
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
    })


def plot_loss_curves(train_losses, val_losses=None, title="Training Loss Curve", save_path="loss_curve.png"):
    """Plots training and validation loss progression over epochs."""
    setup_style()
    plt.figure(figsize=(7, 4.5), dpi=300)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'o-', color='#1f77b4', label='Train Loss', linewidth=2, markersize=4)
    if val_losses and len(val_losses) > 0:
        plt.plot(epochs, val_losses, 's-', color='#ff7f0e', label='Val Loss', linewidth=2, markersize=4)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title, fontweight='bold')
    plt.legend(frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_roc_curves(model_predictions_dict, save_path="roc_curves.png"):
    """
    model_predictions_dict: dict mapping model_name -> (y_true, y_scores)
    """
    setup_style()
    plt.figure(figsize=(7, 6), dpi=300)
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    for idx, (model_name, (y_true, y_scores)) in enumerate(model_predictions_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        c = colors[idx % len(colors)]
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.4f})", color=c, linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontweight='bold')
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path="confusion_matrix.png"):
    """Renders annotated confusion matrix heatmap."""
    setup_style()
    plt.figure(figsize=(6, 5), dpi=300)
    cm = np.asarray(cm)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_pareto(models_efficiency_dict, save_path="pareto_efficiency.png"):
    """
    models_efficiency_dict: dict mapping model_name -> {'params_k': float, 'accuracy': float, 'is_proposed': bool}
    """
    setup_style()
    plt.figure(figsize=(7, 5), dpi=300)

    for name, data in models_efficiency_dict.items():
        is_prop = data.get('is_proposed', False)
        marker = '*' if is_prop else 'o'
        size = 180 if is_prop else 100
        color = '#d62728' if is_prop else '#1f77b4'
        plt.scatter(data['params_k'], data['accuracy'], marker=marker, s=size, color=color, zorder=5)
        plt.annotate(
            f"  {name}",
            (data['params_k'], data['accuracy']),
            fontweight='bold' if is_prop else 'normal',
            fontsize=9,
            va='center',
        )

    plt.xlabel('Trainable Parameters (Thousands)')
    plt.ylabel('Test Accuracy')
    plt.title('Model Efficiency: Accuracy vs Parameter Count', fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_ablation_bar(ablation_dict, metric_name="Accuracy", save_path="ablation_study.png"):
    """
    ablation_dict: dict mapping variant_name -> {'mean': float, 'std': float}
    """
    setup_style()
    plt.figure(figsize=(8, 4.5), dpi=300)
    names = list(ablation_dict.keys())
    means = [ablation_dict[k]['mean'] for k in names]
    stds = [ablation_dict[k]['std'] for k in names]

    colors = ['#2ca02c' if 'Full' in k or 'PhySSL' in k else '#4a90e2' for k in names]
    y_pos = np.arange(len(names))

    plt.barh(y_pos, means, xerr=stds, align='center', color=colors, alpha=0.85, capsize=5)
    plt.yticks(y_pos, names)
    plt.xlabel(metric_name)
    plt.title(f'PhySSL-Net Component Ablation Study ({metric_name})', fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path)
    plt.close()
