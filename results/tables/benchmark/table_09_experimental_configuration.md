### TABLE 9: Experimental & Hardware Configuration

| Parameter | Value |
| --- | --- |
| Host Accelerator Backend | Google Cloud TPU v5e-8 / NVIDIA RTX 3050 |
| PyTorch Version | 2.6.0+cu124 |
| Random Evaluation Seeds | 42, 123, 456, 789, 999 (5 independent seeds) |
| Optimizer | AdamW (weight decay = 1e-4) |
| Learning Rate Schedule | CosineAnnealingLR (T_max = epochs, eta_min = 1e-6) |
| Batch Size | 256 (TPU/GPU hardware saturation) |
| Gradient Clipping | L2 Norm <= 5.0 |
| SSL Masking Strategy | Stochastic 4x4 Patch Masking (25% ratio) |
| Self-Supervised Objective | Symmetric Negative Cosine Similarity (SimSiam) |
| Supervised Objective | Cross-Entropy / BCE with Logits |