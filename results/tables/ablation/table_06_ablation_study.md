### TABLE 6: Architectural Ablation Study

| Ablation Variant | Parameters | Accuracy | ROC-AUC | F1-Score | Precision | Recall | Δ Accuracy | Δ ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w/o BandPass Block | 52,918 | 84.13% | 0.9646 | 0.8863 | 80.25% | 98.97% | -3.37% | +0.0269 |
| w/o Curvature Attn | 46,262 | 86.22% | 0.9461 | 0.9 | 82.34% | 99.23% | -1.28% | +0.0084 |
| w/o IntensityRBF | 43,974 | 83.81% | 0.966 | 0.8848 | 79.67% | 99.49% | -3.69% | +0.0283 |
| w/o Token Mixer | 24,562 | 81.73% | 0.9583 | 0.8719 | 77.60% | 99.49% | -5.77% | +0.0206 |
| PhySSL-Net (Proposed) | 46,294 | 87.50% | 0.9377 | 0.9078 | 84.21% | 98.46% | Baseline | Baseline |