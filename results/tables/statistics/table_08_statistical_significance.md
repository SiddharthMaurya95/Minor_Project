### TABLE 8: Statistical Significance Analysis

| Pairwise Comparison | Both Correct | PhySSL Correct / Opponent Wrong | PhySSL Wrong / Opponent Correct | Both Wrong | McNemar Statistic (χ²) | p-value | Statistically Significant | Academic Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PhySSL-Net vs convnext_femto | 524 | 22 | 19 | 59 | 0.0976 | 7.5478e-01 | No (p ≥ 0.05) | p-value = 7.5478e-01 (No statistically significant difference). |
| PhySSL-Net vs vanilla_cnn | 523 | 23 | 15 | 63 | 1.2895 | 2.5614e-01 | No (p ≥ 0.05) | p-value = 2.5614e-01 (No statistically significant difference). |
| PhySSL-Net vs vgg16 | 520 | 26 | 14 | 64 | 3.025 | 8.1990e-02 | No (p ≥ 0.05) | p-value = 8.1990e-02 (No statistically significant difference). |
| PhySSL-Net vs resnet50v2 | 515 | 31 | 16 | 62 | 4.1702 | 4.1141e-02 | Yes (p < 0.05) | p-value = 4.1141e-02 (Statistically significant). |