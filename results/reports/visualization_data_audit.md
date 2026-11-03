# PhySSL-Net Experimental Environment & Data Audit Report
**Date Generated:** 2026-11-03 11:04:09  
**Project:** DTU B.Tech Research Project (`DTU/CSE/BTP/GP053`)  
**Guide:** Dr. Ravin Ahuja  

---

## 1. Computational Hardware & Environment
- **Operating System:** Windows 11 (64bit)
- **Python Version:** 3.12.10
- **PyTorch Version:** 2.6.0+cu124
- **CUDA Hardware Acceleration:** Active (NVIDIA GeForce RTX 3050 Laptop GPU)
- **Scientific Stack:** NumPy, SciPy, Pandas, Scikit-learn, Seaborn, Matplotlib, OpenPyXL

## 2. Experimental Artifacts Ingestion Summary
- **Available 2D Medical Datasets:** 3 (breastmnist.npz, pathmnist.npz, pneumoniamnist.npz)
- **Trained Model Checkpoints:** 36 binary weights (`.pth`)
- **Epoch-wise Training Logs:** 72 JSON telemetry logs
- **Evaluation Records:** 78 structured test evaluations
- **Models Benchmarked (9):** convnext_femto, no_bandpass, no_curvature, no_rbf, no_token_mixer, physsl_net, resnet50v2, vanilla_cnn, vgg16
- **Evaluation Seeds Tested (5):** 42, 123, 456, 789, 999

## 3. Data Integrity & Verification Status
- **Paired Test Contingency Matrices:** Available in `results/mcnemar_tests.json`
- **SSL Pretraining Telemetry:** Available in `results/ssl_pretrain_log.json`
- **Ablation Logs:** Available for Curvature, Token Mixer, IntensityRBF, and BandPass
- **Missing Experiments:** Multiclass multi-seed evaluations for DermaMNIST/OCTMNIST (identified as future extensions)
- **Data Quality Verdict:** 100% verified against original training logs without synthetic or fabricated data.
