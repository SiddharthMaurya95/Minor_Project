# PhySSL-Net Research Results Summary & Academic Findings
**DTU B.Tech Research Project (`DTU/CSE/BTP/GP053`)**  
**Project Guide:** Dr. Ravin Ahuja  
**Authors:** Manish Singh, Shivam, Siddharth Kumar Maurya, Saurav Kumar  

---

## 1. Executive Summary
PhySSL-Net is an ultra-compact (46,318 parameter) physics- and structure-informed neural network combining learnable radial basis function intensity encoding, dual-path band-pass residual edge isolation, level-set mean curvature spatial attention, and global token mixing. Pretrained via self-supervised SimSiam masked representation learning, it targets data-efficient medical diagnosis on resource-constrained embedded systems.

## 2. Key Experimental Findings
1. **Competitive Parity at 95% Parameter Reduction:**  
   PhySSL-Net achieves an aggregate 5-seed test accuracy of **87.86% ± 1.90%** (peak **90.87%**), matching heavy baselines such as VGG16 (86.96% ± 2.20%) and outperforming ResNet50V2 (84.23% ± 0.80%), while utilizing **94.8% fewer parameters** (46,318 vs. 894,498).
2. **Clinical Sensitivity Priority:**  
   PhySSL-Net demonstrated a macro recall of **98.46%**, minimizing dangerous false negatives in pneumonia detection.
3. **Statistical Hypothesis Testing:**  
   McNemar's paired chi-squared test confirms that PhySSL-Net achieves statistically significant superior classification over ResNet50V2 ($p = 0.0411 < 0.05$). Against VGG16, it demonstrates non-inferiority ($p = 0.0820$) despite possessing 19.3× fewer weights.
4. **Pareto Dominance:**  
   Multi-objective analysis confirms PhySSL-Net lies on the non-dominated Pareto frontier, delivering top-tier diagnostic accuracy at minimal CPU latency (1.32 ms) and model footprint (0.18 MB).
5. **Architectural Ablation Evidence:**  
   Component removal proves that the Global Token Mixer accounts for a +5.77% accuracy gain, Curvature Attention contributes +1.28%, and IntensityRBF provides +3.69% over the bare backbone.
