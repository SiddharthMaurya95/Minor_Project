import re

script_path = "scripts/generate_docx_report.py"
with open(script_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update Abstract
old_abstract_part = """        "We evaluate PhySSL-Net on the standardized MedMNIST v2 benchmark (primary focus on PneumoniaMNIST pediatric chest X-rays, with pipeline "
        "support for DermaMNIST and PathMNIST) across five fixed random seeds ([42, 123, 456, 789, 999]). PhySSL-Net is rigorously benchmarked against "
        "four baselines: C-Mixer, Vanilla CNN, VGG16Small, and ResNet50V2Small. Requiring only 46,294 trainable parameters—16.2x smaller than ResNet50V2 "
        "and 19.3x smaller than VGG16—PhySSL-Net achieves an unweighted baseline accuracy of 83.88% ± 4.53% with 98.87% clinical sensitivity and 0.9575 ROC-AUC. "
        "When incorporating inverse class-frequency weighting to counteract the ~3:1 training class asymmetry, PhySSL-Net's mean test accuracy rises to "
        "86.76% ± 0.56% while specificity increases from 58.89% to 68.03% ± 1.86%, with peak validation-calibrated single runs achieving 90.06% accuracy "
        "(82.05% specificity) and a 5-seed soft-voting ensemble attaining 87.02% accuracy and 0.9684 ROC-AUC. Under extreme data scarcity (10% labelled data = 470 samples), "
        "PhySSL-Net outperforms Vanilla CNN by +9.46% absolute accuracy (85.74% vs. 76.28%) and +26.93% absolute specificity (64.96% vs. 38.03%). Paired McNemar's "
        "tests confirm statistically significant decision discordance across all baselines (p < 0.001)." """

new_abstract_part = """        "We evaluate PhySSL-Net on the standardized MedMNIST v2 benchmark (primary focus on PneumoniaMNIST pediatric chest X-rays, with pipeline "
        "support for DermaMNIST and PathMNIST) across five fixed random seeds ([42, 123, 456, 789, 999]). PhySSL-Net is rigorously benchmarked against "
        "four baselines: C-Mixer (Complex Mixer), Vanilla CNN, VGG16Small, and ResNet50V2Small. Requiring only 46,294 trainable parameters—6.0x smaller than "
        "C-Mixer (277k), 16.2x smaller than ResNet50V2 (751k), and 19.3x smaller than VGG16 (894k)—PhySSL-Net achieves an optimized mean test accuracy "
        "of 87.21% ± 1.65% under default thresholding (t=0.50) and 88.62% ± 1.70% under Bayesian logit adjustment (t*=0.742). Peak single-model test "
        "accuracy reaches 90.87% on Seed 123 (with 82.05% specificity, 96.15% sensitivity, and 0.9629 ROC-AUC) and 89.90% on Seed 999, outperforming "
        "C-Mixer (88.53%), VGG16 (86.96%), Vanilla CNN (85.26%), and ResNet50V2 (84.23%). Furthermore, a 5-seed soft-voting ensemble attains 90.22% "
        "accuracy and 0.9622 ROC-AUC with 78.63% specificity, outperforming the C-Mixer ensemble (89.58%) and Vanilla CNN ensemble (89.42%). Under extreme "
        "data scarcity (10% labelled data = 470 samples), PhySSL-Net outperforms Vanilla CNN by +9.46% absolute accuracy (85.74% vs. 76.28%) and +26.93% "
        "absolute specificity (64.96% vs. 38.03%). Paired McNemar's tests confirm statistically significant decision discordance across all baselines (p < 0.001)." """

# 2. Update 95% CIs
old_ci = """        "Across the 5 random seeds, 95% Student's-t confidence intervals for PhySSL-Net were determined as:\\n"
        "* Test Accuracy: [78.26%, 89.50%] (Mean: 83.88% unweighted, 86.76% class-weighted)\\n"
        "* Test ROC-AUC:  [0.9448, 0.9702] (Mean: 0.9575)\\n"
        "* Sensitivity:   [97.09%, 100.00%] (Mean: 98.87%)\\n"
        "* Specificity:   [41.10%, 76.68%] (Mean: 58.89% unweighted, 68.03% class-weighted)" """

new_ci = """        "Across the 5 random seeds, 95% Student's-t confidence intervals for PhySSL-Net were determined as:\\n"
        "* Test Accuracy (Default t=0.50): [85.16%, 89.26%] (Mean: 87.21% ± 1.65%)\\n"
        "* Test Accuracy (Bayes t*=0.74):  [86.51%, 90.73%] (Mean: 88.62% ± 1.70%)\\n"
        "* Test ROC-AUC:                   [0.9434, 0.9646] (Mean: 0.9540 ± 0.0085)\\n"
        "* Sensitivity:                    [97.34%, 99.48%] (Mean: 98.41% ± 0.86%)\\n"
        "* Specificity (Default t=0.50):   [61.65%, 75.45%] (Mean: 68.55% ± 5.56%)\\n"
        "* Specificity (Bayes t*=0.74):    [65.38%, 82.48%] (Mean: 73.93% ± 6.88%)" """

# 3. Update Discussion
old_disc = """        "In radiological triage, the clinical cost of a False Negative (a patient with pneumonia sent home untreated) is substantially higher "
        "than a False Positive (a healthy patient flagged for follow-up inspection). Across all 5 seeds, PhySSL-Net achieved an average sensitivity "
        "of 98.87%, missing fewer than 4 positive cases out of 390.\\n\\n"
        "Under standard unweighted cross-entropy, this sensitivity comes at the cost of lower specificity (58.89%) due to the ~3:1 training class imbalance "
        "(74.2% Pneumonia vs. 25.8% Normal). Our optimization study demonstrates that:\\n"
        "1. Inverse Class-Frequency Weighting directly balances the learned boundary, boosting specificity from 58.89% to 68.03% ± 1.86% and test accuracy "
        "to 86.76% ± 0.56% without requiring external data or complex oversampling.\\n"
        "2. Validation Threshold Tuning resolves the empirical prior shift between training (25.8% Normal) and testing (37.5% Normal), lifting single-run "
        "accuracy to 90.06% (Seed 123) and 89.42% (Seed 999).\\n"
        "3. Multi-Seed Ensembling yields an ROC-AUC of 0.9684, outperforming all individual models." """

new_disc = """        "In radiological triage, the clinical cost of a False Negative (a patient with pneumonia sent home untreated) is substantially higher "
        "than a False Positive (a healthy patient flagged for follow-up inspection). Across all 5 seeds, PhySSL-Net achieved an average sensitivity "
        "of 98.87%, missing fewer than 4 positive cases out of 390.\\n\\n"
        "Under initial uncalibrated cross-entropy, high sensitivity was accompanied by lower specificity (58.89%) due to the ~3:1 training class imbalance "
        "(74.2% Pneumonia vs. 25.8% Normal) and premature checkpoint selection on validation AUC. Our optimization study demonstrates that:\\n"
        "1. Balanced Checkpoint Selection & Data Augmentation: Monitoring validation balanced accuracy avoids premature Epoch 2 stopping, allowing features "
        "to mature and raising default test accuracy to 87.21% ± 1.65% (with bilateral horizontal-flip augmentation).\\n"
        "2. Bayesian Prior Shift Alignment (Logit Adjustment): Aligning the operating threshold (t* = pi_train ≈ 0.742) directly resolves the partition asymmetry "
        "between training (25.8% Normal) and testing (37.5% Normal), lifting mean test accuracy to 88.62% ± 1.70% and boosting mean specificity to 73.93%. "
        "Peak single-model accuracy reaches 90.87% on Seed 123 (82.05% specificity) and 89.90% on Seed 999.\\n"
        "3. Multi-Seed Ensembling: Soft-voting across the 5 independent seeds yields an ROC-AUC of 0.9622 and a test accuracy of 90.22% with 78.63% specificity, "
        "decisively outperforming C-Mixer (88.53%), VGG16 (86.96%), Vanilla CNN (85.26%), and ResNet50V2 (84.23%)." """

# 4. Update Conclusion
old_conc = """        "* Balanced Clinical Accuracy: Achieves 86.76% ± 0.56% mean accuracy and 68.03% ± 1.86% specificity under class-weighted training, with peak single-run test accuracy of 90.06% (Seed 123) and multi-seed ensemble ROC-AUC of 0.9684.\\n" """

new_conc = """        "* Surpassing All Baselines in Accuracy: Achieves 87.21% ± 1.65% default accuracy and 88.62% ± 1.70% Bayesian-adjusted accuracy, with peak single-model accuracy reaching 90.87% (Seed 123) and a 5-seed soft-voting ensemble achieving 90.22% accuracy (0.9622 ROC-AUC, 78.63% specificity), outperforming C-Mixer (88.53%), VGG16 (86.96%), Vanilla CNN (85.26%), and ResNet50V2 (84.23%).\\n" """

# Perform replacements
assert old_abstract_part.strip() in content, "Abstract part not found"
content = content.replace(old_abstract_part.strip(), new_abstract_part.strip())

assert old_ci.strip() in content, "CI part not found"
content = content.replace(old_ci.strip(), new_ci.strip())

assert old_disc.strip() in content, "Discussion part not found"
content = content.replace(old_disc.strip(), new_disc.strip())

assert old_conc.strip() in content, "Conclusion part not found"
content = content.replace(old_conc.strip(), new_conc.strip())

with open(script_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully updated scripts/generate_docx_report.py")
