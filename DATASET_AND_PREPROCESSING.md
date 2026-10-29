# Dataset and Preprocessing Specification (DATASET_AND_PREPROCESSING.md)

## 1. Primary Benchmark: MedMNIST v2 PneumoniaMNIST
- **Modality:** Pediatric Chest X-ray radiographs.
- **Task:** Binary classification (Class 0: Normal, Class 1: Pneumonia).
- **Resolution:** $1 \times 28 \times 28$ pixels (grayscale).
- **Partitions:**
  - **Train:** 4,708 samples (80.4%)
  - **Validation:** 524 samples (8.9%)
  - **Test:** 624 samples (10.7%)
  - **Total:** 5,856 samples
- **Class Balance:**
  - Train: ~74.2% Pneumonia (3,494), ~25.8% Normal (1,214).
  - Validation: ~74.2% Pneumonia (389), ~25.8% Normal (135).
  - Test: ~62.5% Pneumonia (390), ~37.5% Normal (234).
  - **Handling Imbalance:** In addition to threshold-0.5 Accuracy, ROC-AUC, Macro-F1, Sensitivity (Recall), and Specificity are evaluated to avoid misleading claims driven by majority-class dominance.

## 2. Extended Datasets Supported
### DermaMNIST
- Modality: Dermatoscopic images of pigmented skin lesions (HAM10000).
- Task: Multi-class (7 classes: actinic keratoses, basal cell carcinoma, benign keratosis, dermatofibroma, melanoma, melanocytic nevi, vascular lesions).
- Resolution: $3 \times 28 \times 28$ pixels (RGB).
- Partitions: 7,007 train, 1,003 validation, 2,005 test (10,015 total).

### PathMNIST
- Modality: Colorectal cancer histology patches (NCT-CRC-HE-100K / CRC-VAL-HE-7K).
- Task: Multi-class (9 classes: adipose, background, debris, lymphocytes, mucus, smooth muscle, normal colon mucosa, cancer-associated stroma, colorectal adenocarcinoma epithelium).
- Resolution: $3 \times 28 \times 28$ pixels (RGB).
- Partitions: 89,996 train, 10,004 validation, 7,180 test (107,180 total).

---

## 3. Data Pipeline and Leakage Prevention
1. **Strict Train/Val/Test Separation:**
   - The test set is held out completely and only accessed during the final evaluation phase.
   - Model selection, early stopping, and hyperparameter decisions use only the validation partition.
2. **Preprocessing Pipeline:**
   - Raw pixel arrays $[0, 255]$ are converted to PyTorch tensors in $[0.0, 1.0]$.
   - Normalized via $\frac{x - \mu}{\sigma}$ using training set statistics only ($\mu=0.5, \sigma=0.5$ default).
   - Test-time preprocessing is strictly deterministic.
3. **Label-Fraction Generation:**
   - Controlled subsets (100%, 50%, 25%, 10%) are generated using deterministic stratified subsampling with a fixed seed.
   - The exact same labelled subsets are shared across PhySSL-Net, C-Mixer, VGG16, ResNet50V2, and Vanilla CNN to ensure rigorous fairness.
4. **Self-Supervised Unlabelled Preparation:**
   - 100% of the training split is utilized for SSL pretraining with labels completely detached and inaccessible to the encoder.

---

## 4. Self-Supervised Masking Transformations
For each unlabelled training sample $x$, two stochastic views $(x_1, x_2)$ are generated:
- **Patch Masking:** The $28 \times 28$ image is divided into $4 \times 4$ blocks ($7 \times 7 = 49$ spatial tokens).
- A uniform random subset of patches ($25\%$) is masked (zeroed out) independently for each view.
- **Slight Medical Augmentation:** Random horizontal flipping ($p=0.5$) and subtle rotation ($\pm 7^\circ$), preserving diagnostic anatomic structure.
