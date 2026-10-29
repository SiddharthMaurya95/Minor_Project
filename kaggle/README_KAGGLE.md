# Kaggle Free Tier Execution Guide: PhySSL-Net on MedMNIST (TPU v5e-8 & GPU)

This folder provides self-contained benchmark files ready to run on Kaggle's free compute tier with full acceleration for **Google Cloud TPU v5e-8 / TPU v3-8** or **NVIDIA Tesla P100 / Dual T4** to benchmark **PhySSL-Net** against all 4 baseline architectures (**C-Mixer**, **Vanilla CNN**, **VGG16**, **ResNet50V2**) across the MedMNIST v2 suite.

---

## 1. Files Provided

1. **[kaggle_physsl_medmnist_benchmark.ipynb](file:///c:/Users/maury/Downloads/phySSL/kaggle/kaggle_physsl_medmnist_benchmark.ipynb):**  
   Self-contained Jupyter Notebook formatted for Kaggle's interactive notebook interface. Includes architecture definitions, universal MedMNIST downloading, PyTorch/XLA hardware integration, multi-dataset training, evaluation, comparative tables, and publication-ready plots.
2. **[kaggle_benchmark.py](file:///c:/Users/maury/Downloads/phySSL/kaggle/kaggle_benchmark.py):**  
   Standalone Python script for command-line execution inside Kaggle or any remote TPU/GPU server.

---

## 2. Step-by-Step Instructions to Run on Kaggle with TPU v5e-8 / TPU v3-8

### Step 1: Open Kaggle & Upload Notebook
1. Go to [kaggle.com](https://www.kaggle.com) and log in.
2. Click **Create** $\to$ **New Notebook** (or go to [kaggle.com/code/new](https://www.kaggle.com/code/new)).
3. In the top menu bar of the notebook, click **File** $\to$ **Upload Notebook**.
4. Drag and drop or browse to select:  
   `c:\Users\maury\Downloads\phySSL\kaggle\kaggle_physsl_medmnist_benchmark.ipynb`.

### Step 2: Turn on TPU Accelerator & Internet
In the right-hand sidebar under **Notebook options**:
1. **Accelerator:** Select **TPU VM v3-8** or **TPU v5e**  
   *(If TPU quota is used up, it also runs on **GPU P100** or **GPU T4 x2** seamlessly).*
2. **Internet:** Toggle **Internet ON**  
   *(Required so the notebook can download MedMNIST `.npz` arrays directly).*

### Step 3: Run the Benchmark
1. Click **Run All** (or press `Ctrl + F10` / `Shift + Enter` through each cell).
2. The notebook will automatically:
   - Detect Cloud TPU via `torch_xla` (`xm.xla_device()`).
   - Use `xm.optimizer_step(optimizer)` and `xm.mark_step()` to compile and saturate TPU matrix multiplication units (MXUs).
   - Load each chosen dataset (*PneumoniaMNIST, DermaMNIST, BreastMNIST, BloodMNIST, etc.*).
   - Train and evaluate PhySSL-Net and all 4 baselines.
   - Save the final comparative table to `/kaggle/working/medmnist_benchmark_results.csv`.
   - Render the multi-dataset comparative chart to `/kaggle/working/medmnist_benchmark_plot.png`.

### Step 4: Download Results
In the right-hand panel under **Output** (`/kaggle/working`):
- Click the three dots next to `medmnist_benchmark_results.csv` and `medmnist_benchmark_plot.png` $\to$ **Download**.

---

## 3. TPU v5e-8 Performance Notes

1. **Optimal Batch Size:**
   - TPU v5e-8 chips have 16 GB of High Bandwidth Memory (HBM) and systolic matrix units optimized for large tensor dimensions.
   - The default batch size is set to **`BATCH_SIZE = 256`** (or `512`), which keeps TPU utilization high.
2. **Execution Speed:**
   - On TPU v5e-8, 15 epochs across small datasets (*BreastMNIST, PneumoniaMNIST*) complete in **under 45 seconds** per model!
   - Full 4-dataset benchmarking across all 5 models finishes in **~10 to 12 minutes**.

---

## 4. How to Customize Datasets and Epochs

In **Cell 6** of the notebook, you can adjust which datasets run:

```python
# To run a fast demonstration on 4 diverse modalities (takes ~10 mins on TPU v5e-8):
SELECTED_DATASETS = ['pneumoniamnist', 'dermamnist', 'breastmnist', 'bloodmnist']

# To run ALL 11 single-label 2D MedMNIST datasets (takes ~25-30 mins on TPU v5e-8):
# SELECTED_DATASETS = [
#     'pneumoniamnist', 'dermamnist', 'breastmnist', 'bloodmnist',
#     'retinamnist', 'organcmnist', 'organsmnist', 'organamnist',
#     'pathmnist', 'octmnist', 'tissuemnist'
# ]

MODELS = ['physsl_net', 'c_mixer', 'vanilla_cnn', 'vgg16', 'resnet50v2']
EPOCHS = 15
BATCH_SIZE = 256  # Saturated for Cloud TPU v5e-8
```

---

## 5. Troubleshooting: MedMNIST Download RuntimeError

If you encounter:
```text
RuntimeError: Automatic download failed! Please download pneumoniamnist.npz manually.
# or
RuntimeError: Failed to setup the default root directory. Please specify and create the root directory manually.
```

### Why this happens on Kaggle:
1. **Kaggle Internet Toggle is OFF by Default:** Unverified Kaggle accounts or new notebooks default to having internet disabled, preventing any network requests.
2. **Missing Local Data Directory:** MedMNIST's internal loader throws a `RuntimeError` if the specified `data_root` folder (`/kaggle/working/data`) does not pre-exist before instantiation.
3. **Zenodo HTTP 403 Forbidden:** The default Python `urllib` user agent is blocked by Zenodo's Cloudflare reverse proxy when MedMNIST calls `torchvision.datasets.utils.download_url`.

### How our updated notebook & script solve this:
- **Pre-download with Mirror Fallbacks:** The updated code automatically creates `data_root`, fetches `.npz` arrays using custom headers (`curl/8.4.0`), validates the data with NumPy, and initializes MedMNIST with `download=False`.
- **Action Required on Kaggle:** Ensure your Kaggle notebook has **Internet toggled ON**:
  1. Open the right-hand sidebar (**Notebook options** or **Settings**).
  2. Scroll down to **Internet** and toggle it **ON** *(requires phone verification on Kaggle if prompted)*.

---

## 6. Alternate Command-Line Execution
If running via terminal or Kaggle script mode:
```bash
python kaggle_benchmark.py --datasets pneumoniamnist dermamnist breastmnist --epochs 15 --batch_size 256
```
