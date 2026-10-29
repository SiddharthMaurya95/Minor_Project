"""
PhySSL-Net: Physics- and Structure-Informed Self-Supervised Learning
Automated Benchmark Runner for MedMNIST v2 on Kaggle (TPU v5e-8 / TPU v3-8 / GPU / CPU)
"""

import os
import sys
import time
import math
import random
import argparse
import urllib.request
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as TF

# Hardware Accelerator Detection: Google Cloud TPU (TPU v5e-8 / v3-8) via PyTorch/XLA, CUDA GPU, or CPU
HAS_TPU = False
xm = None
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    HAS_TPU = True
    print(f"[*] Detected Cloud TPU: {device} (PyTorch/XLA backend active)")
    print(f"[*] Optimized for Google Cloud TPU v5e-8 / TPU v3-8 execution!")
except Exception:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[*] Running on device: {device} ({gpu_name})")

try:
    import medmnist
    from medmnist import INFO
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "medmnist"])
    import medmnist
    from medmnist import INFO

# ==========================================
# 1. MODEL ARCHITECTURES
# ==========================================

class IntensityRBF(nn.Module):
    def __init__(self, in_channels, num_centers=8):
        super().__init__()
        self.in_channels = in_channels
        self.num_centers = num_centers
        centers = torch.linspace(0.0, 1.0, num_centers).view(1, 1, 1, 1, num_centers)
        self.register_buffer('centers', centers)
        self.log_bandwidth = nn.Parameter(torch.zeros(1, in_channels, 1, 1, num_centers))

    def forward(self, x):
        B, C, H, W = x.shape
        x_exp = x.unsqueeze(-1)
        beta = F.softplus(self.log_bandwidth) + 1e-4
        rbf = torch.exp(-beta * (x_exp - self.centers) ** 2)
        rbf = rbf.view(B, C * self.num_centers, H, W)
        return torch.cat([x, rbf], dim=1)

class BandPassBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        high = torch.abs(self.dw_conv(x))
        out = self.bn(self.proj(high))
        return x + out

class CurvatureAttention(nn.Module):
    def __init__(self, channels, eps=1e-2):
        super().__init__()
        self.eps = eps
        self.proj = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, x):
        u = self.proj(x)
        ux = F.conv2d(u, self.kx, padding=1)
        uy = F.conv2d(u, self.ky, padding=1)
        grad_mag = torch.sqrt(ux ** 2 + uy ** 2 + self.eps)
        vx = ux / grad_mag
        vy = uy / grad_mag
        div = F.conv2d(vx, self.kx, padding=1) + F.conv2d(vy, self.ky, padding=1)
        att = torch.sigmoid(-div)
        return x * att + x

class GlobalTokenMixer(nn.Module):
    def __init__(self, channels=64, num_tokens=49):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, num_tokens),
            nn.GELU(),
            nn.Linear(num_tokens, num_tokens)
        )
        self.norm2 = nn.LayerNorm(channels)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        norm_t = self.norm1(tokens)
        tokens = tokens + self.token_mlp(norm_t.transpose(1, 2)).transpose(1, 2)
        tokens = tokens + self.channel_mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).view(B, C, H, W)

class SpatialPyramidHead(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.2):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        total_dim = in_channels * (1 + 4)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_dim, num_classes)
        )

    def forward(self, x):
        f1 = self.pool1(x).flatten(1)
        f2 = self.pool2(x).flatten(1)
        f = torch.cat([f1, f2], dim=1)
        return self.classifier(f)

class PhySSLNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.rbf = IntensityRBF(in_channels=in_channels, num_centers=8)
        stem_in = in_channels * 9
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.bandpass = BandPassBlock(32)
        self.curvature = CurvatureAttention(32)
        self.down = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.mixer = GlobalTokenMixer(channels=64, num_tokens=49)
        self.head = SpatialPyramidHead(in_channels=64, num_classes=num_classes)

    def extract_features(self, x):
        x = self.rbf(x)
        x = self.stem(x)
        x = self.bandpass(x)
        x = self.curvature(x)
        x = self.down(x)
        x = self.mixer(x)
        f1 = self.head.pool1(x).flatten(1)
        f2 = self.head.pool2(x).flatten(1)
        return torch.cat([f1, f2], dim=1)

    def forward(self, x):
        feat = self.extract_features(x)
        return self.head.classifier(feat)

# BASELINES
class VanillaCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.net(x)

class VGG16Small(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)

        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class ResNet50V2Small(nn.Module):
    """
    Standard ResNet50V2 adapted for MedMNIST (751,522 parameters):
    Bottleneck stages [3, 4, 6, 3] with residual projection shortcuts.
    """
    def __init__(self, in_channels=1, num_classes=2, dropout=0.2):
        super().__init__()
        self.in_planes = 32
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 4, stride=2)
        self.layer3 = self._make_layer(64, 6, stride=2)
        self.layer4 = self._make_layer(64, 3, stride=1)

        self.bn = nn.BatchNorm2d(64 * PreActBottleneck.expansion)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(64 * PreActBottleneck.expansion, num_classes),
        )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(PreActBottleneck(self.in_planes, planes, s))
            self.in_planes = planes * PreActBottleneck.expansion
        return nn.Sequential(*layers)

    def extract_features(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        return self.gap(out).flatten(1)

    def forward(self, x):
        feat = self.extract_features(x)
        return self.classifier(feat)

class CMixer(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, hidden_dim=64, num_tokens=49):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=4)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, num_tokens),
            nn.GELU(),
            nn.Linear(num_tokens, num_tokens)
        )
        self.chan_norm = nn.LayerNorm(hidden_dim)
        self.chan_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = x + self.token_mlp(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.chan_mlp(self.chan_norm(x))
        return self.head(x.mean(dim=1))

# ==========================================
# 2. DATA LOADERS & DATASET WRAPPER
# ==========================================

class MedMNISTDataset(Dataset):
    def __init__(self, images, labels, is_multilabel=False):
        self.images = images
        self.labels = labels
        self.is_multilabel = is_multilabel

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.ndim == 2:
            img = img[:, :, None]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = (img - 0.5) / 0.5
        label = self.labels[idx]
        if self.is_multilabel:
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = torch.tensor(label.item(), dtype=torch.long)
        return img, label

def download_medmnist_file(dataset_name, data_root):
    """
    Robust dataset downloader with fallback mirrors and user-agent bypass.
    Prevents HTTP 403 Forbidden and directory-not-found errors on Kaggle.
    """
    os.makedirs(data_root, exist_ok=True)
    info = INFO[dataset_name]
    flag = dataset_name
    npz_path = os.path.join(data_root, f"{flag}.npz")
    
    # Check if already present and valid
    if os.path.exists(npz_path) and os.path.getsize(npz_path) > 1024:
        try:
            with open(npz_path, 'rb') as f:
                d = np.load(f)
                if 'train_images' in d:
                    d.close()
                    return npz_path
        except Exception:
            pass
            
    candidate_urls = [
        info.get('url'),
        f"https://zenodo.org/records/10519652/files/{flag}.npz?download=1",
        f"https://zenodo.org/records/6496656/files/{flag}.npz?download=1",
    ]
    candidate_urls = [u for u in candidate_urls if u]
    
    headers_list = [
        {'User-Agent': 'curl/8.4.0'},
        {'User-Agent': 'MedMNIST-Downloader/1.0'},
        {}
    ]
    
    print(f"[*] Downloading {flag}.npz to {data_root}...")
    tmp_path = npz_path + ".tmp"
    success = False
    
    for url in candidate_urls:
        for headers in headers_list:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp, open(tmp_path, "wb") as out_f:
                    while chunk := resp.read(65536):
                        out_f.write(chunk)
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1024:
                    with open(tmp_path, 'rb') as f:
                        d = np.load(f)
                        valid = 'train_images' in d
                        d.close()
                    if valid:
                        if os.path.exists(npz_path):
                            os.remove(npz_path)
                        os.replace(tmp_path, npz_path)
                        success = True
                        break
            except Exception:
                if os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except Exception: pass
            if success:
                break
        if success:
            break

    # Fallback to system curl if available on Linux/Kaggle
    if not success and os.name != 'nt':
        for url in candidate_urls:
            try:
                subprocess.run(["curl", "-L", "-s", "-o", tmp_path, url], timeout=60, check=True)
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1024:
                    with open(tmp_path, 'rb') as f:
                        d = np.load(f)
                        valid = 'train_images' in d
                        d.close()
                    if valid:
                        if os.path.exists(npz_path):
                            os.remove(npz_path)
                        os.replace(tmp_path, npz_path)
                        success = True
                        break
            except Exception:
                if os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except Exception: pass
            if success:
                break

    if not success:
        raise RuntimeError(
            f"\n"
            f"====================================================================================\n"
            f"[ERROR] Could not download dataset '{dataset_name}.npz' to '{data_root}'.\n"
            f"CRITICAL KAGGLE CONFIGURATION:\n"
            f"1. On Kaggle, open the right-hand sidebar under 'Settings' / 'Notebook options'.\n"
            f"2. Ensure the 'Internet' toggle is switched to 'ON' (requires verified Kaggle account).\n"
            f"3. Alternatively, upload '{dataset_name}.npz' directly to '{data_root}/'.\n"
            f"===================================================================================="
        )
    return npz_path

def load_medmnist_dataset(name="pneumoniamnist", batch_size=128, data_root="/kaggle/working/data"):
    data_root = os.path.abspath(data_root)
    os.makedirs(data_root, exist_ok=True)
    download_medmnist_file(name, data_root)

    info = INFO[name]
    DataClass = getattr(medmnist, info['python_class'])
    is_multilabel = (info['task'] == 'multi-label, binary-class')

    train_raw = DataClass(split='train', root=data_root, download=False)
    val_raw = DataClass(split='val', root=data_root, download=False)
    test_raw = DataClass(split='test', root=data_root, download=False)

    train_loader = DataLoader(MedMNISTDataset(train_raw.imgs, train_raw.labels, is_multilabel), batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(MedMNISTDataset(val_raw.imgs, val_raw.labels, is_multilabel), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(MedMNISTDataset(test_raw.imgs, test_raw.labels, is_multilabel), batch_size=batch_size, shuffle=False)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'in_channels': info['n_channels'],
        'num_classes': len(info['label']) if is_multilabel else len(info['label']),
        'is_multilabel': is_multilabel,
        'task': info['task'],
        'name': name,
        'total_train': len(train_raw),
        'total_test': len(test_raw)
    }

# ==========================================
# 3. TRAINING & EVALUATION FUNCTIONS
# ==========================================

def train_and_eval(model_name, dataset_info, epochs=15, lr=1e-3):
    in_channels = dataset_info['in_channels']
    num_classes = dataset_info['num_classes']
    is_multilabel = dataset_info['is_multilabel']

    if model_name == "physsl_net":
        model = PhySSLNet(in_channels=in_channels, num_classes=num_classes).to(device)
    elif model_name == "c_mixer":
        model = CMixer(in_channels=in_channels, num_classes=num_classes).to(device)
    elif model_name == "vanilla_cnn":
        model = VanillaCNN(in_channels=in_channels, num_classes=num_classes).to(device)
    elif model_name == "vgg16":
        model = VGG16Small(in_channels=in_channels, num_classes=num_classes).to(device)
    elif model_name == "resnet50v2":
        model = ResNet50V2Small(in_channels=in_channels, num_classes=num_classes).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_score = 0.0
    best_weights = None

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in dataset_info['train_loader']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            if HAS_TPU:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()
        scheduler.step()
        if HAS_TPU:
            xm.mark_step()

        # Val check
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in dataset_info['val_loader']:
                x = x.to(device)
                logits = model(x)
                if is_multilabel:
                    probs = torch.sigmoid(logits).cpu().numpy()
                else:
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                val_preds.append(probs)
                val_targets.append(y.numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)

        if is_multilabel:
            val_score = roc_auc_score(val_targets, val_preds, average='macro')
        else:
            val_score = accuracy_score(val_targets, np.argmax(val_preds, axis=1))

        if val_score > best_val_score or best_weights is None:
            best_val_score = val_score
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t0

    # Final Evaluation on Held-Out Test Set
    model.load_state_dict(best_weights)
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for x, y in dataset_info['test_loader']:
            x = x.to(device)
            logits = model(x)
            if is_multilabel:
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = F.softmax(logits, dim=1).cpu().numpy()
            test_preds.append(probs)
            test_targets.append(y.numpy())

    test_preds = np.concatenate(test_preds, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)

    if is_multilabel:
        try:
            from sklearn.metrics import roc_auc_score
            test_auc = roc_auc_score(test_targets, test_preds, average='macro')
        except:
            test_auc = 0.5
        test_acc = accuracy_score(test_targets, (test_preds >= 0.5).astype(int))
        test_f1 = f1_score(test_targets, (test_preds >= 0.5).astype(int), average='macro')
    else:
        pred_labels = np.argmax(test_preds, axis=1)
        test_acc = accuracy_score(test_targets, pred_labels)
        test_f1 = f1_score(test_targets, pred_labels, average='macro')
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(test_targets, test_preds[:, 1])
            test_auc = auc(fpr, tpr)
        else:
            try:
                from sklearn.metrics import roc_auc_score
                test_auc = roc_auc_score(test_targets, test_preds, multi_class='ovr')
            except:
                test_auc = 0.5

    return {
        'model': model_name,
        'dataset': dataset_info['name'],
        'params': num_params,
        'test_accuracy': round(test_acc * 100, 2),
        'test_auc': round(test_auc, 4),
        'test_f1': round(test_f1, 4),
        'train_time_sec': round(train_time, 1)
    }

# ==========================================
# 4. BENCHMARK EXECUTION
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["pneumoniamnist", "dermamnist", "breastmnist"],
                        help="List of 2D MedMNIST datasets to run (or 'all')")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (256 or 512 recommended for TPU v5e-8)")
    parser.add_argument("--output_dir", default="/kaggle/working")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_2d = [
        "pneumoniamnist", "dermamnist", "breastmnist", "bloodmnist",
        "retinamnist", "organcmnist", "organsmnist", "organamnist",
        "pathmnist", "octmnist", "tissuemnist"
    ]

    selected_datasets = all_2d if "all" in args.datasets else args.datasets
    models = ["physsl_net", "c_mixer", "vanilla_cnn", "vgg16", "resnet50v2"]

    records = []
    print(f"\n=================================================================")
    print(f" Starting Benchmark: {len(selected_datasets)} Datasets x {len(models)} Models")
    print(f"=================================================================\n")

    for d_name in selected_datasets:
        print(f"\n--> Loading Dataset: {d_name.upper()}...")
        d_info = load_medmnist_dataset(d_name, batch_size=args.batch_size)

        for m_name in models:
            print(f"  [+] Training & Evaluating {m_name.upper()} on {d_name} ({args.epochs} epochs)...", end="", flush=True)
            res = train_and_eval(m_name, d_info, epochs=args.epochs)
            print(f" Done! Acc: {res['test_accuracy']}% | AUC: {res['test_auc']} | Params: {res['params']:,}")
            records.append(res)

    df = pd.DataFrame(records)
    csv_path = os.path.join(args.output_dir, "medmnist_benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Results successfully saved to: {csv_path}")
    print(df.to_string())

    # Plot Comparison
    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(data=df, x="dataset", y="test_accuracy", hue="model", palette="Set2")
    plt.title("MedMNIST Multi-Dataset Benchmark: PhySSL-Net vs Baselines", fontsize=14, fontweight="bold")
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.xlabel("Dataset", fontsize=12)
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "medmnist_benchmark_plot.png")
    plt.savefig(plot_path)
    print(f"[+] Comparison plot saved to: {plot_path}")
