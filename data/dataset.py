"""
Dataset and Data Loading Pipeline for PhySSL-Net Research.
Supports MedMNIST datasets (PneumoniaMNIST, DermaMNIST, PathMNIST) with:
- Strict train/val/test leakage prevention.
- Unlabelled two-view stochastic masking for self-supervised pretraining.
- Controlled stratified label-fraction subsampling (100%, 50%, 25%, 10%).
- Pure deterministic preprocessing for evaluation.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from torchvision.transforms import functional as TF

try:
    import medmnist
    from medmnist import INFO
    MEDMNIST_AVAILABLE = True
except ImportError:
    MEDMNIST_AVAILABLE = False


class MaskedPairTransform:
    """
    Generates two stochastic masked views of the same input tensor for SimSiam SSL pretraining.
    Divides the image into patch_size x patch_size patches and randomly zeroes out a mask_ratio fraction.
    """
    def __init__(self, img_size=28, patch_size=4, mask_ratio=0.25):
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches_1d = img_size // patch_size
        self.total_patches = self.num_patches_1d * self.num_patches_1d
        self.num_masked = int(self.total_patches * mask_ratio)

    def _apply_mask(self, x):
        # x is (C, H, W)
        masked_x = x.clone()
        mask_indices = random.sample(range(self.total_patches), self.num_masked)
        for idx in mask_indices:
            row = idx // self.num_patches_1d
            col = idx % self.num_patches_1d
            r_start = row * self.patch_size
            r_end = r_start + self.patch_size
            c_start = col * self.patch_size
            c_end = c_start + self.patch_size
            masked_x[:, r_start:r_end, c_start:c_end] = 0.0
        return masked_x

    def __call__(self, x):
        # Apply slight random horizontal flip
        if random.random() > 0.5:
            x_flip = TF.hflip(x)
        else:
            x_flip = x

        view1 = self._apply_mask(x_flip)
        view2 = self._apply_mask(x_flip)
        return view1, view2


class MedMNISTDataset(Dataset):
    """
    In-memory Dataset for MedMNIST npz arrays.
    """
    def __init__(self, images, labels, transform=None, is_ssl=False, ssl_transform=None):
        # images: (N, H, W) or (N, H, W, C)
        # labels: (N, 1) or (N,)
        self.transform = transform
        self.is_ssl = is_ssl
        self.ssl_transform = ssl_transform

        if images.ndim == 3:
            # Grayscale: (N, H, W) -> (N, 1, H, W)
            images = images[:, np.newaxis, :, :]
        elif images.ndim == 4 and images.shape[-1] in (1, 3):
            # (N, H, W, C) -> (N, C, H, W)
            images = np.transpose(images, (0, 3, 1, 2))

        # Convert to float32 in [0, 1]
        if images.dtype == np.uint8:
            self.images = torch.from_numpy(images).float() / 255.0
        else:
            self.images = torch.from_numpy(images).float()

        self.labels = torch.from_numpy(labels).squeeze().long()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        if self.transform is not None:
            x = self.transform(x)

        if self.is_ssl:
            if self.ssl_transform is not None:
                view1, view2 = self.ssl_transform(x)
                return view1, view2
            return x, x

        return x, y


def load_medmnist_raw(dataset_name="pneumoniamnist", data_dir="./data"):
    """
    Loads raw numpy arrays for MedMNIST, automatically downloading if needed.
    """
    os.makedirs(data_dir, exist_ok=True)
    npz_path = os.path.join(data_dir, f"{dataset_name}.npz")

    if not os.path.exists(npz_path):
        if MEDMNIST_AVAILABLE:
            info = INFO[dataset_name]
            DataClass = getattr(medmnist, info['python_class'])
            # Trigger download via medmnist
            _ = DataClass(split='train', root=data_dir, download=True)
        else:
            # Direct download fallback from MedMNIST Zenodo url
            import urllib.request
            url = f"https://zenodo.org/record/6496656/files/{dataset_name}.npz?download=1"
            print(f"Downloading {dataset_name} from {url} to {npz_path}...")
            urllib.request.urlretrieve(url, npz_path)

    data = np.load(npz_path)
    return {
        'train_images': data['train_images'],
        'train_labels': data['train_labels'],
        'val_images': data['val_images'],
        'val_labels': data['val_labels'],
        'test_images': data['test_images'],
        'test_labels': data['test_labels'],
    }


def get_stratified_subset_indices(labels, fraction=1.0, seed=42):
    """
    Deterministically samples a stratified subset of indices based on class proportions.
    """
    if fraction >= 1.0:
        return list(range(len(labels)))

    rng = np.random.RandomState(seed)
    labels = np.array(labels).squeeze()
    unique_classes = np.unique(labels)
    selected_indices = []

    for c in unique_classes:
        c_indices = np.where(labels == c)[0]
        rng.shuffle(c_indices)
        num_to_select = max(1, int(round(len(c_indices) * fraction)))
        selected_indices.extend(c_indices[:num_to_select])

    rng.shuffle(selected_indices)
    return selected_indices


def build_dataloaders(
    dataset_name="pneumoniamnist",
    data_dir="./data",
    batch_size=64,
    label_fraction=1.0,
    seed=42,
    num_workers=0,
    pin_memory=True,
    mask_ratio=0.25,
    patch_size=4,
):
    """
    Constructs train, validation, test, and unlabelled SSL dataloaders.
    Strictly prevents any data leakage from validation or test partitions.
    """
    raw = load_medmnist_raw(dataset_name=dataset_name, data_dir=data_dir)

    # Base transforms: normalization with mean=0.5, std=0.5
    normalize = T.Normalize(mean=[0.5], std=[0.5]) if raw['train_images'].ndim == 3 or raw['train_images'].shape[-1] == 1 else T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # Unlabelled SSL training dataset (two-view masked)
    ssl_transform = MaskedPairTransform(img_size=28, patch_size=patch_size, mask_ratio=mask_ratio)
    ssl_dataset = MedMNISTDataset(
        raw['train_images'],
        raw['train_labels'],
        transform=normalize,
        is_ssl=True,
        ssl_transform=ssl_transform,
    )
    ssl_loader = DataLoader(
        ssl_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    # Supervised labelled training dataset with stratified subsampling
    train_full = MedMNISTDataset(
        raw['train_images'],
        raw['train_labels'],
        transform=normalize,
        is_ssl=False,
    )
    subset_indices = get_stratified_subset_indices(raw['train_labels'], fraction=label_fraction, seed=seed)
    train_subset = Subset(train_full, subset_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # Validation dataset (deterministic)
    val_dataset = MedMNISTDataset(
        raw['val_images'],
        raw['val_labels'],
        transform=normalize,
        is_ssl=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Test dataset (strictly held out)
    test_dataset = MedMNISTDataset(
        raw['test_images'],
        raw['test_labels'],
        transform=normalize,
        is_ssl=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    dataset_info = {
        'name': dataset_name,
        'in_channels': 1 if (raw['train_images'].ndim == 3 or raw['train_images'].shape[-1] == 1) else 3,
        'num_classes': len(np.unique(raw['train_labels'])),
        'total_train': len(raw['train_images']),
        'labelled_train': len(subset_indices),
        'val_samples': len(raw['val_images']),
        'test_samples': len(raw['test_images']),
        'label_fraction': label_fraction,
        'seed': seed,
    }

    return {
        'ssl_loader': ssl_loader,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'info': dataset_info,
    }
