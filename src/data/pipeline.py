"""PyTorch input pipeline: Dataset with images cached in memory, DataLoaders for train and eval."""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.inventory import IMAGES_DIR
from src.data.labels import LABEL_MAP_PATH
from src.data.splits import INTERIM_DIR, load_config


class CXRDataset(Dataset):
    """One example per image, with its report's 14 labels; unlabeled reports dropped."""

    def __init__(self, split_df, labels_df, categories, size, transform=None):
        """Load every image of the split once, squashed to size x size grayscale, and keep it
        in memory. transform: applied in __getitem__ (augmentation for train, None for eval).
        Keeps image_ids (m,) str, labels (m, 14) float32, images (m, 1, size, size) uint8,
        row i of all three describing the same image."""
        # each image gets its report's labels; unlabeled reports dropped
        useful_df = labels_df[labels_df["labeled"] == 1]
        merged_df = pd.merge(split_df, useful_df, on="report_id")

        self.image_ids = merged_df["image_id"].to_numpy(dtype=str)
        # float32: the model's predictions are float32, the labels must match
        labels = merged_df[categories].to_numpy(dtype=np.float32)
        self.labels = torch.from_numpy(labels)

        # the cache: uint8 and square, 1 byte per pixel (/255 happens per example)
        images_list = []
        for image_id in self.image_ids:
            image_path = IMAGES_DIR / f"{image_id}.png"
            with Image.open(image_path) as img:
                new_img = img.convert("L").resize((size, size))
                img_array = np.asarray(new_img)
                images_list.append(img_array)
        cache = np.stack(images_list)

        # (m, size, size) -> (m, 1, size, size): channel axis, first in PyTorch
        self.images = torch.from_numpy(cache)
        self.images = self.images.unsqueeze(1)
        self.transform = transform

    def __len__(self):
        """Number of examples, m."""
        return len(self.images)

    def __getitem__(self, i):
        """Example i: image (1, size, size) float32 in 0..1, labels (14,) float32.
        The transform runs here, after the cache, so every call gets a fresh augmentation."""
        image = self.images[i].float() / 255
        if self.transform is not None:
            image = self.transform(image)
        labels = self.labels[i]
        return image, labels


def make_loader(split, size, batch_size, transform=None):
    """DataLoader for one split ("train", "val" or "test"); shuffled only for train."""
    # 1. read data/interim/{split}.csv and data/interim/labels.csv
    split_df = pd.read_csv(INTERIM_DIR / f"{split}.csv")
    labels_df = pd.read_csv(INTERIM_DIR / "labels.csv")
    # 2. categories: the 14 names from label_map.yaml (same line as in small_images.main)
    categories = list(load_config(LABEL_MAP_PATH))
    # 3. dataset = CXRDataset(...)
    dataset = CXRDataset(split_df, labels_df, categories, size, transform=transform)
    # 4. return DataLoader(dataset, batch_size=batch_size, shuffle=...)   shuffle True only for "train"
    shuffle = split == "train"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
