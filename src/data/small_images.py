"""Squash the X-rays to small grayscale arrays and save one .npz per split."""

import numpy as np
import pandas as pd
from PIL import Image

from src.data.inventory import IMAGES_DIR, PROJECT_ROOT
from src.data.labels import LABEL_MAP_PATH
from src.data.splits import INTERIM_DIR, load_config

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline_numpy.yaml"


def load_small_image(path, size):
    """One image -> grayscale, size x size, pixels scaled to 0..1,
    flattened to shape (size*size,) float32."""
    with Image.open(path) as img:
        small = img.convert("L").resize((size, size))
        # "L" means grayscale, one channel

        img_array = (np.asarray(small) / 255).astype(np.float32).reshape(-1)

    return img_array


def build_split_arrays(split_df, labels_df, categories, size):
    """One example per image, with its report's labels; unlabeled reports dropped.
    Returns X (m, size*size), Y (m, len(categories)) and image_ids (m,),
    row i of all three describing the same image."""
    # each image gets its report's labels
    df = split_df.merge(labels_df, on="report_id")

    # keep labeled reports only
    usable_df = df[df["labeled"] == 1]

    # X: load every image, stack into (m, size*size)
    images_list = []
    for img_id in usable_df["image_id"]:
        img_path = IMAGES_DIR / f"{img_id}.png"
        img_array = load_small_image(img_path, size)
        images_list.append(img_array)
    X = np.stack(images_list)

    # Y: the category columns, in the order of `categories` -> (m, 14)
    Y = usable_df[categories].to_numpy()

    # image_ids: so a row can be traced back to its file
    image_ids = usable_df["image_id"].to_numpy(dtype=str)

    return X, Y, image_ids


def main():
    """Build X, Y, image_ids for train, val and test and save each to
    data/processed/{split}_{size}.npz, with size from configs/baseline_numpy.yaml."""
    cfg = load_config(CONFIG_PATH)
    size = cfg["image_size"]

    # the 14 category names, in rulebook order
    categories = list(load_config(LABEL_MAP_PATH))
    labels_df = pd.read_csv(INTERIM_DIR / "labels.csv")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # each split: build the arrays, save them, print the shapes
    for split in ["train", "val", "test"]:
        df = pd.read_csv(INTERIM_DIR / f"{split}.csv")
        X, Y, image_ids = build_split_arrays(df, labels_df, categories, size)
        path = PROCESSED_DIR / f"{split}_{size}.npz"
        np.savez_compressed(path, X=X, Y=Y, image_ids=image_ids)
        print(split, X.shape, Y.shape)


if __name__ == "__main__":
    main()
