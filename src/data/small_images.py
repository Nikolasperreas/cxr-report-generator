import numpy as np
import pandas as pd
from PIL import Image

from src.data.inventory import IMAGES_DIR, PROJECT_ROOT
from src.data.labels import LABEL_MAP_PATH
from src.data.splits import INTERIM_DIR, load_config

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline_numpy.yaml"


def load_small_image(path, size):

    with Image.open(path) as img:
        small = img.convert("L").resize((size, size))
        # "L" means grayscale, one channel

        img_array = (np.asarray(small) / 255).astype(np.float32).reshape(-1)

    return img_array


def build_split_arrays(split_df, labels_df, categories, size):
    # 1. each image gets its report's labels
    df = split_df.merge(labels_df, on="report_id")
    usable_df = df[df["labeled"] == 1]

    images_list = []
    for img_id in usable_df["image_id"]:
        img_path = IMAGES_DIR / f"{img_id}.png"
        img_array = load_small_image(img_path, size)
        images_list.append(img_array)
    # 2. keep labeled reports only
    # 3. X: load every image, stack into (m, size*size)
    X = np.stack(images_list)

    # 4. Y: the category columns, in the order of `categories` -> (m, 14)
    Y = usable_df[categories].to_numpy()

    image_ids = usable_df["image_id"].to_numpy(dtype=str)
    # 5. image_ids: so a row can be traced back to its file
    return X, Y, image_ids


def main():
    # 1. read image_size from the config
    cfg = load_config(CONFIG_PATH)
    size = cfg["image_size"]

    # 2. read labels.csv and the 14 category names
    categories = list(load_config(LABEL_MAP_PATH))
    labels_df = pd.read_csv(INTERIM_DIR / "labels.csv")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # 3. for each split ("train", "val", "test"):
    for split in ["train", "val", "test"]:
        df = pd.read_csv(INTERIM_DIR / f"{split}.csv")
        X, Y, image_ids = build_split_arrays(df, labels_df, categories, size)
        path = PROCESSED_DIR / f"{split}_{size}.npz"
        np.savez_compressed(path, X=X, Y=Y, image_ids=image_ids)
        #       read its csv, build the arrays, save them
        print(split, X.shape, Y.shape)
    # 4. print the shapes so you can see it worked

    return


if __name__ == "__main__":
    main()
