"""Patient-level train/val/test split of IU X-Ray
(one report = one patient, see notebooks/02_splits.ipynb)."""

import math

import numpy as np
import yaml

from src.data.inventory import PROJECT_ROOT, build_images_table, build_reports_table

CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def split_report_ids(report_ids, ratios, seed):
    """Shuffle report IDs with a fixed seed and
    cut them into train/val/test."""
    assert math.isclose(sum(ratios.values()), 1), "split ratios must add up to 1"

    rng = np.random.default_rng(seed)
    shuffled_reports = rng.permutation(report_ids)

    n_train = int(ratios["train"] * len(shuffled_reports))
    n_val = int(ratios["val"] * len(shuffled_reports))

    train_ids = shuffled_reports[:n_train]
    val_ids = shuffled_reports[n_train : n_train + n_val]
    test_ids = shuffled_reports[n_train + n_val :]

    return train_ids, val_ids, test_ids


def make_splits(reports_df, images_df, ratios, seed):
    """Split the reports that have images, then
    send each image to its report's split."""
    reports = reports_df.loc[reports_df["n_images"] > 0, "report_id"]

    train_ids, val_ids, test_ids = split_report_ids(reports, ratios, seed)
    train_images = images_df.loc[
        images_df["report_id"].isin(train_ids), ["image_id", "report_id"]
    ]
    # use the inside as filter to select
    # the images that go into the right split. the right portion of the dataframe
    val_images = images_df.loc[
        images_df["report_id"].isin(val_ids), ["image_id", "report_id"]
    ]
    test_images = images_df.loc[
        images_df["report_id"].isin(test_ids), ["image_id", "report_id"]
    ]

    return train_images, val_images, test_images


def load_config(path=CONFIG_PATH):
    """Read a YAML config file into a dict."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

        return cfg


def save_splits(train_df, val_df, test_df, out_dir=INTERIM_DIR):
    """Write the splits to train.csv / val.csv / test.csv in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = {"train": train_df, "val": val_df, "test": test_df}

    for name, df in dfs.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"{name}: {df['report_id'].nunique()} reports, {len(df)} images")


def main():
    """Build the tables, split with the config's ratios and seed, save, print a summary."""
    cfg = load_config()
    reports_df = build_reports_table()
    images_df = build_images_table(reports_df)
    train_df, val_df, test_df = make_splits(
        reports_df, images_df, cfg["split"]["ratios"], cfg["split"]["seed"]
    )
    save_splits(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
