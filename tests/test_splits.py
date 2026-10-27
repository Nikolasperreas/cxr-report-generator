"""Tests for src/data/splits.py, run with: python -m pytest"""

import pandas as pd

from src.data.splits import make_splits

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def make_fake_tables():
    """10 small fake reports (one without images) and their images."""
    n_images = {"CXR1": 2, "CXR2": 2, "CXR3": 1, "CXR4": 3, "CXR5": 2,
                "CXR6": 2, "CXR7": 1, "CXR8": 2, "CXR9": 2, "CXR10": 0}
    reports_df = pd.DataFrame({"report_id": list(n_images), "n_images": list(n_images.values())})
    image_rows = []
    for report_id, n in n_images.items():
        for k in range(n):
            image_rows.append({"image_id": f"{report_id}_img{k}", "report_id": report_id})
    images_df = pd.DataFrame(image_rows)
    return reports_df, images_df


def test_no_report_in_two_sets():
    """A report (= patient) must never appear in two sets."""
    reports_df, images_df = make_fake_tables()
    train, val, test = make_splits(reports_df, images_df, RATIOS, seed=42)
    tr, va, te = set(train["report_id"]), set(val["report_id"]), set(test["report_id"])
    assert tr & va == set()
    assert tr & te == set()
    assert va & te == set()


def test_every_image_used_once():
    """The three image counts add up to the total: no image lost or duplicated."""
    reports_df, images_df = make_fake_tables()
    train, val, test = make_splits(reports_df, images_df, RATIOS, seed=42)
    assert len(train) + len(val) + len(test) == len(images_df)


def test_same_seed_same_split():
    """Two calls with the same seed give exactly the same split."""
    reports_df, images_df = make_fake_tables()
    first = make_splits(reports_df, images_df, RATIOS, seed=42)
    second = make_splits(reports_df, images_df, RATIOS, seed=42)
    for a, b in zip(first, second):
        assert a.equals(b)
