"""Inventory of the IU X-Ray dataset: parse reports and images into tables."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = DATA_RAW / "ecgen-radiology"
IMAGES_DIR = DATA_RAW / "images"


def parse_report(path):
    """Parse one IU X-Ray XML report into a flat dict (one row of the reports table)."""
    root = ET.parse(path).getroot()

    # helper: XML label -> text, e.g. {"FINDINGS": "...", "IMPRESSION": "..."}
    sections = {s.get("Label"): s.text for s in root.iter("AbstractText")}

    image_ids = [img.get("id") for img in root.findall("parentImage")]

    mesh = root.find("MeSH")
    mesh_major = [m.text for m in mesh.findall("major")]
    mesh_automatic = [m.text for m in mesh.findall("automatic")]

    return {
        "report_id": root.find("uId").get("id"),
        "comparison": sections.get("COMPARISON"),
        "indication": sections.get("INDICATION"),
        "findings": sections.get("FINDINGS"),
        "impression": sections.get("IMPRESSION"),
        "image_ids": image_ids,
        "n_images": len(image_ids),
        "mesh_major": mesh_major,
        "mesh_automatic": mesh_automatic,
    }


def build_reports_table(reports_dir=REPORTS_DIR):
    """Parse every XML report in reports_dir into a DataFrame (one row per report)."""
    reports = sorted(reports_dir.glob("*.xml"))
    records = []
    for report in reports:
        sections = parse_report(report)
        records.append(sections)

    report_df = pd.DataFrame(records)

    return report_df


def map_images_to_reports(reports_df):
    """Return a dict image_id -> report_id, printing any image claimed by two reports."""
    image_to_report = {}
    for report_id, image_ids in zip(reports_df["report_id"], reports_df["image_ids"]):
        for image_id in image_ids:
            if image_id not in image_to_report:
                image_to_report[image_id] = report_id
            else:
                print("duplicate:", image_id)
    return image_to_report


def build_images_table(reports_df, images_dir=IMAGES_DIR):
    """Read size and mode of every PNG in images_dir into a DataFrame (one row per image)."""
    image_to_report = map_images_to_reports(reports_df)

    image_records = []
    images = sorted(images_dir.glob("*.png"))

    for image_path in images:
        image_id = image_path.stem
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode

        image_records.append(
            {
                "image_id": image_id,  # from path
                "report_id": image_to_report.get(
                    image_id
                ),  # .get: an orphan image becomes None instead of crashing
                "width": width,
                "height": height,
                "mode": mode,
            }
        )
    img_df = pd.DataFrame(image_records)
    # your image_records loop from cell 75 (sorted glob!)
    return img_df


def main():
    """Build both tables and print the dataset inventory."""
    reports_df = build_reports_table()
    images_df = build_images_table(reports_df)

    print("=== Reports ===")
    print(f"reports: {len(reports_df)}")
    print(f"duplicate report IDs: {reports_df['report_id'].duplicated().sum()}")

    print("\nempty sections:")
    print(reports_df[["comparison", "indication", "findings", "impression"]].isna().sum())

    print("\nimages per report:")
    print(reports_df["n_images"].value_counts().sort_index())

    has_findings = reports_df["findings"].notna()
    has_images = reports_df["n_images"] > 0
    usable = reports_df[has_findings & has_images]
    both_problems = (~has_findings & ~has_images).sum()
    print(f"\nusable reports (findings + >=1 image): {len(usable)}")
    print(f"reports with no findings AND no images: {both_problems}")

    print("\n=== Images ===")
    non_png = [p.name for p in IMAGES_DIR.iterdir() if p.suffix != ".png"]
    print(f"PNG files: {len(images_df)}")
    print(f"non-PNG files in images folder: {non_png}")

    print("\nmodes:")
    print(images_df["mode"].value_counts())

    print(f"\nwidth: min {images_df['width'].min()}, max {images_df['width'].max()}")
    print("height:")
    print(images_df["height"].describe())

    n_sizes = len(images_df[["width", "height"]].drop_duplicates())
    portrait = (images_df["height"] > images_df["width"]).sum()
    landscape = (images_df["height"] < images_df["width"]).sum()
    square = (images_df["height"] == images_df["width"]).sum()
    print(f"\ndistinct sizes: {n_sizes}")
    print(f"portrait / landscape / square: {portrait} / {landscape} / {square}")

    referenced = set(map_images_to_reports(reports_df))
    on_disk = set(images_df["image_id"])
    print(f"\norphan images (file but no report): {images_df['report_id'].isna().sum()}")
    print(f"missing files (in a report but no file): {len(referenced - on_disk)}")

    print("\n=== 5 random usable findings (seed 42) ===")
    sample = usable.sample(5, random_state=42)
    for report_id, findings in zip(sample["report_id"], sample["findings"]):
        print(f"[{report_id}] {findings}\n")


if __name__ == "__main__":
    main()
