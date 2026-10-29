"""Turn each report's MeSH terms into 14 CheXpert-style 0/1 labels,
using the rulebook in configs/label_map.yaml."""

import pandas as pd

from src.data.inventory import PROJECT_ROOT, build_reports_table
from src.data.splits import INTERIM_DIR, load_config

LABEL_MAP_PATH = PROJECT_ROOT / "configs" / "label_map.yaml"
UNLABELED_TERM = "No Indexing"  # never annotated: its labels are unknown, not 0


def split_term(text):
    """Split a MeSH term at '/' and clean each part
    (lowercase, single spaces).
    'Mediastinum/paratracheal/enlarged ' ->
    ['mediastinum', 'paratracheal', 'enlarged']"""
    parts = text.split("/")
    clean_parts = []
    for part in parts:
        part = " ".join(part.split()).lower()
        clean_parts.append(part)

    return clean_parts


def term_matches(term, pattern):
    """True if a MeSH term fits a rulebook pattern: same head, and every
    extra word of the pattern appears after the head in the term.
    ('Costophrenic Angle/right/blunted', 'Costophrenic Angle/blunted') -> True"""
    term_parts = split_term(term)
    pattern_parts = split_term(pattern)

    if term_parts[0] != pattern_parts[0]:
        return False

    for word in pattern_parts[1:]:
        if word not in term_parts[1:]:
            return False
    return True


def report_labels(terms, label_map):
    """One report's MeSH terms -> {category: 0 or 1} for every category in the rulebook.
    ['Cardiomegaly/mild', 'Pleural Effusion/right'] -> {'No Finding': 0, ..., 'Cardiomegaly': 1, ...}"""
    labels = {}

    for category, patterns in label_map.items():
        labels[category] = 0
        for term in terms:
            for pattern in patterns:
                if term_matches(term, pattern):
                    labels[category] = 1

    return labels


def build_labels_table(reports_df, label_map):
    """One row per report: report_id, labeled (0 = never annotated),
    then one 0/1 column per category in the rulebook."""
    records = []

    for report_id, terms in zip(reports_df["report_id"], reports_df["mesh_major"]):
        row = {}
        row["report_id"] = report_id
        row["labeled"] = 0 if (UNLABELED_TERM in terms) else 1
        row.update(report_labels(terms, label_map))
        records.append(row)

    return pd.DataFrame(records)


def main():
    """Build the labels table from the rulebook, save it to data/interim/labels.csv,
    and print how many labeled reports have each category."""
    cfg = load_config(path=LABEL_MAP_PATH)

    reports_df = build_reports_table()
    labels_df = build_labels_table(reports_df, cfg)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(INTERIM_DIR / "labels.csv", index=False)

    has_values_df = labels_df[labels_df["labeled"] == 1]
    counts = has_values_df[list(cfg)].sum().sort_values(ascending=False)
    percent = (counts / len(has_values_df) * 100).round(1)
    print(f"labeled reports: {len(has_values_df)} of {len(labels_df)}")
    print(pd.DataFrame({"reports": counts, "percent": percent}))


if __name__ == "__main__":
    main()
