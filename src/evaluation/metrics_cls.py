"""Classification metrics for the 14-category multi-label task."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def confusion_counts(y_true, y_pred):
    """per-category TP, FP, FN, TN; inputs are (N, C)
    arrays of 0/1; returns four arrays of shape (C,)"""
    y_true = np.asarray(y_true)
    # so we can pass a dataframe or a list into a numpy array
    y_pred = np.asarray(y_pred)

    truly_pos = y_true == 1
    pred_pos = y_pred == 1

    tp_mask = truly_pos & pred_pos
    fp_mask = ~truly_pos & pred_pos
    fn_mask = truly_pos & ~pred_pos
    tn_mask = ~truly_pos & ~pred_pos

    tp = tp_mask.sum(axis=0)
    fp = fp_mask.sum(axis=0)
    fn = fn_mask.sum(axis=0)
    tn = tn_mask.sum(axis=0)

    return tp, fp, fn, tn


def safe_divide(num, den, fill):
    """num / den element by element; where den is 0, put fill instead of dividing.
    Works for arrays and for single numbers (the micro row).
    ([2, 0], [3, 0], fill=0.0) -> [0.667, 0.0]"""
    num = np.asarray(num)
    den = np.asarray(den)

    # float, or numpy would cut 0.667 down to 0 when fill is an int
    result = np.full(den.shape, fill, dtype=float)
    ok = den > 0
    result[ok] = num[ok] / den[ok]

    return result


def precision_recall_f1(tp, fp, fn):
    """Per-category precision, recall and F1 from the confusion counts.
    When a denominator is 0:
    - precision -> 0: the model never said yes, a failure
    - recall, F1 -> NaN: no real cases in the set, nothing to grade
    - precision -> NaN too when there are no real cases and no yes answers
    ([2, 1, 0], [1, 0, 0], [1, 1, 1]) -> precision [0.667, 1.0, 0.0],
    recall [0.667, 0.5, 0.0], f1 [0.667, 0.667, 0.0]"""
    precision = safe_divide(tp, tp + fp, fill=0.0)
    recall = safe_divide(tp, tp + fn, fill=np.nan)
    f1 = safe_divide(2 * tp, 2 * tp + fp + fn, fill=np.nan)

    nothing_to_grade = (tp + fp + fn) == 0  # no real cases and no "yes"
    precision[nothing_to_grade] = np.nan

    return precision, recall, f1


def per_class_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    # so we can pass a dataframe or a list into a numpy array
    y_prob = np.asarray(y_prob)

    n_classes = y_true.shape[1]
    auc = np.full(n_classes, np.nan)

    for c in range(n_classes):
        positives = y_true[:, c].sum()
        if 0 < positives < len(y_true):
            auc[c] = roc_auc_score(y_true[:, c], y_prob[:, c])
    return auc


def classification_metrics(y_true, y_prob, threshold, class_names):
    """Run the whole chain and return one row per category, plus a macro row
    (average of the category scores) and a micro row (all cells pooled as one
    category). Columns: positives, tp, fp, fn, tn, precision, recall, f1, auc.
    y_true: (N, C) 0/1 labels; y_prob: (N, C) probabilities;
    threshold: one number, or one per category with shape (C,)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    auc = per_class_auc(y_true, y_prob)  # probabilities: AUC skips the threshold

    table = pd.DataFrame(
        {
            "positives": tp + fn,  # every real case is either caught or missed
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        },
        index=class_names,
    )

    scores = ["precision", "recall", "f1", "auc"]

    # macro: average of the category scores; pandas skips NaN (nothing to grade).
    # Must come before the micro row exists, or the mean would include it.
    table.loc["macro", scores] = table[scores].mean()

    # micro: the whole grid as one big category -> sum the counts, then score once
    micro_tp, micro_fp, micro_fn, micro_tn = tp.sum(), fp.sum(), fn.sum(), tn.sum()
    micro_precision, micro_recall, micro_f1 = precision_recall_f1(
        micro_tp, micro_fp, micro_fn
    )
    micro_auc = per_class_auc(y_true.reshape(-1, 1), y_prob.reshape(-1, 1))[0]
    table.loc["micro"] = [
        micro_tp + micro_fn,
        micro_tp,
        micro_fp,
        micro_fn,
        micro_tn,
        float(micro_precision),
        float(micro_recall),
        float(micro_f1),
        micro_auc,
    ]

    return table
