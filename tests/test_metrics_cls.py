"""Tests for src/evaluation/metrics_cls.py, run with: python -m pytest"""

import numpy as np
import pytest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from src.evaluation.metrics_cls import classification_metrics, confusion_counts

# any warning (0/0 division, AUC with one class) makes a test fail
pytestmark = pytest.mark.filterwarnings("error")

# the toy from notebooks/04_metrics.ipynb: 6 reports x 3 categories
NAMES = ["No Finding", "Cardiomegaly", "Pneumothorax"]
Y_TRUE = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 0], [1, 0, 0]])
Y_PROB = np.array([[0.9, 0.1, 0.0], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1],
                   [0.1, 0.4, 0.2], [0.7, 0.2, 0.1], [0.4, 0.1, 0.0]])
SCORES = ["precision", "recall", "f1", "auc"]


def test_confusion_counts_toy():
    """Counts match the ones checked by hand, and every category sums to N."""
    y_pred = (Y_PROB >= 0.5).astype(int)
    tp, fp, fn, tn = confusion_counts(Y_TRUE, y_pred)
    np.testing.assert_array_equal(tp, [2, 1, 0])
    np.testing.assert_array_equal(fp, [1, 0, 0])
    np.testing.assert_array_equal(fn, [1, 1, 1])
    np.testing.assert_array_equal(tn, [2, 4, 5])
    np.testing.assert_array_equal(tp + fp + fn + tn, [6, 6, 6])


def test_per_category_scores_match_sklearn():
    """Precision, recall, F1 and AUC per category agree with sklearn."""
    table = classification_metrics(Y_TRUE, Y_PROB, 0.5, NAMES)
    y_pred = (Y_PROB >= 0.5).astype(int)
    p, r, f, _ = precision_recall_fscore_support(Y_TRUE, y_pred, average=None, zero_division=0)
    auc = roc_auc_score(Y_TRUE, Y_PROB, average=None)
    np.testing.assert_allclose(table.loc[NAMES, "precision"], p)
    np.testing.assert_allclose(table.loc[NAMES, "recall"], r)
    np.testing.assert_allclose(table.loc[NAMES, "f1"], f)
    np.testing.assert_allclose(table.loc[NAMES, "auc"], auc)


def test_macro_micro_match_sklearn():
    """The macro and micro rows agree with sklearn's averages."""
    table = classification_metrics(Y_TRUE, Y_PROB, 0.5, NAMES)
    y_pred = (Y_PROB >= 0.5).astype(int)
    for avg in ["macro", "micro"]:
        p, r, f, _ = precision_recall_fscore_support(Y_TRUE, y_pred, average=avg, zero_division=0)
        auc = roc_auc_score(Y_TRUE, Y_PROB, average=avg)
        np.testing.assert_allclose(table.loc[avg, SCORES], [p, r, f, auc])


def test_nothing_to_grade_is_nan():
    """A category with no real cases and no "yes" gets NaN everywhere, and macro skips it.
    (Our own rule: sklearn would give 0, so this is checked by hand.)"""
    table = classification_metrics(Y_TRUE[:3], Y_PROB[:3], 0.5, NAMES)
    assert table.loc["Pneumothorax", SCORES].isna().all()
    np.testing.assert_allclose(table.loc["macro", "precision"], 1.0)


def test_threshold_per_category():
    """A lower Pneumothorax threshold catches its one case, and AUC does not change."""
    base = classification_metrics(Y_TRUE, Y_PROB, 0.5, NAMES)
    low = classification_metrics(Y_TRUE, Y_PROB, np.array([0.5, 0.5, 0.15]), NAMES)
    assert low.loc["Pneumothorax", "tp"] == 1
    np.testing.assert_allclose(low.loc["Pneumothorax", "f1"], 1.0)
    np.testing.assert_allclose(low["auc"], base["auc"])
