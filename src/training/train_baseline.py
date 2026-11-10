"""Train the numpy logistic-regression baseline (Step 0.5) on train, score it on
train and val, and append one row to experiments/results.csv.
Run: python -m src.training.train_baseline --comment "one sentence about this run"
"""

import argparse
import json
from datetime import datetime

import numpy as np

from src.data.inventory import PROJECT_ROOT
from src.data.labels import LABEL_MAP_PATH
from src.data.small_images import CONFIG_PATH, PROCESSED_DIR
from src.data.splits import load_config
from src.evaluation.metrics_cls import classification_metrics
from src.models.baseline_numpy import initialize, optimize, predict_proba, propagate
from src.utils import RESULTS_PATH, append_result, set_seeds


def main():
    # Part A: setup

    # 1. read --comment from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", required=True)
    args = parser.parse_args()

    # 2. read the config, set the seeds
    cfg = load_config(CONFIG_PATH)
    set_seeds(cfg["seed"])
    size = cfg["image_size"]
    num_iterations = cfg["num_iterations"]
    learning_rate = cfg["learning_rate"]
    threshold = cfg["threshold"]
    categories = list(load_config(LABEL_MAP_PATH))

    # 3. load train and val arrays
    data = np.load(PROCESSED_DIR / f"train_{size}.npz")
    data_val = np.load(PROCESSED_DIR / f"val_{size}.npz")
    X_train = data["X"]
    Y_train = data["Y"]
    X_val = data_val["X"]
    Y_val = data_val["Y"]
    print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)
    n_classes = Y_train.shape[1]
    n_x = X_train.shape[1]

    # Part B: train and evaluate

    # 4. initialize, optimize on train
    W, b = initialize(n_x, n_classes)
    # optimize prints the cost every 100 steps; the list itself isn't needed here
    W, b, _ = optimize(W, b, X_train, Y_train, learning_rate, num_iterations)
    # the printed costs stop at step 1900; this is the cost of the finished model
    _, final_cost = propagate(W, b, X_train, Y_train)

    # 5. predict_proba on train and val -> classification_metrics -> two tables
    prob_train = predict_proba(W, b, X_train)
    prob_val = predict_proba(W, b, X_val)
    table_train = classification_metrics(Y_train, prob_train, threshold, categories)
    table_val = classification_metrics(Y_val, prob_val, threshold, categories)

    # 6. print the val table
    print(table_val.round(3).to_string())

    # Part C: log

    # 7. build one row, append it to experiments/results.csv
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": "baseline_numpy",
        "config_path": CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "seed": cfg["seed"],
        "config": json.dumps(
            cfg
        ),  # the whole config as text: later edits can't lose it
        "final_cost": final_cost,
        "train_macro_f1": table_train.loc["macro", "f1"],
        "val_macro_f1": table_val.loc["macro", "f1"],
        "train_micro_f1": table_train.loc["micro", "f1"],
        "val_micro_f1": table_val.loc["micro", "f1"],
        "train_macro_auc": table_train.loc["macro", "auc"],
        "val_macro_auc": table_val.loc["macro", "auc"],
        "comment": args.comment,
    }

    append_result(row)
    print(
        f"val macro F1 {row['val_macro_f1']:.3f}, micro F1 {row['val_micro_f1']:.3f}, "
        f"macro AUC {row['val_macro_auc']:.3f} -> row added to {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
