"""Train the small CNN (Step 1.3) on train for `epochs`, keep the epoch with the lowest
val loss, score it on train and val, save its weights and append one row to
experiments/results.csv.
Run: python -m src.training.train_cnn --comment "one sentence about this run"
"""

import argparse
import json
from datetime import datetime

import pandas as pd
import torch
from torch import nn

from src.data.inventory import PROJECT_ROOT
from src.data.labels import LABEL_MAP_PATH
from src.data.pipeline import make_loader
from src.data.splits import load_config
from src.evaluation.metrics_cls import classification_metrics
from src.models.encoders import SmallCNN
from src.training.loops import evaluate, fit
from src.utils import RESULTS_PATH, append_result, set_seeds
from src.data.augment import build_train_transform


CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
HISTORY_DIR = PROJECT_ROOT / "experiments" / "history"


def main():
    # Part A: setup

    # 1. read --comment from the command line (as in train_baseline)
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", required=True)
    parser.add_argument("--config", default="configs/small_cnn.yaml")

    args = parser.parse_args()
    # 2. read the config, set the seeds; categories; device ("cuda" if available, else "cpu");
    config_path = PROJECT_ROOT / args.config
    cfg = load_config(config_path)
    categories = list(load_config(LABEL_MAP_PATH))

    set_seeds(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")   (names this run's files)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    # 3. loaders: train and val (image_size and batch_size from the config)
    image_size = cfg["image_size"]
    batch_size = cfg["batch_size"]
    transform = build_train_transform(
        cfg.get("augmentation")
    )  # no section (small_cnn.yaml) -> None
    train_loader = make_loader("train", image_size, batch_size, transform)
    val_loader = make_loader("val", image_size, batch_size, None)  # never augmented
    train_plain_loader = make_loader("train", image_size, batch_size, None)

    # 4. model = SmallCNN(channels from the config, number of categories), on the device;
    model = SmallCNN(cfg["channels"], len(categories))
    #    loss_fn = BCEWithLogitsLoss; optimizer = Adam with the config's learning_rate
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), cfg["learning_rate"])

    # Part B: train, keep the best epoch

    # 5-6. train for `epochs`, keep the epoch with the lowest val loss (loop in loops.fit)
    best_state, best_epoch, best_val_loss, history = fit(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device,
        cfg["epochs"],
        cfg["threshold"],
        categories,
        recalibrate_loader=train_plain_loader,  # SmallCNN's BatchNorm trains -> recompute
    )

    # 7.
    model.load_state_dict(best_state)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"small_cnn_{run_id}.pt"
    torch.save(best_state, checkpoint_path)
    # 8.

    train_loss_kept, y_train, p_train = evaluate(
        model, train_plain_loader, loss_fn, device
    )
    val_loss_kept, y_val, p_val = evaluate(model, val_loader, loss_fn, device)
    table_train = classification_metrics(y_train, p_train, cfg["threshold"], categories)
    table_val = classification_metrics(y_val, p_val, cfg["threshold"], categories)
    print(table_val.round(3).to_string())
    print(
        f"kept epoch {best_epoch}: val loss {val_loss_kept:.6f} (best seen {best_val_loss:.6f})"
    )

    # Part C: log

    # 9. one row per epoch -> experiments/history/ (the learning curve)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history_path = HISTORY_DIR / f"small_cnn_{run_id}.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    table_val.to_csv(HISTORY_DIR / f"small_cnn_{run_id}_val_classes.csv")

    # 10. build one row, append it to experiments/results.csv
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": "small_cnn",
        "config_path": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "seed": cfg["seed"],
        # the whole config as text: later edits can't lose it
        "config": json.dumps(cfg),
        # the kept model's loss on train, like the baseline's final cost
        "final_cost": train_loss_kept,
        "train_macro_f1": table_train.loc["macro", "f1"],
        "val_macro_f1": table_val.loc["macro", "f1"],
        "train_micro_f1": table_train.loc["micro", "f1"],
        "val_micro_f1": table_val.loc["micro", "f1"],
        "train_macro_auc": table_train.loc["macro", "auc"],
        "val_macro_auc": table_val.loc["macro", "auc"],
        "best_epoch": best_epoch,
        "val_loss": val_loss_kept,
        "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "comment": args.comment,
    }
    append_result(row)
    print(
        f"best epoch {best_epoch}/{cfg['epochs']}: val macro F1 {row['val_macro_f1']:.3f}, "
        f"micro F1 {row['val_micro_f1']:.3f}, macro AUC {row['val_macro_auc']:.3f} "
        f"-> row added to {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
