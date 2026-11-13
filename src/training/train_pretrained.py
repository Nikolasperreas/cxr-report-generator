"""Train an ImageNet-pretrained encoder in two stages (Step 1.4 ResNet50, Step 1.5 MobileNetV2).
The config's `encoder` picks the network: a key of PRETRAINED_ENCODERS in src/models/encoders.py.
Stage 1: frozen ImageNet backbone, only the head trains. Stage 2: from the kept stage-1 weights,
the parts in stage2.unfreeze train too, at a lower learning rate. Each stage keeps its epoch with
the lowest val loss, scores it on train and val, saves its weights and appends one row to
experiments/results.csv.
Run: python -m src.training.train_pretrained --config configs/resnet50.yaml --comment "one sentence about this run"
     (stage 2 only, from a saved stage 1:)
     python -m src.training.train_pretrained --config configs/resnet50.yaml --stage1-checkpoint checkpoints/resnet50_stage1_<run_id>.pt --comment "..."
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from src.data.inventory import PROJECT_ROOT
from src.data.labels import LABEL_MAP_PATH
from src.data.pipeline import make_loader
from src.data.splits import load_config
from src.evaluation.metrics_cls import classification_metrics
from src.models.encoders import PRETRAINED_ENCODERS
from src.training.loops import evaluate, fit
from src.utils import RESULTS_PATH, append_result, set_seeds
from src.data.augment import build_train_transform


CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
HISTORY_DIR = PROJECT_ROOT / "experiments" / "history"


def score_and_log(
    name,
    model,
    best_state,
    best_epoch,
    best_val_loss,
    history,
    train_plain_loader,
    val_loader,
    loss_fn,
    device,
    cfg,
    categories,
    config_path,
    run_id,
    comment,
    init_checkpoint,
):
    """Steps 7-10 for one stage: load the kept weights into the model, score them on plain
    train and val, save checkpoint + history + per-class val table, append one results row.
    name: "<encoder>_stage1" or "<encoder>_stage2", e.g. "resnet50_stage1" (file names and the
    row's model).
    init_checkpoint: the checkpoint this stage started from (None for stage 1).
    Returns the checkpoint path as written in the row (checkpoints/...pt)."""
    # 7. fit leaves the LAST epoch's weights in the model -> load the kept ones, save them
    model.load_state_dict(best_state)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{name}_{run_id}.pt"
    torch.save(best_state, checkpoint_path)

    # 8. score the kept model on plain train and val
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

    # 9. one row per epoch -> experiments/history/ (the learning curve)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history_path = HISTORY_DIR / f"{name}_{run_id}.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    table_val.to_csv(HISTORY_DIR / f"{name}_{run_id}_val_classes.csv")

    # 10. build one row, append it to experiments/results.csv
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": name,
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
        # the weights this stage started from (empty for stage 1)
        "init_checkpoint": init_checkpoint,
        "comment": comment,
    }
    append_result(row)
    print(
        f"best epoch {best_epoch}/{len(history)}: val macro F1 {row['val_macro_f1']:.3f}, "
        f"micro F1 {row['val_micro_f1']:.3f}, macro AUC {row['val_macro_auc']:.3f} "
        f"-> row added to {RESULTS_PATH}"
    )
    return row["checkpoint"]


def main():
    # Part A: setup

    # 1. read --comment, --config (and optionally --stage1-checkpoint) from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", required=True)
    # required: a default would silently train that encoder whenever --config is forgotten
    parser.add_argument("--config", required=True)  # e.g. configs/resnet50.yaml
    parser.add_argument(
        "--stage1-checkpoint", default=None
    )  # given -> skip stage 1, start stage 2 from it

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
    transform = build_train_transform(cfg.get("augmentation"))
    train_loader = make_loader("train", image_size, batch_size, transform)
    val_loader = make_loader("val", image_size, batch_size, None)  # never augmented
    train_plain_loader = make_loader("train", image_size, batch_size, None)

    # 4. the config's encoder with a frozen backbone (only the head trains), on the device; loss
    encoder = cfg["encoder"]  # e.g. "resnet50": names the files and the rows' model too
    model = PRETRAINED_ENCODERS[encoder](len(categories), cfg["weights"])
    model.to(device)
    model.freeze_backbone()
    loss_fn = nn.BCEWithLogitsLoss()

    # Part B: stage 1, head only (skipped when --stage1-checkpoint is given);
    # either way, `model` ends up holding the kept stage-1 weights
    if args.stage1_checkpoint is None:
        optimizer = torch.optim.Adam(
            model.head.parameters(), cfg["stage1"]["learning_rate"]
        )
        best_state, best_epoch, best_val_loss, history = fit(
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            device,
            cfg["stage1"]["epochs"],
            cfg["threshold"],
            categories,
        )
        stage1_checkpoint = score_and_log(
            f"{encoder}_stage1",
            model,
            best_state,
            best_epoch,
            best_val_loss,
            history,
            train_plain_loader,
            val_loader,
            loss_fn,
            device,
            cfg,
            categories,
            config_path,
            run_id,
            args.comment,
            init_checkpoint=None,
        )
    else:
        # as_posix: written like the checkpoint column (checkpoints/...pt), whatever was typed
        stage1_checkpoint = Path(args.stage1_checkpoint).as_posix()
        model.load_state_dict(
            torch.load(PROJECT_ROOT / stage1_checkpoint, map_location=device)
        )

    # Part C: stage 2, head + the stages in stage2.unfreeze, from the kept stage-1 weights
    set_seeds(cfg["seed"])  # same shuffles whether stage 1 ran just now or came from a file
    model.unfreeze(cfg["stage2"]["unfreeze"])
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, cfg["stage2"]["learning_rate"])
    best_state, best_epoch, best_val_loss, history = fit(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device,
        cfg["stage2"]["epochs"],
        cfg["threshold"],
        categories,
    )  # still no recalibrate_loader: the unfrozen parts' BatchNorm keeps the ImageNet statistics
    score_and_log(
        f"{encoder}_stage2",
        model,
        best_state,
        best_epoch,
        best_val_loss,
        history,
        train_plain_loader,
        val_loader,
        loss_fn,
        device,
        cfg,
        categories,
        config_path,
        run_id,
        args.comment,
        init_checkpoint=stage1_checkpoint,
    )


if __name__ == "__main__":
    main()
