"""Step 1.5: the 3 encoders x 3 metrics table. For each checkpoint in configs/base.yaml ->
efficiency.checkpoints: the optimizing metric (val macro AUC, from the checkpoint's results.csv
row), the model size and the CPU latency (measured here, all models in this one process, back to
back), pass/fail against the thresholds written before measuring, and the winner.
Prints the table and saves it to reports/encoders.csv.
Run: python -m src.evaluation.compare_encoders
"""

import json

import pandas as pd
import torch

from src.data.inventory import PROJECT_ROOT
from src.data.labels import LABEL_MAP_PATH
from src.data.splits import load_config
from src.evaluation.efficiency import cpu_latency_ms, model_size_mb
from src.models.encoders import PRETRAINED_ENCODERS, SmallCNN
from src.utils import RESULTS_PATH, set_seeds

BASE_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
TABLE_PATH = PROJECT_ROOT / "reports" / "encoders.csv"


def build_model(name, cfg, n_classes):
    """The network a results row describes, with random weights (the caller loads the checkpoint).
    name: the row's model without "_stageN": "small_cnn", "resnet50" or "mobilenet_v2".
    cfg: the row's config as a dict."""
    if name == "small_cnn":
        return SmallCNN(cfg["channels"], n_classes)  # its channels come from the config
    # weights None: no ImageNet weights, the checkpoint replaces every parameter and buffer anyway
    return PRETRAINED_ENCODERS[name](n_classes, None)


def main():
    # 1. settings, seeds, the number of categories, all results rows
    eff = load_config(BASE_CONFIG_PATH)["efficiency"]
    set_seeds(eff["seed"])
    n_classes = len(load_config(LABEL_MAP_PATH))
    results = pd.read_csv(RESULTS_PATH)

    rows = []
    for checkpoint in eff["checkpoints"]:
        # 2. the checkpoint's results row (exactly one: run_ids are unique), its config as a dict
        row = results[results["checkpoint"] == checkpoint].iloc[0]
        cfg = json.loads(row["config"])
        # 3. the network, then the kept weights; map_location="cpu": no GPU needed (the use case)
        # "resnet50_stage2" -> "resnet50"; "small_cnn" stays
        name = row["model"].split("_stage")[0]
        model = build_model(name, cfg, n_classes)
        model.load_state_dict(torch.load(PROJECT_ROOT / checkpoint, map_location="cpu"))
        # 4. the two satisficing metrics
        size = model_size_mb(model)
        latency = cpu_latency_ms(
            model,
            cfg["image_size"],
            eff["latency_threads"],
            eff["latency_warmup"],
            eff["latency_runs"],
        )
        # 5. one table row: the optimizing metric, the two satisficing ones, pass/fail
        rows.append(
            {
                "model": row["model"],
                # the column named in the config: "val_macro_auc"
                eff["optimize"]: row[eff["optimize"]],
                "size_mb": size,
                "latency_ms": latency,
                "size_ok": size <= eff["max_size_mb"],
                "latency_ok": latency <= eff["max_latency_ms"],
                "checkpoint": checkpoint,
            }
        )

    # 6. print and save the table
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_PATH, index=False)

    # 7. the rule written before measuring: among the models that pass BOTH satisficing
    #    metrics, the highest optimizing metric wins
    passing = table[table["size_ok"] & table["latency_ok"]]
    winner = passing.loc[passing[eff["optimize"]].idxmax(), "model"]
    print(
        f"winner: {winner} (highest {eff['optimize']} with size <= {eff['max_size_mb']} MB "
        f"and latency <= {eff['max_latency_ms']} ms)"
    )


if __name__ == "__main__":
    main()
