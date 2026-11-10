"""Small helpers shared by every entry point."""

import random

import numpy as np
import torch
import pandas as pd

from src.data.inventory import PROJECT_ROOT

RESULTS_PATH = PROJECT_ROOT / "experiments" / "results.csv"


def set_seeds(seed):
    """Seed Python's random, numpy and torch (CPU and every GPU) so a run
    repeats exactly. Call it first thing in every script; seed comes from the config."""
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # numpy's global random generator
    torch.manual_seed(seed)  # PyTorch
    # cuDNN uses only algorithms that add in a fixed order (a rerun gives identical numbers, ~6% slower)
    torch.backends.cudnn.deterministic = True
    # don't let cuDNN try algorithms at startup and pick the fastest (that choice can differ between runs)
    torch.backends.cudnn.benchmark = False


def append_result(row, path=RESULTS_PATH):
    """Add one run's row (a dict) to the results file. Columns the file doesn't have
    yet are added; older rows keep empty cells there."""
    # 1. the row as a one-row DataFrame, rounded to 4 decimals (as the baseline did)
    new_row = pd.DataFrame([row]).round(4)
    # 2. if the file exists: read it and put the new row under it (pd.concat, ignore_index=True)
    if path.exists():
        old_table = pd.read_csv(path)
        table = pd.concat([old_table, new_row], ignore_index=True)
    else:
        table = new_row
    #    otherwise: the new row is the whole table
    # 3. make sure the folder exists, then write the whole table back, without the index
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
