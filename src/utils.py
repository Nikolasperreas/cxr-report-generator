"""Small helpers shared by every entry point."""

import random

import numpy as np
import torch


def set_seeds(seed):
    """Seed Python's random, numpy and torch (CPU and every GPU) so a run
    repeats exactly. Call it first thing in every script; seed comes from the config."""
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # numpy's global random generator
    torch.manual_seed(seed)  # PyTorch
