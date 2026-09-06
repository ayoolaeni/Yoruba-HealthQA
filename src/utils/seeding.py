"""Deterministic seeding shared by every pipeline stage.

Rule 2 of the build spec: every script sets and records a random seed. Call
`seed_everything` once, near the top of a script's `main()`, and pass the same
seed into the run manifest via `src.utils.manifest`.
"""
from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> int:
    """Seed python, numpy, and torch (if imported) RNGs. Returns the seed for logging."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
