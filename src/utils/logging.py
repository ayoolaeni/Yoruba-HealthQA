"""Shared console logging setup (rich-backed, falls back to stdlib logging)."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured (e.g. re-imported in tests)

    logger.setLevel(level)
    try:
        from rich.logging import RichHandler

        handler = RichHandler(show_path=False, rich_tracebacks=True)
        formatter = logging.Formatter("%(message)s")
    except ImportError:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class MissingInputError(RuntimeError):
    """Raised when a pipeline stage's required input file/dir does not exist.

    Per build-spec rule 1 ("never invent data"): stages must fail loudly with
    this error naming the missing path, rather than silently skipping or
    fabricating a substitute.
    """

    def __init__(self, path: str, stage: str, hint: str = ""):
        msg = f"[{stage}] required input not found: {path}"
        if hint:
            msg += f"\n  hint: {hint}"
        super().__init__(msg)
        self.path = path
        self.stage = stage
