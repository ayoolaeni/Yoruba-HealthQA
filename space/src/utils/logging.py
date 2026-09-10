"""Shared console logging setup (rich-backed, falls back to stdlib logging)."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured (e.g. re-imported in tests)

    logger.setLevel(level)

    # Windows' default console codepage (cp1252) cannot represent most Yoruba
    # diacritics. Without this, logging a Yoruba string (e.g. an import
    # validation error quoting the offending text) crashes the whole script
    # with a raw UnicodeEncodeError from deep inside the logging handler --
    # verified happening from rich's legacy Win32 console writer. Reconfigure
    # both streams to UTF-8 with a safe fallback so a log line never crashes
    # the process, on any platform/console.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
            except Exception:
                pass  # piped/redirected streams may not support reconfigure; non-fatal

    try:
        from rich.console import Console
        from rich.logging import RichHandler

        # legacy_windows=False forces rich onto its ANSI/UTF-8 code path
        # instead of the Win32 console API that crashed above.
        console = Console(file=sys.stderr, legacy_windows=False)
        handler = RichHandler(console=console, show_path=False, rich_tracebacks=True)
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
