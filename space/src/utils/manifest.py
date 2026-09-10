"""Per-run manifest writer.

Rule 2 of the build spec: every run writes a JSON manifest to
runs/<timestamp>/manifest.json containing config, git commit hash, package
versions, seed, and wall-clock duration. Every number reported in Chapter 4
must be traceable to one of these manifests (definition-of-done item).

Usage:
    with RunManifest(stage="10_evaluate", config=cfg, seed=42) as run:
        ... do work, write outputs into run.dir ...
        run.record_output("reports/table_4_5_automatic_metrics.csv")
    # manifest.json is written automatically on exit, including duration
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit_hash(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "UNKNOWN (not a git repo, no commits yet, or git unavailable)"


def _git_dirty(repo_root: Path) -> bool | str:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return "UNKNOWN"


def _pip_freeze() -> list[str]:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        return sorted(out.stdout.strip().splitlines())
    except Exception:
        return ["UNKNOWN (pip freeze failed)"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class RunManifest:
    def __init__(self, stage: str, config: dict[str, Any] | None = None, seed: int | None = None,
                 runs_dir: Path | str | None = None):
        self.stage = stage
        self.config = config or {}
        self.seed = seed
        self.repo_root = _repo_root()
        self.runs_dir = Path(runs_dir) if runs_dir else self.repo_root / "runs"
        self.timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dir = self.runs_dir / f"{self.timestamp}_{stage}"
        self.outputs: list[str] = []
        self._start = None

    def __enter__(self) -> "RunManifest":
        self.dir.mkdir(parents=True, exist_ok=True)
        self._start = time.time()
        return self

    def record_output(self, path: str | Path) -> None:
        self.outputs.append(str(path))

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_s = time.time() - self._start if self._start else None
        manifest = {
            "stage": self.stage,
            "timestamp_utc": self.timestamp,
            "seed": self.seed,
            "config": self.config,
            "outputs": self.outputs,
            "duration_seconds": duration_s,
            "status": "failed" if exc_type else "ok",
            "error": f"{exc_type.__name__}: {exc_val}" if exc_type else None,
            "git": {
                "commit": _git_commit_hash(self.repo_root),
                "dirty": _git_dirty(self.repo_root),
            },
            "environment": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "pip_freeze": _pip_freeze(),
            },
        }
        with open(self.dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
        # do not suppress exceptions
        return None
