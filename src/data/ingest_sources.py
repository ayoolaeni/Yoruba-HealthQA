"""HTTP fetch + licence gate shared by scripts/01_ingest.py.

Kept out of the script file so it can be unit-tested (licence gate logic)
without making network calls.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LicenceMissingError(RuntimeError):
    def __init__(self, source_name: str):
        super().__init__(
            f"source '{source_name}' has no recorded licence in configs/data.yaml "
            f"(licence: null) -- refusing to ingest. Add a licence and licence_url "
            f"after confirming usage terms, per build spec rule: "
            f"'Refuse to ingest a source with no recorded licence.'"
        )
        self.source_name = source_name


@dataclass
class IngestedDocument:
    source_name: str
    url: str | None
    retrieval_date: str
    licence: str
    licence_url: str | None
    sha256: str
    raw_text: str

    def to_json_record(self) -> dict[str, Any]:
        d = asdict(self)
        # raw_text can be large; still stored, since the manifest rule requires
        # every artefact be traceable, but keep it last for readability.
        return d


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_licence_or_raise(source_name: str, source_cfg: dict) -> None:
    if not source_cfg.get("licence"):
        raise LicenceMissingError(source_name)


def build_document(source_name: str, source_cfg: dict, url: str | None, raw_text: str) -> IngestedDocument:
    check_licence_or_raise(source_name, source_cfg)
    return IngestedDocument(
        source_name=source_name,
        url=url,
        retrieval_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        licence=source_cfg["licence"],
        licence_url=source_cfg.get("licence_url"),
        sha256=sha256_of(raw_text),
        raw_text=raw_text,
    )


def write_document(doc: IngestedDocument, raw_dir: Path, slug: str) -> Path:
    out_dir = raw_dir / doc.source_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc.to_json_record(), f, ensure_ascii=False, indent=2)
    return out_path
