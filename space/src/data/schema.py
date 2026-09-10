"""Record schema for the final JSONL dataset (build spec Table 3.5).

    {
      "id": "yhqa-000123",
      "claim_id": "who-malaria-007",
      "instruction": "...",
      "instruction_nodia": "...",
      "input": "",
      "output": "...",
      "topic": "malaria",
      "question_type": "derived | elicited | out_of_scope",
      "source_en": "...",
      "source_ref": "...",
      "validated_ling": true,
      "validated_clin": true
    }

Every stage from 02_build_qa.py onward reads and writes records matching this
schema. Validate with `validate_record` before writing anything to
data/interim, data/validated, or data/final.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

from src.data.yoruba_text import normalize_nfc, strip_diacritics
from src.eval.metrics import MIN_LANGUAGE_CONSISTENCY_FOR_YORUBA_TEXT, language_consistency

QUESTION_TYPES = {"derived", "elicited", "out_of_scope"}

REQUIRED_FIELDS = [
    "id", "claim_id", "instruction", "instruction_nodia", "input", "output",
    "topic", "question_type", "source_en", "source_ref",
    "validated_ling", "validated_clin",
]

YORUBA_TEXT_FIELDS = ["instruction", "output"]


def load_topics(data_config_path: str | Path = "configs/data.yaml") -> set[str]:
    with open(data_config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return set(cfg["topics"])


class SchemaError(ValueError):
    pass


def validate_record(record: dict[str, Any], valid_topics: set[str] | None = None, strict_charset: bool = True) -> list[str]:
    """Return a list of human-readable problems with `record` (empty = valid).
    Does not raise -- callers decide whether a bad record is discarded,
    corrected, or fails the whole build."""
    problems: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in record:
            problems.append(f"missing field '{field}'")

    if problems:
        return problems  # can't check further without the base fields

    if record["question_type"] not in QUESTION_TYPES:
        problems.append(f"question_type '{record['question_type']}' not in {QUESTION_TYPES}")

    if valid_topics is not None and record["topic"] not in valid_topics:
        problems.append(f"topic '{record['topic']}' not in configured topic list")

    if not isinstance(record["validated_ling"], bool):
        problems.append("validated_ling must be boolean")
    if not isinstance(record["validated_clin"], bool):
        problems.append("validated_clin must be boolean")

    expected_nodia = strip_diacritics(record["instruction"])
    if record["instruction_nodia"] != expected_nodia:
        problems.append(
            f"instruction_nodia out of sync: got {record['instruction_nodia']!r}, "
            f"expected {expected_nodia!r} (re-run after any instruction edit)"
        )

    for field in YORUBA_TEXT_FIELDS:
        text = record.get(field, "")
        if not text:
            continue
        if normalize_nfc(text) != text:
            problems.append(f"field '{field}' is not NFC-normalised")
        if strict_charset:
            # Proportion-based, not per-character: see
            # src.eval.metrics.MIN_LANGUAGE_CONSISTENCY_FOR_YORUBA_TEXT for
            # why a "zero non-Yoruba characters anywhere" check is wrong for
            # real post-edited health text (it rejects legitimate drug/virus/
            # organisation names that professional translation keeps in
            # English).
            consistency = language_consistency([text])
            if consistency < MIN_LANGUAGE_CONSISTENCY_FOR_YORUBA_TEXT:
                problems.append(
                    f"field '{field}' is only {consistency:.0%} Yoruba-consistent "
                    f"(need >= {MIN_LANGUAGE_CONSISTENCY_FOR_YORUBA_TEXT:.0%}) -- looks mostly untranslated"
                )

    return problems


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise SchemaError(f"{path}:{line_no}: invalid JSON ({e})") from e


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n
