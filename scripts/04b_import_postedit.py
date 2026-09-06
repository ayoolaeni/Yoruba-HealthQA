#!/usr/bin/env python
"""Phase 1.5 -- Import a completed post-editing spreadsheet.

Validates that question_yo_final/answer_yo_final are filled in and use only
legal Yoruba characters (src/data/yoruba_text.validate_charset), merges the
edits back into the working JSONL, and sets post_edited=True for rows the
editor actually changed relative to the _mt columns.

Usage:
    python scripts/04b_import_postedit.py --edited data/interim/postedit_batch_01.xlsx \
        --working-jsonl data/interim/03_translate.jsonl \
        --out data/interim/04b_import_postedit.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import read_jsonl, write_jsonl
from src.data.yoruba_text import normalize_nfc, validate_charset
from src.utils.logging import MissingInputError, get_logger
from src.utils.manifest import RunManifest
from src.utils.seeding import seed_everything

logger = get_logger("04b_import_postedit")

REQUIRED_COLUMNS = ["id", "question_yo_final", "answer_yo_final"]


def read_xlsx_rows(path: Path) -> list[dict]:
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)
    header = [str(h) if h is not None else "" for h in next(rows_iter)]
    rows = []
    for values in rows_iter:
        row = dict(zip(header, values))
        if row.get("id"):
            rows.append({k: ("" if v is None else str(v)) for k, v in row.items()})
    return rows


def read_csv_rows(path: Path) -> list[dict]:
    import csv

    with open(path, encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("id")]


def validate_edited_rows(rows: list[dict]) -> list[str]:
    """Returns a list of problem strings (empty = all rows valid)."""
    problems = []
    for row in rows:
        for col in REQUIRED_COLUMNS:
            if not (row.get(col) or "").strip():
                problems.append(f"row id={row.get('id', '?')}: '{col}' is empty")
        for field in ("question_yo_final", "answer_yo_final"):
            text = (row.get(field) or "").strip()
            if not text:
                continue
            result = validate_charset(text, strict=True, allow_loanwords=True)
            if not result.is_valid:
                problems.append(f"row id={row.get('id', '?')}: '{field}' has illegal characters "
                                 f"{result.invalid_chars!r} -- {text!r}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edited", required=True)
    parser.add_argument("--working-jsonl", required=True)
    parser.add_argument("--out", default="data/interim/04b_import_postedit.jsonl")
    parser.add_argument("--allow-invalid-charset", action="store_true",
                         help="import anyway despite charset validation failures (still logs them); "
                              "use only for a deliberately mixed-script debugging row, never for real data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    edited_path = Path(args.edited)
    if not edited_path.exists():
        raise MissingInputError(str(edited_path), stage="04b_import_postedit")
    working_path = Path(args.working_jsonl)
    if not working_path.exists():
        raise MissingInputError(str(working_path), stage="04b_import_postedit")

    seed = seed_everything(args.seed)

    if edited_path.suffix == ".xlsx":
        edited_rows = read_xlsx_rows(edited_path)
    elif edited_path.suffix == ".csv":
        edited_rows = read_csv_rows(edited_path)
    else:
        raise ValueError(f"--edited must be .xlsx or .csv, got {edited_path}")

    problems = validate_edited_rows(edited_rows)
    if problems and not args.allow_invalid_charset:
        logger.error(f"{len(problems)} validation problem(s) in {edited_path}:")
        for p in problems[:50]:
            logger.error(f"  {p}")
        raise ValueError(f"{edited_path} failed post-edit import validation ({len(problems)} problems); "
                          f"fix the spreadsheet and re-run, or pass --allow-invalid-charset to override")
    elif problems:
        logger.warning(f"{len(problems)} validation problem(s) found but importing anyway (--allow-invalid-charset)")

    edited_by_id = {row["id"]: row for row in edited_rows}
    working_records = list(read_jsonl(working_path))

    with RunManifest(stage="04b_import_postedit", config={"edited": str(edited_path)}, seed=seed) as run:
        n_edited = 0
        for record in working_records:
            edit = edited_by_id.get(record["id"])
            if not edit:
                continue
            question_final = normalize_nfc((edit.get("question_yo_final") or "").strip())
            answer_final = normalize_nfc((edit.get("answer_yo_final") or "").strip())
            changed = (
                question_final != normalize_nfc((record.get("question_yo_mt") or "").strip())
                or answer_final != normalize_nfc((record.get("answer_yo_mt") or "").strip())
            )
            record["question_yo_final"] = question_final
            record["answer_yo_final"] = answer_final
            record["editor_id"] = edit.get("editor_id") or record.get("editor_id")
            record["notes"] = edit.get("notes") or record.get("notes", "")
            record["post_edited"] = changed
            n_edited += 1

        n = write_jsonl(working_records, args.out)
        run.record_output(args.out)
        logger.info(f"merged {n_edited} post-edited rows into {n} total records -> {args.out}")


if __name__ == "__main__":
    main()
