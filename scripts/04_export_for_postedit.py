#!/usr/bin/env python
"""Phase 1.5 -- Export the MT'd working set to a spreadsheet for human
post-editing.

Columns (build spec 1.5, exact order):
    id | topic | question_en | answer_en | question_yo_mt | answer_yo_mt |
    question_yo_final | answer_yo_final | editor_id | notes

question_yo_final/answer_yo_final start out as a COPY of the _mt columns so
the editor is correcting machine output in place rather than typing from
scratch; 04b_import_postedit.py detects which rows were actually changed.

Usage:
    python scripts/04_export_for_postedit.py --in data/interim/03_translate.jsonl \
        --out data/interim/postedit_batch_01.xlsx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import read_jsonl
from src.utils.logging import MissingInputError, get_logger
from src.utils.manifest import RunManifest
from src.utils.seeding import seed_everything

logger = get_logger("04_export_for_postedit")

POSTEDIT_COLUMNS = [
    "id", "topic", "question_en", "answer_en", "question_yo_mt", "answer_yo_mt",
    "question_yo_final", "answer_yo_final", "editor_id", "notes",
]


def build_rows(records: list[dict]) -> list[dict]:
    rows = []
    for r in records:
        rows.append({
            "id": r["id"],
            "topic": r.get("topic", ""),
            "question_en": r.get("question_en", ""),
            "answer_en": r.get("answer_en", ""),
            "question_yo_mt": r.get("question_yo_mt", ""),
            "answer_yo_mt": r.get("answer_yo_mt", ""),
            # Pre-filled with MT output; the editor overwrites in place.
            "question_yo_final": r.get("question_yo_final") or r.get("question_yo_mt", ""),
            "answer_yo_final": r.get("answer_yo_final") or r.get("answer_yo_mt", ""),
            "editor_id": r.get("editor_id", ""),
            "notes": r.get("notes", ""),
        })
    return rows


def write_xlsx(rows: list[dict], out_path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "postedit"
    ws.append(POSTEDIT_COLUMNS)
    for row in rows:
        ws.append([row[c] for c in POSTEDIT_COLUMNS])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


def write_csv(rows: list[dict], out_path: Path) -> None:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POSTEDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input", default="data/interim/03_translate.jsonl")
    parser.add_argument("--out", default="data/interim/postedit_batch_01.xlsx")
    parser.add_argument("--only-untranslated-ok", action="store_true",
                         help="include records even if question_yo_mt/answer_yo_mt are empty "
                              "(default: skip them, since there is nothing to post-edit yet)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise MissingInputError(str(in_path), stage="04_export_for_postedit",
                                 hint="run scripts/03_translate.py first")

    seed = seed_everything(args.seed)
    records = list(read_jsonl(in_path))
    if not args.only_untranslated_ok:
        before = len(records)
        records = [r for r in records if r.get("question_yo_mt") or r.get("answer_yo_mt")]
        skipped = before - len(records)
        if skipped:
            logger.warning(f"skipped {skipped} records with no MT output yet (question_type=out_of_scope "
                            f"or elicited records with no question_en to translate)")

    out_path = Path(args.out)
    with RunManifest(stage="04_export_for_postedit", config={"input": str(in_path)}, seed=seed) as run:
        rows = build_rows(records)
        if out_path.suffix == ".xlsx":
            write_xlsx(rows, out_path)
        elif out_path.suffix == ".csv":
            write_csv(rows, out_path)
        else:
            raise ValueError(f"--out must end in .xlsx or .csv, got {out_path}")
        run.record_output(str(out_path))
        logger.info(f"wrote {len(rows)} rows to {out_path} for post-editing")


if __name__ == "__main__":
    main()
