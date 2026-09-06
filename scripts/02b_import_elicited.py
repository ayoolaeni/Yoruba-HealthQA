#!/usr/bin/env python
"""Phase 1.2b -- Import researcher-collected Yoruba questions from a CSV and
match them to source-grounded English answers.

Expected input CSV columns (header row required):
    question_yo, matched_claim_id (optional), notes (optional)

matched_claim_id, if given, must reference a claim_id already present in one
of the --working-jsonl files (typically data/interim/02_build_qa.jsonl) --
its answer_en/source_ref/topic are copied across. Rows with no
matched_claim_id, or one that does not resolve, are kept but marked
question_type="out_of_scope" with an empty answer, per build spec 1.2:
"marking unmatched ones question_type=out_of_scope".

This script never invents an answer for an elicited question: matching is
purely a CSV-driven join over claim_id, done by the researcher who collected
the questions, not guessed by this script.

Usage:
    python scripts/02b_import_elicited.py --csv data/raw/elicited_questions.csv \
        --working-jsonl data/interim/02_build_qa.jsonl \
        --out data/interim/02b_import_elicited.jsonl
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import read_jsonl, write_jsonl
from src.data.working_record import WorkingRecord, make_id
from src.data.yoruba_text import normalize_nfc
from src.utils.logging import MissingInputError, get_logger
from src.utils.manifest import RunManifest
from src.utils.seeding import seed_everything

logger = get_logger("02b_import_elicited")

REQUIRED_CSV_COLUMNS = {"question_yo"}


def load_claims_by_id(working_jsonl_paths: list[Path]) -> dict[str, dict]:
    by_claim_id: dict[str, dict] = {}
    for path in working_jsonl_paths:
        for record in read_jsonl(path):
            by_claim_id.setdefault(record["claim_id"], record)
    return by_claim_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--working-jsonl", nargs="+", default=["data/interim/02_build_qa.jsonl"])
    parser.add_argument("--out", default="data/interim/02b_import_elicited.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise MissingInputError(str(csv_path), stage="02b_import_elicited",
                                 hint="researcher must supply a CSV of elicited Yoruba questions")

    working_paths = [Path(p) for p in args.working_jsonl]
    for p in working_paths:
        if not p.exists():
            raise MissingInputError(str(p), stage="02b_import_elicited",
                                     hint="run scripts/02_build_qa.py first, or point --working-jsonl elsewhere")

    seed = seed_everything(args.seed)
    claims_by_id = load_claims_by_id(working_paths)

    with RunManifest(stage="02b_import_elicited", config={"csv": str(csv_path)}, seed=seed) as run:
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not REQUIRED_CSV_COLUMNS.issubset(set(reader.fieldnames)):
                raise ValueError(f"{csv_path} must have at least columns {REQUIRED_CSV_COLUMNS}, "
                                 f"got {reader.fieldnames}")
            rows = list(reader)

        # continue id numbering after the highest existing yhqa-###### id across
        # all working files, so ids stay globally unique when scripts are re-run
        existing_ids = [int(cid.split("-")[1]) for cid in
                        (r["id"] for r in claims_by_id.values()) if cid.startswith("yhqa-")]
        next_id = [max(existing_ids, default=0) + 1]

        matched, unmatched = 0, 0
        records: list[WorkingRecord] = []
        for row in rows:
            question_yo = normalize_nfc(row["question_yo"].strip())
            if not question_yo:
                continue
            claim_id = (row.get("matched_claim_id") or "").strip()
            source_claim = claims_by_id.get(claim_id) if claim_id else None

            if source_claim:
                matched += 1
                record = WorkingRecord(
                    id=make_id(next_id[0]),
                    claim_id=claim_id,
                    topic=source_claim["topic"],
                    question_type="elicited",
                    question_en="",  # elicited questions originate in Yoruba, not English
                    answer_en=source_claim["answer_en"],
                    source_ref=source_claim["source_ref"],
                    source_name=source_claim["source_name"],
                    source_url=source_claim.get("source_url"),
                    question_yo_final=question_yo,
                    notes=row.get("notes", ""),
                )
            else:
                unmatched += 1
                if claim_id:
                    logger.warning(f"matched_claim_id '{claim_id}' not found -- marking out_of_scope")
                record = WorkingRecord(
                    id=make_id(next_id[0]),
                    claim_id=claim_id or f"elicited-unmatched-{next_id[0]:06d}",
                    topic="unclassified",
                    question_type="out_of_scope",
                    question_en="",
                    answer_en="",
                    source_ref="",
                    source_name="elicited",
                    question_yo_final=question_yo,
                    notes=row.get("notes", ""),
                )
            next_id[0] += 1
            records.append(record)

        n = write_jsonl((r.to_dict() for r in records), args.out)
        run.record_output(args.out)
        logger.info(f"wrote {n} elicited records ({matched} matched, {unmatched} unmatched) to {args.out}")


if __name__ == "__main__":
    main()
