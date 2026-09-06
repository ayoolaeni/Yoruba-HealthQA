#!/usr/bin/env python
"""Phase 1.4 -- Extract recurring clinical terms for the researcher to fix
canonical Yoruba renderings before post-editing starts (build spec 1.4).

Two modes:
  --extract (default): scans answer_en across the working JSONL and writes
      data/terminology.csv with term_en/frequency, canonical_yo left blank
      for the researcher to fill in.
  --check: once the researcher has returned terminology.csv with canonical_yo
      filled in, re-run with --check to flag records whose Yoruba text
      (question_yo_final/answer_yo_final, falling back to the _mt fields)
      does not contain the canonical rendering of a term used in its
      answer_en. Writes a flags report next to --out.

Usage:
    python scripts/03b_terminology.py --extract --in data/interim/02_build_qa.jsonl
    python scripts/03b_terminology.py --check --in data/interim/04b_import_postedit.jsonl \
        --terminology data/terminology.csv --out reports/terminology_flags.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import read_jsonl
from src.data.terminology import (
    check_record_terminology,
    extract_term_frequencies,
    load_canonical_terminology,
    write_terminology_csv,
)
from src.utils.logging import MissingInputError, get_logger
from src.utils.manifest import RunManifest
from src.utils.seeding import seed_everything

logger = get_logger("03b_terminology")


def record_yoruba_text(record: dict) -> str:
    parts = [
        record.get("question_yo_final") or record.get("question_yo_mt") or "",
        record.get("answer_yo_final") or record.get("answer_yo_mt") or "",
    ]
    return " ".join(p for p in parts if p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input", default="data/interim/02_build_qa.jsonl")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--terminology", default="data/terminology.csv")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.extract == args.check:
        parser.error("pass exactly one of --extract or --check")

    in_path = Path(args.input)
    if not in_path.exists():
        raise MissingInputError(str(in_path), stage="03b_terminology")

    seed = seed_everything(args.seed)
    records = list(read_jsonl(in_path))

    if args.extract:
        with RunManifest(stage="03b_terminology_extract", config={"input": str(in_path)}, seed=seed) as run:
            counts = extract_term_frequencies([r.get("answer_en", "") for r in records])
            n = write_terminology_csv(counts, args.terminology)
            run.record_output(args.terminology)
            logger.info(f"wrote {n} terms to {args.terminology} -- researcher must fill in canonical_yo "
                        f"before post-editing (04_export_for_postedit.py) starts")
        return

    terminology_path = Path(args.terminology)
    if not terminology_path.exists():
        raise MissingInputError(str(terminology_path), stage="03b_terminology",
                                 hint="run --extract first, then have the researcher fill in canonical_yo")
    canonical = load_canonical_terminology(terminology_path)
    if not canonical:
        logger.warning(f"{terminology_path} has no completed canonical_yo entries yet -- nothing to check")

    out_path = args.out or "reports/terminology_flags.csv"
    with RunManifest(stage="03b_terminology_check", config={"input": str(in_path), "terminology": str(terminology_path)}, seed=seed) as run:
        flagged_rows = []
        for record in records:
            yoruba_text = record_yoruba_text(record)
            if not yoruba_text:
                continue
            flags = check_record_terminology(record.get("answer_en", ""), yoruba_text, canonical)
            for term in flags:
                flagged_rows.append({"id": record["id"], "claim_id": record["claim_id"], "term_en": term,
                                      "expected_canonical_yo": canonical[term]})

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "claim_id", "term_en", "expected_canonical_yo"])
            writer.writeheader()
            writer.writerows(flagged_rows)
        run.record_output(out_path)
        logger.info(f"flagged {len(flagged_rows)} (record, term) pairs for review -> {out_path}")


if __name__ == "__main__":
    main()
