#!/usr/bin/env python
"""Pull a smaller, finishable post-editing batch: a fixed number of records
per topic (not proportional to how much source content each topic has), so
every disease/health area in the taxonomy gets real representation instead
of the sample being dominated by whichever topics happen to have the most
source pages ingested.

This exists because 1,441 rows is more than one person can realistically
post-edit and validate before a typical deadline -- the build spec's own
suggested order of work (section 14) is to prove the full pipeline on a
~200-pair seed set first, then scale up later, rather than blocking on the
full dataset. This script produces that seed set as its own spreadsheet,
using the exact same columns as 04_export_for_postedit.py's output, so the
existing 04b_import_postedit.py works on it unchanged.

Usage:
    python scripts/sample_for_review.py --in data/interim/03_translate.jsonl \
        --out data/interim/postedit_sample_200.xlsx --per-topic 16
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import read_jsonl
from src.utils.logging import MissingInputError, get_logger
from src.utils.seeding import seed_everything

logger = get_logger("sample_for_review")

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
    ws.title = "postedit_sample"
    ws.append(POSTEDIT_COLUMNS)
    for row in rows:
        ws.append([row[c] for c in POSTEDIT_COLUMNS])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


def sample_fixed_per_topic(records: list[dict], per_topic: int, seed: int) -> list[dict]:
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_topic[r.get("topic", "unclassified")].append(r)

    rng = random.Random(seed)
    sample = []
    for topic in sorted(by_topic):
        topic_records = by_topic[topic]
        k = min(per_topic, len(topic_records))
        sample.extend(rng.sample(topic_records, k))
        if k < per_topic:
            logger.warning(f"topic '{topic}' only has {len(topic_records)} eligible records "
                            f"(wanted {per_topic}) -- took all of them")
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input", default="data/interim/03_translate.jsonl")
    parser.add_argument("--out", default="data/interim/postedit_sample_200.xlsx")
    parser.add_argument("--per-topic", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise MissingInputError(str(in_path), stage="sample_for_review",
                                 hint="run scripts/03_translate.py first")

    seed = seed_everything(args.seed)
    records = list(read_jsonl(in_path))
    eligible = [r for r in records if r.get("question_yo_mt") or r.get("answer_yo_mt")]
    skipped = len(records) - len(eligible)
    if skipped:
        logger.warning(f"skipped {skipped} records with no MT output yet")

    sample = sample_fixed_per_topic(eligible, args.per_topic, seed)
    n_topics = len({r.get("topic", "unclassified") for r in sample})

    rows = build_rows(sample)
    out_path = Path(args.out)
    write_xlsx(rows, out_path)
    logger.info(f"wrote {len(rows)} rows across {n_topics} topics ({args.per_topic}/topic target) -> {out_path}")


if __name__ == "__main__":
    main()
