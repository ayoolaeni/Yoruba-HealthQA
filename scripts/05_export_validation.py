#!/usr/bin/env python
"""Phase 1.7 -- Export linguistic and clinical validation sheets.

Two separate exports (build spec 1.7):
  - clinical: 100% of post-edited records, for health workers.
  - linguistic: 100% if capacity allows, else a stratified-by-topic random
    sample of >= configs/data.yaml validation.linguistic_coverage_floor,
    for native speakers.

Both exports double-annotate >= validation.double_annotation_fraction of
their rows (a second copy of those rows, with a distinct annotator_slot
column, `annotator_slot=2`) so 05b_import_validation.py can compute Cohen's
kappa on the overlap.

Only records with post_edited answer text (question_yo_final/answer_yo_final)
are eligible -- there is nothing to validate before post-editing has run.

Usage:
    python scripts/05_export_validation.py --in data/interim/04b_import_postedit.jsonl \
        --kind clinical --out data/interim/validation_clinical.xlsx
    python scripts/05_export_validation.py --in data/interim/04b_import_postedit.jsonl \
        --kind linguistic --out data/interim/validation_linguistic.xlsx --coverage 1.0
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import read_jsonl
from src.utils.logging import MissingInputError, get_logger
from src.utils.manifest import RunManifest
from src.utils.seeding import seed_everything

logger = get_logger("05_export_validation")

VALIDATION_COLUMNS = [
    "id", "annotator_slot", "topic", "question_yo_final", "answer_yo_final",
    "source_en_reference", "verdict_pass_fail", "reason_if_fail", "annotator_id",
]


def eligible_records(records: list[dict]) -> list[dict]:
    return [r for r in records if (r.get("question_yo_final") or "").strip() and (r.get("answer_yo_final") or "").strip()]


def stratified_sample(records: list[dict], fraction: float, rng: random.Random) -> list[dict]:
    if fraction >= 1.0:
        return list(records)
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_topic[r.get("topic", "unclassified")].append(r)
    sampled = []
    for topic, topic_records in by_topic.items():
        k = max(1, round(len(topic_records) * fraction))
        sampled.extend(rng.sample(topic_records, min(k, len(topic_records))))
    return sampled


def build_rows(records: list[dict], double_annotate_fraction: float, rng: random.Random) -> list[dict]:
    rows = []
    for r in records:
        rows.append(_row(r, annotator_slot=1))
    double_n = round(len(records) * double_annotate_fraction)
    for r in rng.sample(records, min(double_n, len(records))):
        rows.append(_row(r, annotator_slot=2))
    rng.shuffle(rows)
    return rows


def _row(r: dict, annotator_slot: int) -> dict:
    return {
        "id": r["id"],
        "annotator_slot": annotator_slot,
        "topic": r.get("topic", ""),
        "question_yo_final": r.get("question_yo_final", ""),
        "answer_yo_final": r.get("answer_yo_final", ""),
        "source_en_reference": r.get("answer_en", ""),
        "verdict_pass_fail": "",   # annotator fills: pass | fail
        "reason_if_fail": "",
        "annotator_id": "",
    }


def write_xlsx(rows: list[dict], out_path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "validation"
    ws.append(VALIDATION_COLUMNS)
    for row in rows:
        ws.append([row[c] for c in VALIDATION_COLUMNS])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input", default="data/interim/04b_import_postedit.jsonl")
    parser.add_argument("--kind", choices=["clinical", "linguistic"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--coverage", type=float, default=None,
                         help="override configs/data.yaml validation coverage for this kind")
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise MissingInputError(str(in_path), stage="05_export_validation",
                                 hint="run scripts/04b_import_postedit.py first")

    with open(args.data_config, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    seed = seed_everything(data_cfg.get("seed", 42))
    val_cfg = data_cfg["validation"]

    if args.kind == "clinical":
        coverage = args.coverage if args.coverage is not None else val_cfg["clinical_coverage"]
    else:
        coverage = args.coverage if args.coverage is not None else val_cfg["linguistic_coverage_target"]

    records = eligible_records(list(read_jsonl(in_path)))
    if not records:
        raise MissingInputError(str(in_path), stage="05_export_validation",
                                 hint="no records have both question_yo_final and answer_yo_final filled in yet")

    rng = random.Random(seed)
    out_path = Path(args.out)
    with RunManifest(stage=f"05_export_validation_{args.kind}", config={"coverage": coverage}, seed=seed) as run:
        sample = stratified_sample(records, coverage, rng)
        rows = build_rows(sample, val_cfg["double_annotation_fraction"], rng)
        write_xlsx(rows, out_path)
        run.record_output(str(out_path))
        logger.info(f"[{args.kind}] exported {len(sample)}/{len(records)} eligible records "
                    f"({coverage:.0%} coverage) as {len(rows)} rows (incl. double-annotation) -> {out_path}")


if __name__ == "__main__":
    main()
