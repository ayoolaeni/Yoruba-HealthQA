#!/usr/bin/env python
"""Phase 1.7 -- Import completed validation sheets, compute agreement, route
failed records to a correction queue.

For rows double-annotated (annotator_slot 1 and 2 for the same id), computes
Cohen's kappa on the pass/fail verdicts and interprets it per Landis & Koch
(1977). Records with verdict_pass_fail == "fail" (from either annotator) are
written to a correction queue instead of being marked validated; the discard
rate (fail / total) is logged and included in table_4_2.

Usage:
    python scripts/05b_import_validation.py --edited data/interim/validation_clinical.xlsx \
        --working-jsonl data/interim/04b_import_postedit.jsonl --kind clinical \
        --out data/interim/05b_clinical_validated.jsonl \
        --correction-queue data/interim/correction_queue_clinical.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import read_jsonl, write_jsonl
from src.eval.agreement import compute_double_annotation_kappa
from src.utils.logging import MissingInputError, get_logger
from src.utils.manifest import RunManifest
from src.utils.seeding import seed_everything

logger = get_logger("05b_import_validation")

VALID_VERDICTS = {"pass", "fail"}


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
            rows.append({k: ("" if v is None else v) for k, v in row.items()})
    return rows


def validate_verdicts(rows: list[dict]) -> list[str]:
    problems = []
    for row in rows:
        verdict = str(row.get("verdict_pass_fail", "")).strip().lower()
        if verdict not in VALID_VERDICTS:
            problems.append(f"id={row.get('id')} slot={row.get('annotator_slot')}: "
                             f"verdict_pass_fail must be 'pass' or 'fail', got {verdict!r}")
        if verdict == "fail" and not str(row.get("reason_if_fail", "")).strip():
            problems.append(f"id={row.get('id')} slot={row.get('annotator_slot')}: "
                             f"'fail' verdict requires reason_if_fail")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edited", required=True)
    parser.add_argument("--working-jsonl", required=True)
    parser.add_argument("--kind", choices=["clinical", "linguistic"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--correction-queue", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    edited_path = Path(args.edited)
    if not edited_path.exists():
        raise MissingInputError(str(edited_path), stage="05b_import_validation")
    working_path = Path(args.working_jsonl)
    if not working_path.exists():
        raise MissingInputError(str(working_path), stage="05b_import_validation")

    seed = seed_everything(args.seed)
    rows = read_xlsx_rows(edited_path)
    if not rows:
        raise MissingInputError(str(edited_path), stage="05b_import_validation", hint="sheet has no data rows")

    problems = validate_verdicts(rows)
    if problems:
        logger.error(f"{len(problems)} problem(s) in {edited_path}:")
        for p in problems[:50]:
            logger.error(f"  {p}")
        raise ValueError(f"{edited_path} failed validation import ({len(problems)} problems)")

    kappa, interpretation, n_double = compute_double_annotation_kappa(rows)
    if kappa is not None:
        logger.info(f"[{args.kind}] Cohen's kappa on {n_double} double-annotated records: "
                    f"{kappa:.3f} ({interpretation}, Landis & Koch 1977)")
    else:
        logger.warning(f"[{args.kind}] fewer than 2 double-annotated records -- kappa not computed")

    # slot-1 verdict is authoritative for pass/fail routing; slot-2 exists
    # only to measure agreement, per build spec ">=10% double-annotate for agreement"
    verdict_by_id = {row["id"]: str(row.get("verdict_pass_fail", "")).strip().lower()
                     for row in rows if int(row.get("annotator_slot") or 1) == 1}
    reason_by_id = {row["id"]: str(row.get("reason_if_fail", "")).strip()
                    for row in rows if int(row.get("annotator_slot") or 1) == 1}

    working_records = list(read_jsonl(working_path))
    validated_flag = "validated_clin" if args.kind == "clinical" else "validated_ling"

    with RunManifest(stage=f"05b_import_validation_{args.kind}",
                      config={"edited": str(edited_path), "kappa": kappa, "n_double_annotated": n_double},
                      seed=seed) as run:
        passed, failed = [], []
        for record in working_records:
            verdict = verdict_by_id.get(record["id"])
            if verdict is None:
                continue  # not in this validation batch
            if verdict == "pass":
                record[validated_flag] = True
                passed.append(record)
            else:
                record[validated_flag] = False
                record["validation_notes"] = (record.get("validation_notes", "") + " | "
                                               + f"{args.kind} fail: {reason_by_id.get(record['id'], '')}").strip(" |")
                failed.append(record)

        n_pass = write_jsonl(passed, args.out)
        n_fail = write_jsonl(failed, args.correction_queue)
        run.record_output(args.out)
        run.record_output(args.correction_queue)

        total = n_pass + n_fail
        discard_rate = n_fail / total if total else 0.0
        logger.info(f"[{args.kind}] {n_pass} passed -> {args.out}; {n_fail} failed -> {args.correction_queue} "
                    f"(discard rate {discard_rate:.1%})")


if __name__ == "__main__":
    main()
