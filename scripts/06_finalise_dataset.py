#!/usr/bin/env python
"""Phase 1.8 -- Filter, safety-rewrite, split, and assemble the final dataset.

Pipeline:
  1. Load validated working records (validated_ling AND validated_clin both
     True -- only fully-validated records reach the final dataset).
  2. Near-duplicate removal (char n-gram Jaccard over answer_yo_final).
  3. Safety filter: dosage/prescription/emergency-procedure content is
     rewritten as a refusal example rather than dropped.
  4. Assemble final Table 3.5 records (src/data/schema.py REQUIRED_FIELDS).
  5. Split 80/10/10, stratified by topic, GROUPED by claim_id. Asserts zero
     claim_id overlap across splits (build spec rule 3) -- the build fails
     loudly if this is ever violated.
  6. Emits table_4_1_dataset_composition and table_4_2_validation_outcomes
     (CSV + Markdown) to reports/, and data/final/{train,val,test}.jsonl.

Usage:
    python scripts/06_finalise_dataset.py --in data/interim/05b_all_validated.jsonl
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dedup_and_split import (
    compile_safety_patterns,
    dedup_records,
    matches_safety_filter,
    rewrite_as_refusal,
    split_by_claim_group,
)
from src.data.schema import validate_record, write_jsonl
from src.data.schema import read_jsonl
from src.utils.logging import MissingInputError, get_logger
from src.utils.manifest import RunManifest
from src.utils.seeding import seed_everything

logger = get_logger("06_finalise_dataset")


def to_final_schema(record: dict) -> dict | None:
    """Maps a fully-validated WorkingRecord dict to the final Table 3.5
    schema. Returns None (caller skips) if required Yoruba fields are empty."""
    instruction = record.get("question_yo_final") or ""
    output = record.get("answer_yo_final") or ""
    if not instruction.strip() or not output.strip():
        return None
    from src.data.yoruba_text import strip_diacritics

    return {
        "id": record["id"],
        "claim_id": record["claim_id"],
        "instruction": instruction,
        "instruction_nodia": strip_diacritics(instruction),
        "input": "",
        "output": output,
        "topic": record.get("topic", "unclassified"),
        "question_type": record.get("question_type", "derived"),
        "source_en": record.get("answer_en", ""),
        "source_ref": record.get("source_ref", ""),
        "validated_ling": bool(record.get("validated_ling")),
        "validated_clin": bool(record.get("validated_clin")),
    }


def write_table(df: pd.DataFrame, name: str, reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / f"{name}.csv"
    md_path = reports_dir / f"{name}.md"
    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input", required=True,
                         help="JSONL of working records after BOTH clinical and linguistic validation")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--out-dir", default="data/final")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--require-both-validations", action="store_true", default=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise MissingInputError(str(in_path), stage="06_finalise_dataset",
                                 hint="run 05b_import_validation.py for both --kind clinical and --kind "
                                      "linguistic first, then merge their passing outputs into one file")

    with open(args.data_config, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    seed = seed_everything(data_cfg.get("seed", 42))

    all_records = list(read_jsonl(in_path))
    if not all_records:
        raise MissingInputError(str(in_path), stage="06_finalise_dataset", hint="input file is empty")

    with RunManifest(stage="06_finalise_dataset", config={"input": str(in_path)}, seed=seed) as run:
        n_total = len(all_records)
        validated = [r for r in all_records if r.get("validated_clin") and r.get("validated_ling")]
        n_not_validated = n_total - len(validated)
        logger.info(f"{len(validated)}/{n_total} records passed both clinical and linguistic validation")

        # --- near-duplicate removal ---
        dedup_cfg = data_cfg["dedup"]
        deduped, n_dup_removed = dedup_records(
            validated, "answer_yo_final", dedup_cfg["ngram_size"], dedup_cfg["similarity_threshold"])
        logger.info(f"removed {n_dup_removed} near-duplicate records "
                    f"(ngram={dedup_cfg['ngram_size']}, threshold={dedup_cfg['similarity_threshold']})")

        # --- safety filter ---
        patterns = compile_safety_patterns(data_cfg["safety_filter_terms"])
        n_rewritten = 0
        safety_checked = []
        for r in deduped:
            if matches_safety_filter(r.get("answer_yo_final", "") + " " + r.get("answer_en", ""), patterns):
                r = rewrite_as_refusal(r)
                n_rewritten += 1
            safety_checked.append(r)
        logger.info(f"rewrote {n_rewritten} records as refusal examples (safety filter)")

        # --- assemble final schema ---
        final_records = []
        n_empty_skipped = 0
        for r in safety_checked:
            final = to_final_schema(r)
            if final is None:
                n_empty_skipped += 1
                continue
            final_records.append(final)
        if n_empty_skipped:
            logger.warning(f"skipped {n_empty_skipped} records with empty instruction/output after safety rewrite")

        valid_topics = set(data_cfg["topics"]) | {"unclassified"}
        schema_problems = []
        for r in final_records:
            problems = validate_record(r, valid_topics=valid_topics)
            if problems:
                schema_problems.append((r["id"], problems))
        if schema_problems:
            for rid, problems in schema_problems[:20]:
                logger.error(f"schema validation failed for {rid}: {problems}")
            raise ValueError(f"{len(schema_problems)} record(s) failed final schema validation -- see log above")

        # --- split ---
        split_cfg = data_cfg["split"]
        splits = split_by_claim_group(
            final_records, train=split_cfg["train"], val=split_cfg["val"], test=split_cfg["test"],
            stratify_field=split_cfg["stratify_by"], group_field=split_cfg["group_by"], seed=seed,
        )

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for split_name, records in splits.items():
            path = out_dir / f"{split_name}.jsonl"
            write_jsonl(records, path)
            run.record_output(str(path))
            logger.info(f"wrote {len(records)} records to {path}")

        # --- table_4_1: dataset composition ---
        rows = []
        for split_name, records in splits.items():
            counts = Counter((r["topic"], r["question_type"]) for r in records)
            for (topic, qtype), n in sorted(counts.items()):
                rows.append({"split": split_name, "topic": topic, "question_type": qtype, "n_records": n})
        composition_df = pd.DataFrame(rows)
        reports_dir = Path(args.reports_dir)
        csv_p, md_p = write_table(composition_df, "table_4_1_dataset_composition", reports_dir)
        run.record_output(str(csv_p))
        run.record_output(str(md_p))

        # --- table_4_2: validation outcomes ---
        outcomes_df = pd.DataFrame([{
            "stage": "ingested_working_records", "count": n_total,
        }, {
            "stage": "passed_both_validations", "count": len(validated),
        }, {
            "stage": "failed_or_pending_validation", "count": n_not_validated,
        }, {
            "stage": "near_duplicates_removed", "count": n_dup_removed,
        }, {
            "stage": "rewritten_as_refusal_safety_filter", "count": n_rewritten,
        }, {
            "stage": "final_dataset_total", "count": len(final_records),
        }])
        csv_p, md_p = write_table(outcomes_df, "table_4_2_validation_outcomes", reports_dir)
        run.record_output(str(csv_p))
        run.record_output(str(md_p))

        logger.info(f"final dataset: {len(final_records)} records "
                    f"(train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])})")


if __name__ == "__main__":
    main()
