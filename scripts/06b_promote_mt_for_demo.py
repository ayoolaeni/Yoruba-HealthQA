#!/usr/bin/env python
"""DEMO-ONLY utility -- NOT a formal build-spec pipeline stage.

Promotes raw, UN-REVIEWED machine-translation output (question_yo_mt /
answer_yo_mt) into the question_yo_final / answer_yo_final fields for any
record that a human has not post-edited yet. This exists only to get a
bigger (but lower-quality) training set for a same-day demo fine-tune,
after the researcher was told explicitly -- and confirmed, understanding
the risk -- that these ~1,233 extra records have had ZERO human review of
any kind (unlike the 208 records a real Yoruba-speaking reviewer already
checked). Training on this promoted text risks teaching the model
translation mistakes we have already found examples of (e.g. "breastfeeding"
mistranslated as "pregnancy") more confidently, not just more fluently.

The output of this script must NEVER be treated as the dissertation's
validated dataset, and must never feed src/serve/retrieval.py (which is
deliberately restricted to data/final_demo/, the human-reviewed-only set,
so the live demo never shows unreviewed text). It exists purely as input to
scripts/06_finalise_dataset.py --include-unvalidated-for-demo-only, for a
throwaway training-data-quantity experiment.

Usage:
    python scripts/06b_promote_mt_for_demo.py \
        --in data/interim/04b_import_postedit.jsonl \
        --out data/interim/06b_demo_expanded_with_mt.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import read_jsonl, write_jsonl
from src.utils.logging import MissingInputError, get_logger

logger = get_logger("06b_promote_mt_for_demo")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input", default="data/interim/04b_import_postedit.jsonl")
    parser.add_argument("--out", default="data/interim/06b_demo_expanded_with_mt.jsonl")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise MissingInputError(str(in_path), stage="06b_promote_mt_for_demo")

    records = list(read_jsonl(in_path))
    n_already_reviewed = 0
    n_promoted_from_mt = 0
    n_skipped_no_text = 0
    out_records = []
    for r in records:
        has_final = (r.get("question_yo_final") or "").strip() and (r.get("answer_yo_final") or "").strip()
        if has_final:
            n_already_reviewed += 1
            out_records.append(r)
            continue

        has_mt = (r.get("question_yo_mt") or "").strip() and (r.get("answer_yo_mt") or "").strip()
        if not has_mt:
            n_skipped_no_text += 1
            continue

        r = dict(r)
        r["question_yo_final"] = r["question_yo_mt"]
        r["answer_yo_final"] = r["answer_yo_mt"]
        r["promoted_from_mt_only"] = True  # traceability: this text was NEVER human-reviewed
        n_promoted_from_mt += 1
        out_records.append(r)

    logger.warning("=" * 70)
    logger.warning("DEMO-ONLY: promoting UN-REVIEWED machine translation to 'final' fields.")
    logger.warning(f"  {n_already_reviewed} records already had real human-reviewed text (kept as-is).")
    logger.warning(f"  {n_promoted_from_mt} records had NO human review -- raw MT text promoted anyway.")
    logger.warning(f"  {n_skipped_no_text} records skipped (no MT text available at all).")
    logger.warning("This output must NEVER be reported as the dissertation's validated dataset,")
    logger.warning("and must never be wired into src/serve/retrieval.py's data sources.")
    logger.warning("=" * 70)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_records, out_path)
    logger.info(f"wrote {len(out_records)} records to {out_path}")


if __name__ == "__main__":
    main()
