#!/usr/bin/env python
"""Phase 1.3 -- Machine-translate question_en/answer_en to Yoruba.

Reads a working JSONL (default data/interim/02_build_qa.jsonl), fills
question_yo_mt/answer_yo_mt via an NLLB-derived model, content-hash-cached so
an unchanged string is never re-translated across runs. post_edited stays
false -- this is machine output, not yet reviewed by a human.

Requires torch + transformers (see requirements.txt); CPU works for pilot
batches but is slow -- a GPU is recommended once scaling past ~200 pairs.

Usage:
    python scripts/03_translate.py --in data/interim/02_build_qa.jsonl \
        --out data/interim/03_translate.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import read_jsonl, write_jsonl
from src.data.yoruba_text import normalize_nfc
from src.model.translate import NllbTranslator, TranslationCache, translate_with_cache
from src.utils.logging import MissingInputError, get_logger
from src.utils.manifest import RunManifest
from src.utils.seeding import seed_everything

logger = get_logger("03_translate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input", default="data/interim/02_build_qa.jsonl")
    parser.add_argument("--out", default="data/interim/03_translate.jsonl")
    parser.add_argument("--cache", default="data/interim/.translation_cache.json")
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise MissingInputError(str(in_path), stage="03_translate",
                                 hint="run scripts/02_build_qa.py (and/or 02b_import_elicited.py) first")

    seed = seed_everything(args.seed)
    records = list(read_jsonl(in_path))
    if not records:
        raise MissingInputError(str(in_path), stage="03_translate", hint="input file is empty")

    with RunManifest(stage="03_translate", config={"model": args.model, "input": str(in_path)}, seed=seed) as run:
        cache = TranslationCache(args.cache)
        logger.info(f"loading translator {args.model} (cache has {len(cache)} entries already)")
        translator = NllbTranslator(model_name=args.model)

        questions = [r.get("question_en") or "" for r in records]
        answers = [r.get("answer_en") or "" for r in records]

        logger.info(f"translating {sum(1 for q in questions if q)} questions...")
        translated_questions = translate_with_cache(questions, translator, cache)
        logger.info(f"translating {sum(1 for a in answers if a)} answers...")
        translated_answers = translate_with_cache(answers, translator, cache)
        cache.save()

        for record, tq, ta in zip(records, translated_questions, translated_answers):
            if tq:
                record["question_yo_mt"] = normalize_nfc(tq)
            if ta:
                record["answer_yo_mt"] = normalize_nfc(ta)
            record["post_edited"] = False

        n = write_jsonl(records, args.out)
        run.record_output(args.out)
        run.record_output(args.cache)
        logger.info(f"wrote {n} records with MT fields to {args.out} (cache now has {len(cache)} entries)")


if __name__ == "__main__":
    main()
