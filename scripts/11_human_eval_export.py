#!/usr/bin/env python
"""Phase 7 -- Export blinded human evaluation rating sheets (build spec
section 9).

Draws a stratified-by-topic random sample of >= configs/eval.yaml
human_eval.min_sample_size test questions, blinds every system's response
behind a randomised per-sample slot label, and writes one workbook per rater
(--n-raters) plus the unblinding key (reports/.blinding_key.json --
gitignored, never share this with raters).

Requires reports/generations/<SYSTEM_ID>.jsonl for every system in
configs/eval.yaml (i.e. scripts/10_evaluate.py --generate must have already
run for all of them).

Usage:
    python scripts/11_human_eval_export.py --test-jsonl data/final/test.jsonl --n-raters 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import read_jsonl
from src.eval.human_eval import (
    HARM_DIMENSION,
    LIKERT_DIMENSIONS,
    build_blinded_items,
    build_blinding_map,
    build_rater_sheet_rows,
    stratified_topic_sample,
)
from src.utils.logging import MissingInputError, get_logger
from src.utils.manifest import RunManifest
from src.utils.seeding import seed_everything

logger = get_logger("11_human_eval_export")

SHEET_COLUMNS = (["sample_id", "topic", "question_yo", "slot", "response_yo"]
                  + LIKERT_DIMENSIONS + [HARM_DIMENSION, "rater_comment"])


def write_rater_xlsx(rows: list[dict], out_path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "human_eval"
    ws.append(SHEET_COLUMNS)
    for row in rows:
        ws.append([row[c] for c in SHEET_COLUMNS])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-jsonl", default="data/final/test.jsonl")
    parser.add_argument("--generations-dir", default="reports/generations")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--out-dir", default="reports/human_eval")
    parser.add_argument("--blinding-key", default="reports/.blinding_key.json")
    parser.add_argument("--n-raters", type=int, required=True)
    args = parser.parse_args()

    import yaml

    test_path = Path(args.test_jsonl)
    if not test_path.exists():
        raise MissingInputError(str(test_path), stage="11_human_eval_export")

    eval_config_path = Path(args.eval_config)
    if not eval_config_path.exists():
        raise MissingInputError(str(eval_config_path), stage="11_human_eval_export")
    with open(eval_config_path, encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    system_ids = list(eval_cfg["systems"])
    generations = {}
    for system_id in system_ids:
        gen_path = Path(args.generations_dir) / f"{system_id}.jsonl"
        if not gen_path.exists():
            raise MissingInputError(str(gen_path), stage="11_human_eval_export",
                                     hint=f"run scripts/10_evaluate.py --generate {system_id} first")
        generations[system_id] = {r["id"]: r for r in read_jsonl(gen_path)}

    seed = seed_everything(eval_cfg.get("seed", 42))
    test_records = list(read_jsonl(test_path))
    min_n = eval_cfg["human_eval"]["min_sample_size"]
    sample = stratified_topic_sample(test_records, min_n=min_n, seed=seed)
    logger.info(f"sampled {len(sample)} test items (min_sample_size={min_n})")

    sample_ids = [r["id"] for r in sample]
    blinding_map = build_blinding_map(sample_ids, system_ids, seed=seed)
    items = build_blinded_items(sample, generations, blinding_map)
    logger.info(f"built {len(items)} blinded (sample, system) items across {len(system_ids)} systems")

    out_dir = Path(args.out_dir)
    with RunManifest(stage="11_human_eval_export", config={"n_raters": args.n_raters, "n_sample": len(sample)},
                      seed=seed) as run:
        for rater_i in range(args.n_raters):
            rows = build_rater_sheet_rows(items, seed=seed + rater_i + 1)
            out_path = out_dir / f"rater_{rater_i + 1}.xlsx"
            write_rater_xlsx(rows, out_path)
            run.record_output(str(out_path))
            logger.info(f"wrote {len(rows)} rows -> {out_path}")

        blinding_key_path = Path(args.blinding_key)
        blinding_key_path.parent.mkdir(parents=True, exist_ok=True)
        with open(blinding_key_path, "w", encoding="utf-8") as f:
            json.dump(blinding_map, f, ensure_ascii=False, indent=2)
        logger.info(f"wrote unblinding key -> {blinding_key_path} "
                    f"(KEEP OUT OF RATERS' HANDS -- this file must stay gitignored)")


if __name__ == "__main__":
    main()
