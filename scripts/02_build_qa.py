#!/usr/bin/env python
"""Phase 1.2 -- Segment ingested documents into atomic claims and generate
consumer-style English questions per claim.

Reads data/raw/<source>/*.json (from 01_ingest.py). Writes
data/interim/02_build_qa.jsonl of WorkingRecord dicts (question_type="derived").

Usage:
    python scripts/02_build_qa.py --raw-dir data/raw --out data/interim/02_build_qa.jsonl
    python scripts/02_build_qa.py --question-backend llm   # requires ANTHROPIC_API_KEY
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.claim_extraction import generate_question, split_into_claims
from src.data.schema import write_jsonl
from src.data.working_record import WorkingRecord, make_claim_id, make_id
from src.utils.logging import MissingInputError, get_logger
from src.utils.manifest import RunManifest
from src.utils.seeding import seed_everything

logger = get_logger("02_build_qa")

# Maps a source document's filename stem (without extension) to a topic in
# configs/data.yaml's fixed topic list. Extend this as new sources/slugs are
# added to configs/data.yaml -- a slug with no mapping is skipped with a
# warning rather than guessed at.
SLUG_TO_TOPIC = {
    "malaria": "malaria",
    "typhoid": "typhoid",
    "tuberculosis": "tuberculosis",
    "hiv-aids": "hiv",
    "hivaids.html": "hiv",
    "hypertension": "hypertension",
    "highbloodpressure.html": "hypertension",
    "diabetes": "diabetes",
    "diabetes.html": "diabetes",
    "malnutrition": "nutrition",
    "malaria.html": "malaria",
    "tuberculosis.html": "tuberculosis",
    "maternal-mortality": "maternal_health",
    "children-reducing-mortality": "child_health",
    "pneumonia": "child_health",
    "diarrhoeal-disease": "child_health",
    "immunization-coverage": "immunisation",
    "measles": "immunisation",
    "obesity-and-overweight": "nutrition",
    "anaemia": "nutrition",
    "drinking-water": "wash",
    "sanitation": "wash",
    "hygiene": "wash",
    "cholera": "wash",
    "mental-disorders": "mental_health",
    "depression": "mental_health",
    "coronavirus-disease-(covid-19)": "covid19",
}

TOPIC_DISPLAY_NAMES = {
    "malaria": "malaria",
    "typhoid": "typhoid",
    "tuberculosis": "tuberculosis (TB)",
    "hiv": "HIV",
    "hypertension": "high blood pressure",
    "diabetes": "diabetes",
    "maternal_health": "maternal health",
    "child_health": "child health",
    "immunisation": "immunisation",
    "nutrition": "nutrition",
    "wash": "water, sanitation, and hygiene",
    "mental_health": "mental health",
    "covid19": "COVID-19",
}


def infer_topic(slug: str) -> str | None:
    return SLUG_TO_TOPIC.get(slug)


def process_document(doc_path: Path, doc: dict, valid_topics: set[str], backend: str,
                      next_id: list[int]) -> tuple[list[WorkingRecord], int]:
    slug = doc_path.stem
    topic = infer_topic(slug)
    if topic is None:
        logger.warning(f"[{doc_path}] no topic mapping for slug '{slug}' -- skipping document. "
                        f"Add it to SLUG_TO_TOPIC in scripts/02_build_qa.py.")
        return [], 0
    if topic not in valid_topics:
        logger.error(f"[{doc_path}] mapped topic '{topic}' is not in configs/data.yaml topics -- skipping")
        return [], 0

    claims = split_into_claims(doc["raw_text"])
    topic_display = TOPIC_DISPLAY_NAMES.get(topic, topic)
    records = []
    llm_fallback_count = 0
    for i, claim in enumerate(claims):
        question, backend_used = generate_question(claim, topic_display, backend=backend)
        if backend_used == "template_fallback":
            llm_fallback_count += 1
        record = WorkingRecord(
            id=make_id(next_id[0]),
            claim_id=make_claim_id(doc["source_name"], slug, i),
            topic=topic,
            question_type="derived",
            question_en=question,
            answer_en=claim,
            source_ref=f"{doc['source_name']} ({slug}), retrieved {doc['retrieval_date']}, licence: {doc['licence']}",
            source_name=doc["source_name"],
            source_url=doc.get("url"),
        )
        next_id[0] += 1
        records.append(record)
    return records, llm_fallback_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--out", default="data/interim/02_build_qa.jsonl")
    parser.add_argument("--question-backend", choices=["template", "llm"], default="template")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise MissingInputError(str(raw_dir), stage="02_build_qa",
                                 hint="run scripts/01_ingest.py first")

    data_config_path = Path(args.data_config)
    if not data_config_path.exists():
        raise MissingInputError(str(data_config_path), stage="02_build_qa")
    with open(data_config_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    valid_topics = set(data_cfg["topics"])
    seed = seed_everything(data_cfg.get("seed", 42))

    doc_paths = sorted(raw_dir.glob("*/*.json"))
    if not doc_paths:
        raise MissingInputError(str(raw_dir), stage="02_build_qa",
                                 hint=f"no *.json documents found under {raw_dir} -- run 01_ingest.py first")

    with RunManifest(stage="02_build_qa", config={"question_backend": args.question_backend}, seed=seed) as run:
        all_records: list[WorkingRecord] = []
        next_id = [1]
        total_llm_fallbacks = 0
        for doc_path in doc_paths:
            with open(doc_path, encoding="utf-8") as f:
                doc = json.load(f)
            records, fallbacks = process_document(doc_path, doc, valid_topics, args.question_backend, next_id)
            all_records.extend(records)
            total_llm_fallbacks += fallbacks

        n = write_jsonl((r.to_dict() for r in all_records), args.out)
        run.record_output(args.out)
        logger.info(f"wrote {n} candidate QA records to {args.out}")
        if args.question_backend == "llm" and total_llm_fallbacks:
            logger.warning(f"{total_llm_fallbacks} records fell back to the template backend "
                            f"after an LLM call failed")


if __name__ == "__main__":
    main()
