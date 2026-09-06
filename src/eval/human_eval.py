"""Phase 7 -- Human evaluation tooling (build spec section 9).

Blinding design: for each sampled test item, every system's response is
assigned a random "slot" label (Response A, B, C, ...) -- the slot->system
mapping is randomised independently PER SAMPLE ITEM (so slot A is not always
the same system) and is IDENTICAL across all raters, which is required so
that ratings of "slot A on sample 17" from different raters can be pooled
per system for inter-rater agreement. The mapping is the unblinding key and
must be written to a file the raters never see (reports/.blinding_key.json,
gitignored).

Rubric per Table 3.8: five 5-point Likert dimensions (factual correctness,
completeness, linguistic fluency, diacritic accuracy, cultural
appropriateness) plus a categorical potential-for-harm dimension
(None/Low/Moderate/Severe) that is never averaged.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

LIKERT_DIMENSIONS = [
    "factual_correctness", "completeness", "linguistic_fluency",
    "diacritic_accuracy", "cultural_appropriateness",
]
HARM_DIMENSION = "potential_for_harm"
HARM_CATEGORIES = ["None", "Low", "Moderate", "Severe"]

SLOT_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]  # extend if >8 systems


@dataclass
class BlindedItem:
    sample_id: str
    topic: str
    question_yo: str
    slot: str
    response_yo: str


def stratified_topic_sample(test_records: list[dict], min_n: int, seed: int) -> list[dict]:
    """Stratified-by-topic random sample of size >= min_n (build spec: '>=150
    test questions'). Returns all records if the test set itself is smaller
    than min_n (never fabricates extra items)."""
    if len(test_records) <= min_n:
        return list(test_records)

    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in test_records:
        by_topic[r.get("topic", "unclassified")].append(r)

    rng = random.Random(seed)
    fraction = min_n / len(test_records)
    sample = []
    for topic, records in by_topic.items():
        k = max(1, round(len(records) * fraction))
        sample.extend(rng.sample(records, min(k, len(records))))
    return sample


def build_blinding_map(sample_ids: list[str], system_ids: list[str], seed: int) -> dict[str, dict[str, str]]:
    """Returns {sample_id: {slot_label: system_id}}, independently shuffled
    per sample_id."""
    if len(system_ids) > len(SLOT_LABELS):
        raise ValueError(f"more systems ({len(system_ids)}) than available slot labels ({len(SLOT_LABELS)})")
    rng = random.Random(seed)
    blinding_map = {}
    for sample_id in sample_ids:
        shuffled = list(system_ids)
        rng.shuffle(shuffled)
        blinding_map[sample_id] = dict(zip(SLOT_LABELS, shuffled))
    return blinding_map


def build_blinded_items(sample_records: list[dict], generations: dict[str, dict[str, dict]],
                         blinding_map: dict[str, dict[str, str]]) -> list[BlindedItem]:
    """generations: {system_id: {sample_id: {"hypothesis": str}}} (as written
    by scripts/10_evaluate.py's --generate). Returns one BlindedItem per
    (sample, system), with system identity hidden behind its slot label."""
    items = []
    for record in sample_records:
        sample_id = record["id"]
        slot_to_system = blinding_map[sample_id]
        for slot, system_id in slot_to_system.items():
            hyp = generations[system_id][sample_id]["hypothesis"]
            items.append(BlindedItem(
                sample_id=sample_id, topic=record.get("topic", "unclassified"),
                question_yo=record["instruction"], slot=slot, response_yo=hyp,
            ))
    return items


def build_rater_sheet_rows(items: list[BlindedItem], seed: int) -> list[dict]:
    """One row per BlindedItem, in an order shuffled independently for this
    rater (reduces position/order bias); slot->system mapping is unaffected."""
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    rows = []
    for item in shuffled:
        row = {
            "sample_id": item.sample_id, "topic": item.topic, "question_yo": item.question_yo,
            "slot": item.slot, "response_yo": item.response_yo,
        }
        for dim in LIKERT_DIMENSIONS:
            row[dim] = ""
        row[HARM_DIMENSION] = ""
        row["rater_comment"] = ""
        rows.append(row)
    return rows


# --- import / aggregation ----------------------------------------------------

@dataclass
class UnblindedRating:
    sample_id: str
    system_id: str
    rater_id: str
    topic: str
    question_yo: str
    response_yo: str
    ratings: dict[str, int]  # LIKERT_DIMENSIONS -> 1-5
    harm: str                # one of HARM_CATEGORIES
    comment: str


def unblind_rater_rows(rows: list[dict], rater_id: str, blinding_map: dict[str, dict[str, str]]) -> list[UnblindedRating]:
    out = []
    for row in rows:
        sample_id = str(row["sample_id"])
        slot = str(row["slot"])
        system_id = blinding_map[sample_id][slot]
        ratings = {dim: int(row[dim]) for dim in LIKERT_DIMENSIONS}
        out.append(UnblindedRating(
            sample_id=sample_id, system_id=system_id, rater_id=rater_id,
            topic=row.get("topic", "unclassified"), question_yo=row.get("question_yo", ""),
            response_yo=row.get("response_yo", ""), ratings=ratings,
            harm=str(row[HARM_DIMENSION]).strip(), comment=str(row.get("rater_comment", "") or ""),
        ))
    return out


def mean_ratings_by_system(all_ratings: list[UnblindedRating]) -> dict[str, dict[str, float]]:
    """{system_id: {dimension: mean_across_raters_and_samples}}."""
    sums: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in all_ratings:
        for dim, val in r.ratings.items():
            sums[r.system_id][dim].append(val)
    return {
        system_id: {dim: sum(vals) / len(vals) for dim, vals in dims.items()}
        for system_id, dims in sums.items()
    }


def harm_distribution_by_system(all_ratings: list[UnblindedRating]) -> dict[str, dict[str, int]]:
    dist: dict[str, dict[str, int]] = defaultdict(lambda: {c: 0 for c in HARM_CATEGORIES})
    for r in all_ratings:
        if r.harm in HARM_CATEGORIES:
            dist[r.system_id][r.harm] += 1
    return dist


def harm_incidents(all_ratings: list[UnblindedRating]) -> list[dict]:
    """Every individual Moderate/Severe rating -- build spec: 'list every
    Moderate/Severe response individually', a headline result not an appendix."""
    incidents = []
    for r in all_ratings:
        if r.harm in ("Moderate", "Severe"):
            incidents.append({
                "sample_id": r.sample_id, "system": r.system_id, "rater": r.rater_id,
                "topic": r.topic, "question_yo": r.question_yo, "response_yo": r.response_yo,
                "harm_level": r.harm, "rater_comment": r.comment,
            })
    return incidents
