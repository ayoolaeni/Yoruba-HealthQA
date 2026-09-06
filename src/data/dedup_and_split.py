"""Near-duplicate removal, safety filtering, and claim_id-grouped stratified
splitting for 06_finalise_dataset.py (build spec 1.8).
"""
from __future__ import annotations

import random
import re
from collections import defaultdict

# --- near-duplicate removal ---------------------------------------------------

def char_ngrams(text: str, n: int) -> set[str]:
    bare = re.sub(r"\s+", " ", text.strip().lower())
    if len(bare) < n:
        return {bare} if bare else set()
    return {bare[i:i + n] for i in range(len(bare) - n + 1)}


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_near_duplicates(texts: list[str], ngram_size: int, threshold: float) -> list[tuple[int, int]]:
    """Returns (i, j) index pairs with i < j whose char n-gram Jaccard
    similarity is >= threshold. O(n^2) -- fine at the pilot/dissertation scale
    (1,500-3,000 records) the build spec targets; would need blocking/LSH at
    much larger scale.
    """
    ngram_sets = [char_ngrams(t, ngram_size) for t in texts]
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if jaccard_similarity(ngram_sets[i], ngram_sets[j]) >= threshold:
                pairs.append((i, j))
    return pairs


def dedup_records(records: list[dict], text_field: str, ngram_size: int, threshold: float) -> tuple[list[dict], int]:
    """Drops the later record of each near-duplicate pair (by list order).
    Returns (kept_records, n_removed)."""
    texts = [r.get(text_field, "") for r in records]
    dup_pairs = find_near_duplicates(texts, ngram_size, threshold)
    to_drop = {j for _, j in dup_pairs}
    kept = [r for i, r in enumerate(records) if i not in to_drop]
    return kept, len(to_drop)


# --- safety filter -------------------------------------------------------------

def compile_safety_patterns(terms: list[str]) -> list[re.Pattern]:
    return [re.compile(term, re.IGNORECASE) for term in terms]


def matches_safety_filter(text: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


REFUSAL_ANSWER_TEMPLATE = (
    "N kò lè fúnni ní ìwọ̀n òògùn tàbí ìtọ́nisọ́nà ìtọ́jú pàtó fún ìbéèrè yìí. "
    "Jọ̀wọ́ bá dókítà tàbí nọ́ọ̀sì tí ó tọ́ sọ̀rọ̀ fún ìrànlọ́wọ́ tí ó dájú àti tí ó dá."
)


def rewrite_as_refusal(record: dict) -> dict:
    """Build spec 1.8: records matching the safety filter are 'removed from
    the answer set and rewritten as refusal examples' rather than dropped
    outright -- the model should still learn the associated question maps to
    a safe refusal, not silently disappear from the topic's question coverage."""
    rewritten = dict(record)
    rewritten["answer_yo_final"] = REFUSAL_ANSWER_TEMPLATE
    rewritten["question_type"] = "out_of_scope"
    rewritten["notes"] = (record.get("notes", "") + " | safety-filter: rewritten as refusal").strip(" |")
    return rewritten


# --- stratified, claim_id-grouped split ----------------------------------------

def split_by_claim_group(records: list[dict], train: float, val: float, test: float,
                          stratify_field: str, group_field: str, seed: int) -> dict[str, list[dict]]:
    """Splits `records` 80/10/10 (or whatever train/val/test are) such that:
      - every record sharing the same group_field value (claim_id) lands in
        the SAME split, and
      - splits are approximately stratified by stratify_field (topic),
        computed at the GROUP level (a group's stratify label is its first
        record's value).
    Raises AssertionError if the resulting splits are not disjoint in
    group_field values (should be structurally impossible given the
    algorithm, but build-spec rule 3 requires this be asserted in code, not
    just implied by the implementation).
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError(f"train+val+test must sum to 1.0, got {train + val + test}")

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r[group_field]].append(r)

    by_topic: dict[str, list[str]] = defaultdict(list)
    for group_id, group_records in groups.items():
        by_topic[group_records[0].get(stratify_field, "unclassified")].append(group_id)

    rng = random.Random(seed)
    split_groups: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for topic, group_ids in by_topic.items():
        group_ids = list(group_ids)
        rng.shuffle(group_ids)
        n = len(group_ids)
        n_train = round(n * train)
        n_val = round(n * val)
        # whatever remains goes to test, so rounding never drops a group
        split_groups["train"].extend(group_ids[:n_train])
        split_groups["val"].extend(group_ids[n_train:n_train + n_val])
        split_groups["test"].extend(group_ids[n_train + n_val:])

    result = {
        split_name: [r for gid in gids for r in groups[gid]]
        for split_name, gids in split_groups.items()
    }

    train_ids = {r[group_field] for r in result["train"]}
    val_ids = {r[group_field] for r in result["val"]}
    test_ids = {r[group_field] for r in result["test"]}
    assert not (train_ids & val_ids), f"claim_id overlap between train/val: {train_ids & val_ids}"
    assert not (train_ids & test_ids), f"claim_id overlap between train/test: {train_ids & test_ids}"
    assert not (val_ids & test_ids), f"claim_id overlap between val/test: {val_ids & test_ids}"

    return result
