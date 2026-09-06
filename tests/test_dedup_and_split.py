import pytest

from src.data.dedup_and_split import (
    REFUSAL_ANSWER_TEMPLATE,
    char_ngrams,
    compile_safety_patterns,
    dedup_records,
    find_near_duplicates,
    jaccard_similarity,
    matches_safety_filter,
    rewrite_as_refusal,
    split_by_claim_group,
)


def test_char_ngrams_basic():
    grams = char_ngrams("hello", 3)
    assert grams == {"hel", "ell", "llo"}


def test_jaccard_similarity_identical_is_one():
    a = char_ngrams("malaria is bad", 5)
    assert jaccard_similarity(a, a) == 1.0


def test_jaccard_similarity_disjoint_is_zero():
    a = char_ngrams("aaaaa", 5)
    b = char_ngrams("zzzzz", 5)
    assert jaccard_similarity(a, b) == 0.0


def test_find_near_duplicates_detects_near_identical_text():
    texts = [
        "Malaria is a life-threatening disease spread by mosquitoes.",
        "Malaria is a life-threatening disease spread by mosquitoes",  # same, trailing period scraped off
        "Diabetes affects how your body processes blood sugar.",
    ]
    pairs = find_near_duplicates(texts, ngram_size=5, threshold=0.85)
    assert pairs == [(0, 1)]


def test_dedup_records_drops_later_duplicate():
    records = [
        {"id": "a", "answer_en": "Malaria is a life-threatening disease spread by mosquitoes."},
        {"id": "b", "answer_en": "Malaria is a life-threatening disease spread by mosquitoes"},
        {"id": "c", "answer_en": "Diabetes affects how your body processes blood sugar."},
    ]
    kept, n_removed = dedup_records(records, "answer_en", ngram_size=5, threshold=0.85)
    assert n_removed == 1
    assert [r["id"] for r in kept] == ["a", "c"]


def test_safety_filter_flags_dosage_language():
    patterns = compile_safety_patterns(["dosage", "how much.*should i take"])
    assert matches_safety_filter("What dosage of paracetamol is safe?", patterns)
    assert matches_safety_filter("How much should I take for a headache?", patterns)
    assert not matches_safety_filter("What are the symptoms of malaria?", patterns)


def test_rewrite_as_refusal_replaces_answer_and_marks_out_of_scope():
    record = {"answer_yo_final": "Mu miligiramu 500 lẹ́ẹ̀mejì lójúmọ́.", "question_type": "derived", "notes": ""}
    rewritten = rewrite_as_refusal(record)
    assert rewritten["answer_yo_final"] == REFUSAL_ANSWER_TEMPLATE
    assert rewritten["question_type"] == "out_of_scope"
    assert "safety-filter" in rewritten["notes"]
    # original record must be untouched (caller may still want it for auditing)
    assert record["answer_yo_final"] == "Mu miligiramu 500 lẹ́ẹ̀mejì lójúmọ́."


def _make_records(n_claims: int, records_per_claim: int, topic: str, prefix: str) -> list[dict]:
    records = []
    for c in range(n_claims):
        claim_id = f"{prefix}-{c:03d}"
        for r in range(records_per_claim):
            records.append({"id": f"{prefix}-{c:03d}-{r}", "claim_id": claim_id, "topic": topic})
    return records


def test_split_by_claim_group_has_zero_claim_id_overlap():
    records = _make_records(30, 2, "malaria", "m") + _make_records(30, 2, "diabetes", "d")
    result = split_by_claim_group(records, train=0.8, val=0.1, test=0.1,
                                   stratify_field="topic", group_field="claim_id", seed=42)
    train_ids = {r["claim_id"] for r in result["train"]}
    val_ids = {r["claim_id"] for r in result["val"]}
    test_ids = {r["claim_id"] for r in result["test"]}
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)


def test_split_by_claim_group_keeps_claim_records_together():
    records = _make_records(20, 3, "malaria", "m")
    result = split_by_claim_group(records, train=0.8, val=0.1, test=0.1,
                                   stratify_field="topic", group_field="claim_id", seed=1)
    # every claim_id's records must all land in exactly one split
    claim_to_splits: dict[str, set[str]] = {}
    for split_name, split_records in result.items():
        for r in split_records:
            claim_to_splits.setdefault(r["claim_id"], set()).add(split_name)
    assert all(len(splits) == 1 for splits in claim_to_splits.values())


def test_split_by_claim_group_covers_all_records():
    records = _make_records(50, 1, "malaria", "m")
    result = split_by_claim_group(records, train=0.8, val=0.1, test=0.1,
                                   stratify_field="topic", group_field="claim_id", seed=7)
    total = sum(len(v) for v in result.values())
    assert total == len(records)


def test_split_by_claim_group_is_roughly_stratified_by_topic():
    records = _make_records(100, 1, "malaria", "m") + _make_records(100, 1, "diabetes", "d")
    result = split_by_claim_group(records, train=0.8, val=0.1, test=0.1,
                                   stratify_field="topic", group_field="claim_id", seed=3)
    train_topics = [r["topic"] for r in result["train"]]
    malaria_fraction = train_topics.count("malaria") / len(train_topics)
    assert 0.4 < malaria_fraction < 0.6  # roughly balanced given 100/100 input


def test_split_by_claim_group_rejects_bad_ratios():
    with pytest.raises(ValueError):
        split_by_claim_group([], train=0.8, val=0.1, test=0.05,
                              stratify_field="topic", group_field="claim_id", seed=1)


def test_split_by_claim_group_deterministic_given_seed():
    records = _make_records(40, 2, "malaria", "m")
    result_a = split_by_claim_group(records, train=0.8, val=0.1, test=0.1,
                                     stratify_field="topic", group_field="claim_id", seed=99)
    result_b = split_by_claim_group(records, train=0.8, val=0.1, test=0.1,
                                     stratify_field="topic", group_field="claim_id", seed=99)
    assert [r["id"] for r in result_a["train"]] == [r["id"] for r in result_b["train"]]
