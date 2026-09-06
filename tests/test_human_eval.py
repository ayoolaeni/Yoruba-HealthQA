import pytest

from src.eval.human_eval import (
    HARM_CATEGORIES,
    LIKERT_DIMENSIONS,
    build_blinded_items,
    build_blinding_map,
    build_rater_sheet_rows,
    harm_distribution_by_system,
    harm_incidents,
    mean_ratings_by_system,
    stratified_topic_sample,
    unblind_rater_rows,
)


def _make_test_records(n_per_topic: int, topics=("malaria", "diabetes")) -> list[dict]:
    records = []
    for topic in topics:
        for i in range(n_per_topic):
            records.append({"id": f"{topic}-{i:03d}", "topic": topic, "instruction": f"Q {topic} {i}?",
                             "output": f"A {topic} {i}."})
    return records


def test_stratified_topic_sample_returns_all_if_smaller_than_min_n():
    records = _make_test_records(5)
    sample = stratified_topic_sample(records, min_n=150, seed=1)
    assert sample == records


def test_stratified_topic_sample_respects_min_n_and_stratifies():
    records = _make_test_records(100)  # 200 total, 2 topics
    sample = stratified_topic_sample(records, min_n=150, seed=1)
    assert len(sample) >= 150
    topics = [r["topic"] for r in sample]
    assert topics.count("malaria") > 0
    assert topics.count("diabetes") > 0


def test_build_blinding_map_covers_every_sample_and_system():
    sample_ids = ["s1", "s2", "s3"]
    systems = ["B1", "M"]
    blinding = build_blinding_map(sample_ids, systems, seed=1)
    assert set(blinding.keys()) == set(sample_ids)
    for sample_id, slot_map in blinding.items():
        assert set(slot_map.values()) == set(systems)
        assert len(slot_map) == len(systems)


def test_build_blinding_map_rejects_too_many_systems():
    with pytest.raises(ValueError):
        build_blinding_map(["s1"], [f"sys{i}" for i in range(20)], seed=1)


def test_build_blinding_map_is_independently_randomised_per_sample():
    # with enough samples, slot A should not map to the same system every time
    sample_ids = [f"s{i}" for i in range(30)]
    systems = ["B1", "M"]
    blinding = build_blinding_map(sample_ids, systems, seed=1)
    slot_a_systems = {blinding[sid]["A"] for sid in sample_ids}
    assert slot_a_systems == {"B1", "M"}  # both appear at least once in slot A


def test_build_blinded_items_uses_correct_hypothesis_per_slot():
    sample_records = [{"id": "s1", "topic": "malaria", "instruction": "Q1?"}]
    generations = {
        "B1": {"s1": {"hypothesis": "hyp from B1"}},
        "M": {"s1": {"hypothesis": "hyp from M"}},
    }
    blinding_map = {"s1": {"A": "B1", "B": "M"}}
    items = build_blinded_items(sample_records, generations, blinding_map)
    by_slot = {i.slot: i.response_yo for i in items}
    assert by_slot["A"] == "hyp from B1"
    assert by_slot["B"] == "hyp from M"


def test_build_rater_sheet_rows_have_blank_rating_fields():
    sample_records = [{"id": "s1", "topic": "malaria", "instruction": "Q1?"}]
    generations = {"B1": {"s1": {"hypothesis": "hyp"}}}
    blinding_map = {"s1": {"A": "B1"}}
    items = build_blinded_items(sample_records, generations, blinding_map)
    rows = build_rater_sheet_rows(items, seed=1)
    assert len(rows) == 1
    for dim in LIKERT_DIMENSIONS:
        assert rows[0][dim] == ""
    assert rows[0]["potential_for_harm"] == ""
    assert rows[0]["slot"] == "A"


def _sample_rater_row(sample_id, slot, ratings, harm, comment=""):
    row = {"sample_id": sample_id, "topic": "malaria", "question_yo": "Q?", "slot": slot,
           "response_yo": "A.", "potential_for_harm": harm, "rater_comment": comment}
    row.update(ratings)
    return row


def test_unblind_rater_rows_resolves_system_identity():
    blinding_map = {"s1": {"A": "B1", "B": "M"}}
    ratings = {dim: 4 for dim in LIKERT_DIMENSIONS}
    rows = [_sample_rater_row("s1", "A", ratings, "None"), _sample_rater_row("s1", "B", ratings, "None")]
    unblinded = unblind_rater_rows(rows, rater_id="rater1", blinding_map=blinding_map)
    systems = {u.system_id for u in unblinded}
    assert systems == {"B1", "M"}
    assert all(u.rater_id == "rater1" for u in unblinded)


def test_mean_ratings_by_system_averages_across_raters_and_samples():
    blinding_map = {"s1": {"A": "M"}, "s2": {"A": "M"}}
    r1 = unblind_rater_rows([_sample_rater_row("s1", "A", {d: 5 for d in LIKERT_DIMENSIONS}, "None")],
                             "rater1", blinding_map)
    r2 = unblind_rater_rows([_sample_rater_row("s2", "A", {d: 3 for d in LIKERT_DIMENSIONS}, "None")],
                             "rater2", blinding_map)
    means = mean_ratings_by_system(r1 + r2)
    assert means["M"]["factual_correctness"] == pytest.approx(4.0)


def test_harm_distribution_by_system_counts_categories():
    blinding_map = {"s1": {"A": "M"}, "s2": {"A": "M"}, "s3": {"A": "M"}}
    ratings = {d: 3 for d in LIKERT_DIMENSIONS}
    rows = [
        _sample_rater_row("s1", "A", ratings, "None"),
        _sample_rater_row("s2", "A", ratings, "Severe"),
        _sample_rater_row("s3", "A", ratings, "Severe"),
    ]
    unblinded = unblind_rater_rows(rows, "rater1", blinding_map)
    dist = harm_distribution_by_system(unblinded)
    assert dist["M"]["Severe"] == 2
    assert dist["M"]["None"] == 1
    assert dist["M"]["Low"] == 0


def test_harm_incidents_lists_only_moderate_and_severe_with_details():
    blinding_map = {"s1": {"A": "M"}, "s2": {"A": "M"}}
    ratings = {d: 3 for d in LIKERT_DIMENSIONS}
    rows = [
        _sample_rater_row("s1", "A", ratings, "Low"),
        _sample_rater_row("s2", "A", ratings, "Moderate", comment="incorrect dosage-adjacent claim"),
    ]
    unblinded = unblind_rater_rows(rows, "rater1", blinding_map)
    incidents = harm_incidents(unblinded)
    assert len(incidents) == 1
    assert incidents[0]["sample_id"] == "s2"
    assert incidents[0]["harm_level"] == "Moderate"
    assert incidents[0]["rater_comment"] == "incorrect dosage-adjacent claim"


def test_harm_categories_constant_matches_spec():
    assert HARM_CATEGORIES == ["None", "Low", "Moderate", "Severe"]
