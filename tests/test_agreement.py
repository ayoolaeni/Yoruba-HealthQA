import numpy as np
import pytest

from src.eval.agreement import (
    build_category_count_matrix,
    cohens_kappa,
    compute_double_annotation_kappa,
    fleiss_kappa,
    interpret_kappa,
    krippendorff_alpha_ordinal,
)


def test_cohens_kappa_perfect_agreement():
    a = ["pass", "pass", "fail", "pass", "fail"]
    b = ["pass", "pass", "fail", "pass", "fail"]
    assert cohens_kappa(a, b) == pytest.approx(1.0)


def test_cohens_kappa_chance_level_or_below():
    # constructed so observed agreement equals expected chance agreement
    a = ["pass", "pass", "fail", "fail"]
    b = ["pass", "fail", "pass", "fail"]
    kappa = cohens_kappa(a, b)
    assert kappa <= 0.34  # low agreement, not perfect


def test_cohens_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cohens_kappa(["a"], ["a", "b"])


def test_interpret_kappa_bands():
    assert interpret_kappa(-0.1) == "poor"
    assert interpret_kappa(0.05) == "slight"
    assert interpret_kappa(0.25) == "fair"
    assert interpret_kappa(0.55) == "moderate"
    assert interpret_kappa(0.75) == "substantial"
    assert interpret_kappa(0.95) == "almost perfect"
    assert interpret_kappa(1.0) == "almost perfect"


def test_fleiss_kappa_perfect_agreement():
    # 4 items, 3 raters each, all raters agree every time
    item_ratings = [
        ["None", "None", "None"],
        ["Severe", "Severe", "Severe"],
        ["Low", "Low", "Low"],
        ["Moderate", "Moderate", "Moderate"],
    ]
    categories = ["None", "Low", "Moderate", "Severe"]
    matrix = build_category_count_matrix(item_ratings, categories)
    assert fleiss_kappa(matrix) == pytest.approx(1.0)


def test_fleiss_kappa_requires_uniform_rater_count():
    matrix = np.array([[2, 0], [1, 0]])  # item 0 has 2 raters, item 1 has 1
    with pytest.raises(ValueError):
        fleiss_kappa(matrix)


def test_krippendorff_alpha_perfect_agreement():
    # 3 raters, 5 items, identical ordinal ratings -> alpha == 1.0
    item_ratings = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
    ]
    assert krippendorff_alpha_ordinal(item_ratings) == pytest.approx(1.0)


def test_compute_double_annotation_kappa_perfect_agreement():
    rows = [
        {"id": "a", "annotator_slot": 1, "verdict_pass_fail": "pass"},
        {"id": "a", "annotator_slot": 2, "verdict_pass_fail": "pass"},
        {"id": "b", "annotator_slot": 1, "verdict_pass_fail": "fail"},
        {"id": "b", "annotator_slot": 2, "verdict_pass_fail": "fail"},
        {"id": "c", "annotator_slot": 1, "verdict_pass_fail": "pass"},  # not double-annotated
    ]
    kappa, interpretation, n_double = compute_double_annotation_kappa(rows)
    assert kappa == pytest.approx(1.0)
    assert interpretation == "almost perfect"
    assert n_double == 2


def test_compute_double_annotation_kappa_insufficient_overlap_returns_none():
    rows = [
        {"id": "a", "annotator_slot": 1, "verdict_pass_fail": "pass"},
        {"id": "b", "annotator_slot": 1, "verdict_pass_fail": "fail"},
    ]
    kappa, interpretation, n_double = compute_double_annotation_kappa(rows)
    assert kappa is None
    assert interpretation is None
    assert n_double == 0


def test_krippendorff_alpha_tolerates_missing_values():
    item_ratings = [
        [1, 2, 3, None, 5],
        [1, 2, None, 4, 5],
        [1, None, 3, 4, 5],
    ]
    alpha = krippendorff_alpha_ordinal(item_ratings)
    assert alpha == pytest.approx(1.0)
