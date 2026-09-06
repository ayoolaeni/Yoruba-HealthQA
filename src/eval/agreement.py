"""Inter-rater agreement statistics for Phase 1.7 (linguistic/clinical
validation) and Phase 7 (human evaluation).

  - Cohen's kappa: pairwise, double-annotated linguistic/clinical validation
    records (build spec 1.7), interpreted per Landis & Koch (1977).
  - Fleiss' kappa: the categorical "potential for harm" rubric dimension
    across >=2 raters (build spec Phase 7) -- never averaged, always a kappa.
  - Krippendorff's alpha: the ordinal 5-point Likert rubric dimensions across
    raters, tolerant of missing ratings (not every rater rates every item).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

LANDIS_KOCH_BANDS = [
    (0.00, "poor"),
    (0.20, "slight"),
    (0.40, "fair"),
    (0.60, "moderate"),
    (0.80, "substantial"),
    (1.01, "almost perfect"),  # upper bound sentinel, 1.0 falls in this band
]


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch (1977) categorical interpretation bands."""
    prev_upper = -1.0
    for upper, label in LANDIS_KOCH_BANDS:
        if prev_upper <= kappa < upper:
            return label
        prev_upper = upper
    return "almost perfect"


def cohens_kappa(rater_a: list, rater_b: list) -> float:
    """Cohen's kappa for two raters over the same n items with categorical
    labels (build spec 1.7 double-annotation agreement)."""
    if len(rater_a) != len(rater_b):
        raise ValueError("cohens_kappa requires equal-length, paired rating lists")
    n = len(rater_a)
    if n == 0:
        raise ValueError("cohens_kappa requires at least one paired rating")

    labels = sorted(set(rater_a) | set(rater_b))
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    confusion = np.zeros((k, k))
    for a, b in zip(rater_a, rater_b):
        confusion[idx[a], idx[b]] += 1

    po = np.trace(confusion) / n
    row_marg = confusion.sum(axis=1) / n
    col_marg = confusion.sum(axis=0) / n
    pe = float((row_marg * col_marg).sum())

    if pe == 1.0:
        return 1.0  # raters agree on everything and predict the same single category
    return (po - pe) / (1 - pe)


def fleiss_kappa(rating_matrix: np.ndarray) -> float:
    """Fleiss' kappa for >=2 raters rating N items into k categories, where
    not every rater need rate every item in the same fixed order -- caller
    passes an (N, k) count matrix: rating_matrix[i, j] = number of raters who
    assigned item i to category j. Build with `build_category_count_matrix`.
    """
    n_items, n_categories = rating_matrix.shape
    ratings_per_item = rating_matrix.sum(axis=1)
    if not np.allclose(ratings_per_item, ratings_per_item[0]):
        raise ValueError("Fleiss' kappa requires the same number of raters per item")
    n_raters = ratings_per_item[0]
    if n_raters < 2:
        raise ValueError("Fleiss' kappa requires at least 2 raters per item")

    p_j = rating_matrix.sum(axis=0) / (n_items * n_raters)
    P_i = (np.sum(rating_matrix * rating_matrix, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e_bar = float((p_j ** 2).sum())

    if P_e_bar == 1.0:
        return 1.0
    return (P_bar - P_e_bar) / (1 - P_e_bar)


def build_category_count_matrix(item_ratings: list[list], categories: list) -> np.ndarray:
    """item_ratings: one list of rater labels per item (lists may have
    different lengths across items only if you pre-filter to a fixed rater
    count before calling fleiss_kappa). categories: the fixed category
    vocabulary and column order."""
    cat_idx = {c: i for i, c in enumerate(categories)}
    matrix = np.zeros((len(item_ratings), len(categories)))
    for i, ratings in enumerate(item_ratings):
        counts = Counter(ratings)
        for cat, count in counts.items():
            matrix[i, cat_idx[cat]] = count
    return matrix


def krippendorff_alpha_ordinal(item_ratings: list[list[float | None]]) -> float:
    """Krippendorff's alpha, ordinal level of measurement, tolerant of missing
    values (pass None where a rater did not rate a given item).

    item_ratings: one list per rater (not per item) -- i.e. item_ratings[r][i]
    is rater r's rating of item i, or None if unrated. This matches the
    `krippendorff` package's `reliability_data` convention.
    """
    import krippendorff

    reliability_data = [
        [np.nan if v is None else float(v) for v in rater_row]
        for rater_row in item_ratings
    ]
    return float(krippendorff.alpha(reliability_data=reliability_data, level_of_measurement="ordinal"))


def compute_double_annotation_kappa(rows: list[dict]) -> tuple[float | None, str | None, int]:
    """Used by 05b_import_validation.py. `rows` are validation-sheet rows,
    each with 'id', 'annotator_slot' (1 or 2), and 'verdict_pass_fail'.
    Groups rows by id, keeps ids with exactly both slot 1 and slot 2 filled
    in, and computes Cohen's kappa on their verdicts. Returns
    (kappa, landis_koch_interpretation, n_double_annotated); kappa is None if
    fewer than 2 double-annotated ids (kappa is undefined/unstable below that).
    """
    by_id: dict[str, dict[int, str]] = defaultdict(dict)
    for row in rows:
        slot = int(row.get("annotator_slot") or 1)
        verdict = str(row.get("verdict_pass_fail", "")).strip().lower()
        by_id[row["id"]][slot] = verdict

    double = {rid: slots for rid, slots in by_id.items() if 1 in slots and 2 in slots}
    if len(double) < 2:
        return None, None, len(double)

    rater_a = [slots[1] for slots in double.values()]
    rater_b = [slots[2] for slots in double.values()]
    kappa = cohens_kappa(rater_a, rater_b)
    return kappa, interpret_kappa(kappa), len(double)
