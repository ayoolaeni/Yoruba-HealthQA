import pytest

from src.eval.metrics import (
    corpus_bleu,
    corpus_chrf,
    corpus_rouge_l,
    language_consistency,
    paired_bootstrap,
    sentence_chrf_scores,
)


def test_corpus_chrf_perfect_match_is_100():
    hyps = ["Kí ni àwọn àmì àrùn ibà?", "Ẹ mu omi tó pọ̀."]
    refs = hyps
    assert corpus_chrf(hyps, refs) == pytest.approx(100.0, abs=1e-6)


def test_corpus_chrf_worse_for_unrelated_text():
    refs = ["Kí ni àwọn àmì àrùn ibà?"]
    good = ["Kí ni àwọn àmì àrùn ibà?"]
    bad = ["Owo dola ga loni."]
    assert corpus_chrf(good, refs) > corpus_chrf(bad, refs)


def test_sentence_chrf_scores_length_matches_input():
    hyps = ["a b c", "d e f", "g h i"]
    refs = ["a b c", "x y z", "g h i"]
    scores = sentence_chrf_scores(hyps, refs)
    assert len(scores) == 3
    assert scores[0] > scores[1]


def test_corpus_bleu_perfect_match():
    hyps = ["Kí ni àwọn àmì àrùn ibà?"]
    refs = hyps
    score, signature = corpus_bleu(hyps, refs)
    assert score == pytest.approx(100.0, abs=1e-6)
    assert isinstance(signature, str) and len(signature) > 0


def test_corpus_rouge_l_perfect_match():
    hyps = ["Ẹ mu omi tó pọ̀ kí ẹ sì sinmi"]
    refs = hyps
    assert corpus_rouge_l(hyps, refs) == pytest.approx(1.0, abs=1e-6)


def test_language_consistency_all_yoruba():
    hyps = ["Kí ni àwọn àmì àrùn ibà"]
    assert language_consistency(hyps) == pytest.approx(1.0)


def test_language_consistency_detects_english_codeswitch():
    # "doctor" and "vaccine" each contain a letter (c, v) outside the strict
    # Yoruba set, so the orthographic filter should flag them even though it
    # cannot flag every English word (see language_consistency docstring).
    hyps = ["Kí ni the doctor said about the vaccine"]
    score = language_consistency(hyps)
    assert 0.0 < score < 1.0


def test_language_consistency_empty_input_is_zero():
    assert language_consistency([]) == 0.0


def test_paired_bootstrap_detects_clear_improvement():
    scores_baseline = [10.0] * 50
    scores_m = [50.0] * 50
    result = paired_bootstrap(scores_baseline, scores_m, n_resamples=500, seed=42)
    assert result.mean_diff == pytest.approx(40.0)
    assert result.p_value < 0.05
    assert result.ci_low > 0


def test_paired_bootstrap_no_difference_is_not_significant():
    import random

    rng = random.Random(1)
    scores_baseline = [50.0 + rng.uniform(-1, 1) for _ in range(200)]
    scores_m = [50.0 + rng.uniform(-1, 1) for _ in range(200)]
    result = paired_bootstrap(scores_baseline, scores_m, n_resamples=500, seed=42)
    assert result.p_value > 0.05


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0], n_resamples=10)


def test_paired_bootstrap_rejects_empty_input():
    with pytest.raises(ValueError):
        paired_bootstrap([], [], n_resamples=10)
