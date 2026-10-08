"""Tests for the demo-only retrieval fallback (src/serve/retrieval.py).

These test against the REAL data/final_demo files (the same validated,
human-post-edited dataset the app ships with) rather than synthetic
fixtures, because the whole point of retrieval is that every returned
answer is real text that already exists in that file -- a synthetic
fixture would not catch a real mismatch between this code and that data.
"""
from pathlib import Path

import pytest

from src.serve.guardrails import GuardrailOutcome
from src.serve.retrieval import NO_MATCH_YO, RetrievalBackend, RetrievalService

DATASET_PATHS = [
    "data/final_demo/train.jsonl",
    "data/final_demo/val.jsonl",
    "data/final_demo/test.jsonl",
]


def _skip_if_dataset_missing():
    if not any(Path(p).exists() for p in DATASET_PATHS):
        pytest.skip("data/final_demo/*.jsonl not present in this checkout")


def test_exact_question_match_returns_real_answer():
    _skip_if_dataset_missing()
    backend = RetrievalBackend(DATASET_PATHS)
    match = backend.best_match("Kí ni àwọn àmì àrùn ibà?")
    assert match is not None
    assert match.similarity > 0.99
    assert match.answer  # a real, non-empty answer string
    assert "ibà" in match.matched_question.lower() or "iba" in match.matched_question.lower()


def test_unrelated_question_returns_no_match():
    _skip_if_dataset_missing()
    backend = RetrievalBackend(DATASET_PATHS)
    match = backend.best_match("What is the capital of France in the year 3000 quantum edition?")
    assert match is None


def test_generate_falls_back_to_honest_message_on_no_match():
    _skip_if_dataset_missing()
    backend = RetrievalBackend(DATASET_PATHS)
    text = backend.generate("asdkjhaskjdh completely unrelated gibberish query zzz")
    assert text == NO_MATCH_YO


def test_never_invents_text_answer_is_one_of_the_real_dataset_answers():
    _skip_if_dataset_missing()
    backend = RetrievalBackend(DATASET_PATHS)
    real_answers = {answer for *_, answer in backend._entries}
    match = backend.best_match("Kí ni àwọn àmì àrùn ibà?")
    assert match.answer in real_answers


def test_paraphrased_question_now_matches_via_word_overlap():
    """Regression test for a real failure: this exact question used to
    return NO_MATCH_YO because it scored only 0.48 on character overlap
    against the closest real match, just under the 0.55 cutoff -- even
    though both share the meaningful word 'ibà' (fever/malaria). Word
    overlap on content words should now catch this without any risk of
    inventing text -- it still only ever returns a real, existing answer."""
    _skip_if_dataset_missing()
    backend = RetrievalBackend(DATASET_PATHS)
    match = backend.best_match("Kí ni mo lè ṣe tí mo bá ní ibà àti orí fífọ́?")
    assert match is not None
    assert "ibà" in match.matched_question.lower() or "iba" in match.matched_question.lower()


def test_shared_generic_verb_does_not_force_a_wrong_topic_match():
    """Regression test for a real failure found live: 'Kí ló máa ń fa ibà
    ní ara ènìyàn?' (what causes fever?) confidently returned a MEASLES
    answer, because 'fà' (cause) and 'ló' (a focus particle) are shared by
    nearly every "what causes X?" question in the dataset regardless of
    topic X -- a flat stopword list didn't catch this since neither word
    is purely grammatical. Topic-diversity weighting fixes it: whatever
    match survives must be genuinely about malaria/fever, not measles."""
    _skip_if_dataset_missing()
    backend = RetrievalBackend(DATASET_PATHS)
    match = backend.best_match("Kí ló máa ń fa ibà ní ara ènìyàn?")
    if match is not None:
        assert "ibà" in match.matched_question.lower() or "iba" in match.matched_question.lower()
        assert "measles" not in match.answer.lower()


def test_word_overlap_does_not_match_across_unrelated_topics():
    """The word-overlap addition must not make matching sloppy -- a
    question about something else entirely (not just off-topic, which G1
    would catch first, but a different in-dataset-adjacent topic) must
    still return no match rather than a wrong-topic answer."""
    _skip_if_dataset_missing()
    backend = RetrievalBackend(DATASET_PATHS)
    match = backend.best_match("Báwo ni mo ṣe lè se iresi dáadáa?")  # "how do I cook rice well?"
    assert match is None


def test_retrieval_service_still_applies_g1_out_of_scope():
    _skip_if_dataset_missing()
    service = RetrievalService(DATASET_PATHS)
    result = service.answer("Kilode ti aye fi n yi ka oorun?")
    assert result.outcome == GuardrailOutcome.OUT_OF_SCOPE


def test_retrieval_service_still_applies_g2_clinical_boundary():
    _skip_if_dataset_missing()
    service = RetrievalService(DATASET_PATHS)
    result = service.answer("Iwon oogun wo ni mo gbodo mu fun iba?")
    assert result.outcome == GuardrailOutcome.CLINICAL_BOUNDARY


def test_retrieval_service_answers_in_scope_question_with_disclaimer():
    _skip_if_dataset_missing()
    service = RetrievalService(DATASET_PATHS)
    result = service.answer("Kí ni àwọn àmì àrùn ibà?")
    assert result.outcome == GuardrailOutcome.ANSWERED
    assert "Àkíyèsí" in result.text  # G3 disclaimer marker


def test_missing_dataset_files_raise_clear_error():
    with pytest.raises(ValueError):
        RetrievalBackend(["data/does_not_exist_at_all.jsonl"])
