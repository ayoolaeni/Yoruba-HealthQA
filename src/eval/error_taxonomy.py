"""Phase 9 -- Error taxonomy heuristics (build spec section 11).

Classifying WHY a response is wrong (factual error vs hallucination vs
terminological error, say) is a semantic judgment call that this module
deliberately does NOT make automatically -- doing so would mean an
unreviewed heuristic silently stands in for human error analysis, which
build-spec rule 1 ("never invent data") rules out for anything that becomes
a reported result.

What this module DOES do reliably from signal already computed elsewhere
(diacritic_accuracy, language_consistency, length) is pre-fill SUGGESTED
flags for the categories that have a cheap, explainable detector:
  - diacritic_error   : diacritic_accuracy(hyp, ref) below a threshold
  - code_switching    : hypothesis has a high proportion of non-Yoruba tokens
  - omission          : hypothesis is much shorter than the reference
  - disfluency        : hypothesis repeats an immediate word/bigram

These are SUGGESTIONS an error-analyst reviews and corrects in
12_error_analysis.py's exported sheet -- factual_error, terminological_error,
hallucination, and refusal_failure are always left blank for human judgment,
since nothing short of clinical/linguistic review can tell those apart from
the text alone.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.data.yoruba_text import diacritic_accuracy
from src.eval.metrics import language_consistency

ERROR_CATEGORIES = [
    "factual_error", "omission", "terminological_error", "diacritic_error",
    "code_switching", "disfluency", "refusal_failure", "hallucination",
]

DIACRITIC_ERROR_THRESHOLD = 0.7
CODE_SWITCHING_THRESHOLD = 0.85   # language_consistency below this suggests code-switching
OMISSION_LENGTH_RATIO_THRESHOLD = 0.5  # hyp word count / ref word count below this suggests omission


@dataclass
class HeuristicFlags:
    diacritic_error: bool
    code_switching: bool
    omission: bool
    disfluency: bool


def _has_repeated_bigram(text: str) -> bool:
    words = text.split()
    bigrams = list(zip(words, words[1:]))
    return len(bigrams) != len(set(bigrams))


def suggest_flags(hyp: str, ref: str) -> HeuristicFlags:
    hyp_words = hyp.split()
    ref_words = ref.split()
    length_ratio = len(hyp_words) / len(ref_words) if ref_words else 1.0

    return HeuristicFlags(
        diacritic_error=diacritic_accuracy(hyp, ref) < DIACRITIC_ERROR_THRESHOLD,
        code_switching=language_consistency([hyp]) < CODE_SWITCHING_THRESHOLD if hyp.strip() else False,
        omission=length_ratio < OMISSION_LENGTH_RATIO_THRESHOLD,
        disfluency=_has_repeated_bigram(hyp),
    )


def select_lowest_scoring(items: list[dict], score_field: str, n: int) -> list[dict]:
    """items: dicts with at least `score_field` (lower = worse, e.g.
    per-sentence chrF++). Returns the n lowest-scoring items, sorted worst
    first. Returns all items if there are fewer than n."""
    return sorted(items, key=lambda x: x[score_field])[:n]
