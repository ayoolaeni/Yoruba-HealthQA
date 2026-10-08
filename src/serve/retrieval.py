"""Demo-only retrieval fallback for app/app.py -- NOT the dissertation's
formal system.

The dissertation's actual system under evaluation is "M", the fine-tuned
instruction-tuned model (scripts/09_train.py + src/serve/inference.py),
scored offline against baselines B1-B4 exactly as configs/eval.yaml and
scripts/10_evaluate.py specify (build spec section 8). Nothing here changes
that pipeline, those configs, or those results.

This module exists only because the small-scale demo fine-tune (169
examples; see reports/table_4_4_hyperparameter_sweep.md) does not yet
generate reliably accurate live answers. Rather than show a live audience
an unreliable generation, this looks up the closest matching question in
the validated demo dataset and returns its REAL, human-post-edited answer
verbatim -- it never generates or invents text, so there is no
hallucination risk. When presenting this, say so explicitly: it is a
lookup over the validated dataset, not the model under formal evaluation.

Reuses the exact same G1/G2/G3 guardrail pipeline as InferenceService
(src/serve/guardrails.guarded_response), by exposing the same
`generate_fn(question) -> str` call shape.
"""
from __future__ import annotations

import difflib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.data.yoruba_text import strip_diacritics
from src.serve.guardrails import GuardedResponse, guarded_response

# Calibrated against the demo's own example questions (see
# tests/test_retrieval.py): an exact or near-exact question match scores
# close to 1.0, while an unrelated question scores well under this.
DEFAULT_SIMILARITY_THRESHOLD = 0.55

# A second, independent signal alongside the character-level ratio above.
# The character ratio only catches questions worded almost identically to
# one in the dataset -- a real query ("what can I do if I have fever and
# headache?") scored only 0.48 against a real, relevant match ("what are
# malaria's symptoms?") purely because the wording differs, even though
# both share the meaningful word "ibà" (fever/malaria).
#
# A first fix (plain word-overlap on content words) caught that case, but
# a second real test found it was NOT enough on its own: "Kí ló máa ń fa
# ibà ní ara ènìyàn?" (what causes fever?) confidently returned an answer
# about MEASLES, because "fà" (cause) and "ló" (a focus particle) are
# shared by nearly every "what causes X?" question in the dataset,
# regardless of X -- a flat stopword list didn't catch "ló" since it's not
# purely grammatical, and "fà" is a real word, just not a topic-specific
# one here.
#
# The fix: weight each shared word by how many DISTINCT DISEASE TOPICS it
# appears in (not how many questions) -- "ló"/"fà" appear across most or
# all of the dataset's 13 topics (weak signal), while "ibà" appears only
# under "malaria" (strong signal). Verified against all three real cases
# above plus a genuinely unrelated query (cooking rice): the correct
# malaria question now scores highest for the bug case, and the unrelated
# query's best score (~0.15) stays below the threshold used for a real
# match (~0.17-0.27 in the same test).
DEFAULT_WORD_OVERLAP_THRESHOLD = 0.16

# Extra safety net alongside the threshold above: require that at least
# ONE shared word individually clears this bar (appears in few enough
# topics to be a real topic anchor, e.g. "ibà" or "àtọ̀gbẹ"), so a match
# can never be built purely by accumulating several medium-common words
# with no single topic-specific word in common.
ANCHOR_WORD_IDF_THRESHOLD = 2.0

# Common Yoruba function words (question words, pronouns, copula,
# conjunctions, particles) -- standard stop-word filtering, kept as a
# first pass alongside the topic-weighting above. This is a linguistic
# preprocessing list, not a factual/clinical claim.
YORUBA_STOPWORDS = frozenset({
    "ki", "kini", "kilo", "ta", "tani", "ewo", "bawo", "nigba", "nibo", "ee",
    "se", "ni", "mo", "o", "a", "won", "emi", "oun", "iwo", "awa", "eyin",
    "ti", "ko", "si", "naa", "yi", "yii", "na", "ati", "tabi", "bi", "ba",
    "fun", "lati", "to", "je", "wa", "ri", "le", "ma", "maa", "tun", "pe",
    "n", "yoo", "nse", "loo", "sinu", "laarin", "pelu", "sugbon", "yen",
})

NO_MATCH_YO = (
    "Mo ṣì ń kọ́ẹ̀kọ́ nípa irú ìbéèrè yìí, mi ò tíì ní ìdáhùn tí a fọwọ́sí fún un. "
    "Jọ̀wọ́ gbìyànjú láti béèrè lọ́nà mìíràn, tàbí lo ọ̀kan lára àwọn àpẹẹrẹ ìbéèrè tí a fún ní ìsàlẹ̀."
)


def _normalise(text: str) -> str:
    return strip_diacritics(text).lower().strip()


def _content_words(text: str) -> set[str]:
    return {w for w in _normalise(text).replace("?", "").split() if w not in YORUBA_STOPWORDS}


class _TopicWeighting:
    """Weights each word by how many distinct topics it appears in across
    the loaded dataset -- a word confined to one or two topics (e.g.
    "ibà") is a strong, specific signal; a word spread across most topics
    (e.g. "fà"/"ló", generic verbs shared by many question templates) is
    weak, however often it appears in raw word counts. See the module
    docstring's threshold comment for the real bug this fixes."""

    def __init__(self, questions_with_topics: list[tuple[str, str]]):
        topics_per_word: dict[str, set[str]] = defaultdict(set)
        all_topics: set[str] = set()
        for question, topic in questions_with_topics:
            all_topics.add(topic)
            for w in _content_words(question):
                topics_per_word[w].add(topic)
        self._topics_per_word = topics_per_word
        self._n_topics = max(len(all_topics), 1)

    def idf(self, word: str) -> float:
        n_topics_for_word = len(self._topics_per_word.get(word, ()))
        return math.log((self._n_topics + 1) / (n_topics_for_word + 1)) + 1

    def weighted_overlap(self, a_words: set[str], b_words: set[str]) -> tuple[float, float]:
        """Returns (weighted_jaccard_score, strongest_shared_word_idf)."""
        union = a_words | b_words
        if not union:
            return 0.0, 0.0
        shared = a_words & b_words
        score = sum(self.idf(w) for w in shared) / sum(self.idf(w) for w in union)
        best_anchor = max((self.idf(w) for w in shared), default=0.0)
        return score, best_anchor


@dataclass
class RetrievalMatch:
    answer: str
    matched_question: str
    similarity: float


class RetrievalBackend:
    """Looks up the closest real, validated Yoruba answer to a question.

    Every dataset file must contain records with "instruction" (the Yoruba
    question) and "output" (the Yoruba answer) fields -- the same schema
    scripts/09_train.py trains on (src/data/schema.py). No text is ever
    generated: only real, already-existing answers are returned.
    """

    def __init__(self, dataset_paths: list[str | Path], similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                 word_overlap_threshold: float = DEFAULT_WORD_OVERLAP_THRESHOLD):
        self.similarity_threshold = similarity_threshold
        self.word_overlap_threshold = word_overlap_threshold
        # (normalised_question, content_words, original_question, answer)
        self._entries: list[tuple[str, set[str], str, str]] = []
        seen_ids: set[str] = set()
        questions_with_topics: list[tuple[str, str]] = []
        for raw_path in dataset_paths:
            path = Path(raw_path)
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    record_id = record.get("id")
                    if record_id is not None:
                        if record_id in seen_ids:
                            continue
                        seen_ids.add(record_id)
                    question = record["instruction"]
                    answer = record["output"]
                    self._entries.append((_normalise(question), _content_words(question), question, answer))
                    questions_with_topics.append((question, record.get("topic", "unknown")))
        if not self._entries:
            raise ValueError(f"no usable records found in {dataset_paths} -- each record needs "
                              f"'instruction' and 'output' fields (src/data/schema.py)")
        self._weighting = _TopicWeighting(questions_with_topics)

    def best_match(self, question: str) -> RetrievalMatch | None:
        query = _normalise(question)
        query_words = _content_words(question)
        best_score = 0.0
        tied_best: list[tuple[str, str]] = []  # (original_question, answer)
        for norm_q, words_q, orig_q, answer in self._entries:
            # Two independent signals, each already calibrated on its own
            # scale (see the threshold constants' comments) -- a candidate
            # counts as a match if EITHER clears its own bar, not a single
            # blended score, since the two scales aren't comparable (word
            # overlap on short questions rarely exceeds ~0.3 even for a
            # great match, unlike the character ratio).
            char_ratio = difflib.SequenceMatcher(None, query, norm_q).ratio()
            word_ratio, best_anchor_idf = self._weighting.weighted_overlap(query_words, words_q)
            # A real test caught this: two questions can share a generic
            # sentence template ("Báwo ni ... ṣe ...", or "Kí ló ń fa...")
            # and score deceptively high while being about completely
            # different topics (real cases: "how do I cook rice well?"
            # scored 0.60 character overlap against a TB question; "what
            # causes fever?" matched a measles answer via the shared verb
            # "fà"). Requiring a genuine topic-anchor word in common --
            # not just any shared word -- for EITHER signal to count is
            # what actually closes this, not just lowering/raising a
            # single threshold.
            has_topic_anchor = best_anchor_idf >= ANCHOR_WORD_IDF_THRESHOLD
            is_match = has_topic_anchor and (
                word_ratio >= self.word_overlap_threshold or char_ratio >= self.similarity_threshold
            )
            score = max(char_ratio, word_ratio)
            if not is_match:
                continue
            if score > best_score + 1e-9:
                best_score = score
                tied_best = [(orig_q, answer)]
            elif score > best_score - 1e-9:
                tied_best.append((orig_q, answer))

        if not tied_best:
            return None
        # Several source claims can share an identical elicited question
        # (e.g. several malaria facts all phrased as "what are the
        # symptoms?"). Among equally-good question matches, prefer the
        # longest real answer as a simple, content-blind proxy for
        # completeness -- never a choice between real and invented text,
        # only between which already-validated answer to surface first.
        orig_q, answer = max(tied_best, key=lambda pair: len(pair[1]))
        return RetrievalMatch(answer=answer, matched_question=orig_q, similarity=best_score)

    def generate(self, question: str) -> str:
        """Same call shape as ModelBackend.generate(question, decoding) minus
        decoding params, so this can be passed directly as guarded_response's
        `generate_fn` -- reusing G1/G2/G3 completely unchanged."""
        match = self.best_match(question)
        return match.answer if match is not None else NO_MATCH_YO


class RetrievalService:
    """Drop-in alternative to InferenceService with the same `.answer()`
    call shape (returns GuardedResponse), so app/app.py's answer_fn does not
    need to know which mode is active."""

    def __init__(self, dataset_paths: list[str | Path], similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self._backend = RetrievalBackend(dataset_paths, similarity_threshold=similarity_threshold)

    def answer(self, question: str) -> GuardedResponse:
        return guarded_response(question, self._backend.generate)
