"""Service-layer guardrails (Chapter 3, section 3.7.4).

Deliberately rule-based, NOT left to learned model behaviour:
  G1 scope check          -> question outside the health domain: Yoruba refusal (FR4)
  G2 clinical-boundary check -> diagnosis/dosage/prescription request: Yoruba
                                redirection to a qualified professional (FR5)
  G3 disclaimer            -> fixed Yoruba referral disclaimer on every answer (FR6)

Each guardrail is a pure function over the (diacritic-stripped, lowercased)
question text so it is robust to missing tone marks in user input, and each is
independently unit-tested with positive and negative cases in
tests/test_guardrails.py.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from src.data.yoruba_text import strip_diacritics

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")

# --- G1: scope check ---------------------------------------------------------

# Health-domain keyword list: Yoruba terms (diacritic-stripped) plus common
# English loanwords/code-switch forms consumer questions actually use. Extend
# this list from real query logs / the topic taxonomy in configs/data.yaml
# rather than guessing further terms.
HEALTH_KEYWORDS_NODIA = {
    # general
    "ilera", "arun", "aisan", "aisan", "oogun", "dokita", "onisegun", "iwosan",
    "ile-iwosan", "itoju", "ranse", "kokoro", "kokoro arun", "ikolu",
    # topics from configs/data.yaml
    "iba", "malaria", "typhoid", "taifodu", "tb", "iko", "hiv", "aids",
    "titeka", "haipatensonu", "titesan", "atosan", "diabetes", "suga",
    "oyun", "aboyun", "ibimo", "omo tuntun", "omo", "ajesara", "abere",
    "ounje", "ounjeje", "ounje jijẹ", "omi mimu", "imototo", "wash",
    "ilera opolo", "ere okan", "iṣoro ọkan", "covid", "corona",
    # symptom / body words that reliably indicate a health question
    "ara", "iba gbigbona", "eebi", "gbuuru", "orififo", "ikọ", "otutu",
}


def _keyword_hit(bare_text: str, keywords: set[str]) -> bool:
    """Match single-word keywords against tokens (not substrings, to avoid
    e.g. 'tb' or 'ara' matching inside an unrelated longer word), and
    multi-word phrases against the joined text."""
    tokens = {_PUNCT_RE.sub("", tok) for tok in bare_text.split()}
    for kw in keywords:
        if " " in kw:
            if kw in bare_text:
                return True
        elif _PUNCT_RE.sub("", kw) in tokens:
            return True
    return False


def scope_check(question: str) -> bool:
    """Return True if the question is judged in-scope (health domain)."""
    bare = strip_diacritics(question).lower()
    return _keyword_hit(bare, HEALTH_KEYWORDS_NODIA)


G1_REFUSAL_YO = (
    "Má bínú, èmi kò lè dáhùn ìbéèrè yìí nítorí kò jẹ mọ́ ọ̀rọ̀ ìlera. "
    "Mo lè ràn ọ́ lọ́wọ́ pẹ̀lú ìbéèrè nípa àrùn, ìlera, àti ìtọ́jú nìkan."
)

# --- G2: clinical-boundary check --------------------------------------------

CLINICAL_BOUNDARY_KEYWORDS_NODIA = {
    "iwon oogun", "melo ni mo gbodo mu", "melo ninu oogun", "iwon lati mu",
    "se mo ni", "se aisan mi ni", "so fun mi arun wo", "yan oogun fun mi",
    "ko oogun fun mi", "fun mi ni oogun", "juwe oogun", "juwe",
    "dosage", "dose", "mg/kg", "milligram", "prescribe", "prescription",
    "diagnose me", "how much should i take", "what dose",
}


def clinical_boundary_check(question: str) -> bool:
    """Return True if the question asks for a diagnosis, a dosage, or a
    prescription -- content the service must redirect rather than answer."""
    bare = strip_diacritics(question).lower()
    return _keyword_hit(bare, CLINICAL_BOUNDARY_KEYWORDS_NODIA)


G2_REDIRECT_YO = (
    "Èmi kò lè ṣe àyẹ̀wò àrùn tàbí sọ ìwọ̀n òògùn fún ọ, nítoríèyí nílò "
    "akọ́ṣẹ́mọṣẹ́ ìlera tí ó ti kọ́ ẹ̀kọ́ nípa rẹ̀. Jọ̀wọ́ lọ sí ilé-ìwòsàn tàbí "
    "sọ̀rọ̀ pẹ̀lú dókítà tàbí nọ́ọ̀sì tó súnmọ́ ọ fún ìrànlọ́wọ́ tí ó tọ́."
)

# --- G3: disclaimer -----------------------------------------------------------

DISCLAIMER_YO = (
    "Àkíyèsí: Ìwífún yìí kò rọ́pò ìmọ̀ràn dókítà. Bí àrùn bá le tàbí bí kò "
    "bá dára sí i, jọ̀wọ́ lọ sí ilé-ìwòsàn tí ó súnmọ́ ọ láìpẹ́."
)


def apply_disclaimer(answer: str) -> str:
    return f"{answer.strip()}\n\n{DISCLAIMER_YO}"


class GuardrailOutcome(str, Enum):
    OUT_OF_SCOPE = "out_of_scope"
    CLINICAL_BOUNDARY = "clinical_boundary"
    ANSWERED = "answered"


@dataclass
class GuardedResponse:
    outcome: GuardrailOutcome
    text: str


def guarded_response(question: str, generate_fn: Callable[[str], str]) -> GuardedResponse:
    """Full guardrail pipeline: G1 -> G2 -> (model) -> G3.

    `generate_fn` is only ever called when the question passes both G1 and G2,
    so an ungrounded or unsafe generation is never a possibility for those
    cases -- the refusal/redirect text is fixed, not model output.
    """
    if not scope_check(question):
        return GuardedResponse(GuardrailOutcome.OUT_OF_SCOPE, apply_disclaimer(G1_REFUSAL_YO))

    if clinical_boundary_check(question):
        return GuardedResponse(GuardrailOutcome.CLINICAL_BOUNDARY, apply_disclaimer(G2_REDIRECT_YO))

    answer = generate_fn(question)
    return GuardedResponse(GuardrailOutcome.ANSWERED, apply_disclaimer(answer))
