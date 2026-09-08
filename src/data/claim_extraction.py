"""Claim segmentation and consumer-question generation for 02_build_qa.py.

Splits ingested raw text into atomic, fact-bearing sentences ("claims") and
turns each into a consumer-style English question. Two question-generation
backends:

  - "template" (default, offline, deterministic, free): matches the claim
    against a small set of clinical-content keyword categories (symptoms,
    causes, prevention, treatment, transmission, risk) and fills a question
    template for that category. The claim sentence itself becomes the
    answer -- verbatim from the source, so nothing is invented.
  - "llm": uses Claude Haiku 4.5 to phrase a more natural question for the
    same claim/answer pair. Requires ANTHROPIC_API_KEY (see .env.example).
    Falls back to the template backend on any API error so the pipeline
    never blocks on network/API issues.

Both backends produce an ANSWER that is the claim sentence verbatim (or a
near-verbatim light cleanup) -- the LLM is only ever asked to phrase the
QUESTION, never to add or infer clinical content, per build-spec rule 1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# --- claim segmentation ------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# Boilerplate/navigation lines commonly picked up by HTML text extraction.
_NAV_BOILERPLATE_PATTERNS = [
    r"^skip to main content$",
    r"^credits?$",
    r"^share$",
    r"^print$",
    r"^related$",
    r"^see also$",
    r"^references?$",
    r"^key facts?$",
    r"^overview$",
    r"^reading time:?.*$",
    r"^\d{1,2} (january|february|march|april|may|june|july|august|september|october|november|december) \d{4}$",
    r"^who$|^paho$|^ncdc$|^nphcda$|^fmoh$",
    # US .gov site-chrome ("secure connection" padlock notice), verified
    # leaking through as fused pseudo-sentences on MedlinePlus pages.
    r".*official government organization.*",
    r".*safely connected.*\.gov.*",
    # MedlinePlus's own standing site disclaimers (not source content).
    r".*substitute for professional medical care.*",
    r".*medlineplus also links to.*",
    # Citation/further-reading list entries, not prose claims. WHO/MedlinePlus
    # fact sheets end with a numbered bibliography; these patterns were each
    # matched against a real leaked example before being added here:
    r"^(article|journal|report|reference|study|source):\s",
    r".*accessed \d{1,2} \w+ \d{4}.*",             # "...(url, accessed 13 August 2025)."
    r"\bet al\.?\s*$|\bet al\.?,",                  # author-list citations
    r"^\(\d+\)\s",                                  # "(3) Mental health atlas 2020."
    r"^[A-Z][a-zA-Z ]{2,25}:\s.*;\s*\d{4}",         # "Geneva: World Health Organization; 2021"
    r"^licen[cs]e:\s|\bcc by(-nc)?(-sa)?\b",         # "Licence: CC BY-NC-SA 3.0 IGO."
    r"\bdoi:|doi\.org|\b10\.\d{4,9}/",              # DOI citations
    r"\bwebpage\.?\s*$",                            # "... Elimination Programme webpage."
    r",\s*\d+,?\s*\d*\s*\(\d{4}\)\.?\s*$",          # "Reprod Health 18, 216 (2021)."
]
_NAV_BOILERPLATE_RE = re.compile("|".join(_NAV_BOILERPLATE_PATTERNS), re.IGNORECASE)

MIN_CLAIM_CHARS = 25
MAX_CLAIM_CHARS = 400
MIN_CLAIM_WORDS = 5


def is_boilerplate(line: str) -> bool:
    # search(), not match(): several patterns (e.g. "et al.", "webpage.",
    # a trailing journal-citation suffix) are meant to detect a distinctive
    # substring anywhere in the line, not just at position 0. Patterns that
    # need start-anchoring already include an explicit "^".
    return bool(_NAV_BOILERPLATE_RE.search(line.strip()))


def _looks_like_a_sentence(candidate: str) -> bool:
    """Reject link-list text and citation fragments that HTML-to-text
    extraction leaves looking superficially claim-like (right length, capital
    letter) but that are not actually sentences -- e.g. MedlinePlus emits one
    line per related-topic link ("A1C Test and Race/Ethnicity") and citation
    attributions ("(Centers for Disease Control and Prevention)"). Real
    fact-bearing sentences reliably end in sentence punctuation and have
    enough words to carry a verb; both are cheap, conservative filters that
    trade a little recall (unpunctuated bullet-list facts get dropped too)
    for a large reduction in non-claim noise reaching claim_id assignment.
    """
    if not candidate.endswith((".", "!")):
        # Deliberately excludes "?": a claim is meant to become a verbatim
        # declarative ANSWER, and text ending in "?" reaching this stage is
        # reliably a related-article link title ("Cold Remedies: Which Are
        # Safe?"), not a fact the pipeline should answer a question with.
        return False
    if candidate.startswith("(") and candidate.endswith(")"):
        return False
    if len(candidate.split()) < MIN_CLAIM_WORDS:
        return False
    # A genuine sentence starts with a capital letter, a digit, or an opening
    # quote/parenthesis. A candidate starting with a lowercase letter or with
    # continuation punctuation (",", ";", ")") means _merge_wrapped_lines
    # failed to reattach it to its missing subject clause -- e.g. a stray
    # inline-link line break left ", updated in 2021, provides a technical
    # framework..." as its own fragment with no verb's subject. Better to
    # drop it than let a subject-less fragment become an answer.
    first_char = candidate[0]
    if not (first_char.isupper() or first_char.isdigit() or first_char in "\"'“("):
        return False
    return True


def _merge_wrapped_lines(lines: list[str]) -> list[str]:
    """Repair sentences that HTML-to-text extraction fractures across
    multiple "lines" at an inline-tag boundary (e.g. a hyperlink or <em> in
    the middle of a sentence produces a stray newline mid-sentence, such as
    "If you travel\\nto these countries, you are at risk."). If the current
    buffered line does not yet end in sentence punctuation and the next
    non-blank line starts with a lowercase letter OR with continuation
    punctuation (",", ";", ")") -- both strong continuation signals, since a
    real new sentence/heading starts with a capital letter or digit -- join
    them with a space instead of treating them as separate units. The
    comma/semicolon case covers e.g. "The WHO strategy\\n, updated in 2021,
    provides..." where an inline link boundary drops the break right before
    a comma rather than mid-word.
    """
    merged: list[str] = []
    buffer = ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if buffer:
                merged.append(buffer)
                buffer = ""
            continue
        if is_boilerplate(line):
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append(line)
            continue
        if buffer and not buffer.endswith((".", "!", "?", ":")) and (line[:1].islower() or line[:1] in ",;)"):
            buffer = f"{buffer} {line}"
        else:
            if buffer:
                merged.append(buffer)
            buffer = line
    if buffer:
        merged.append(buffer)
    return merged


def split_into_claims(raw_text: str) -> list[str]:
    """Split raw ingested text into candidate atomic claims.

    Heuristic, not a trained sentence-boundary/claim detector: repairs
    mid-sentence line wraps (_merge_wrapped_lines), splits on sentence-ending
    punctuation followed by a capital/digit, then drops lines that are
    boilerplate, too short/long to be a real claim, or that fail the
    _looks_like_a_sentence check (see its docstring).
    """
    claims: list[str] = []
    for line in _merge_wrapped_lines(raw_text.splitlines()):
        if not line or is_boilerplate(line):
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(line):
            sentence = sentence.strip()
            if len(sentence) < MIN_CLAIM_CHARS or len(sentence) > MAX_CLAIM_CHARS:
                continue
            if is_boilerplate(sentence) or not _looks_like_a_sentence(sentence):
                continue
            claims.append(sentence)
    return claims


# --- template question generation -------------------------------------------

@dataclass
class QuestionCategory:
    name: str
    trigger_words: tuple[str, ...]
    template: str  # "{topic}" is substituted with the human-readable topic name


QUESTION_CATEGORIES = [
    QuestionCategory("symptoms", ("symptom", "sign of", "signs of", "feel like"),
                      "What are the symptoms of {topic}?"),
    QuestionCategory("causes", ("caused by", "is caused", "cause of", "due to a"),
                      "What causes {topic}?"),
    QuestionCategory("transmission", ("spread", "transmit", "contagious", "infect"),
                      "How does {topic} spread?"),
    QuestionCategory("prevention", ("prevent", "avoid", "protect against", "vaccine", "vaccination"),
                      "How can {topic} be prevented?"),
    QuestionCategory("treatment", ("treat", "cure", "medicine", "therapy", "manage"),
                      "How is {topic} treated?"),
    QuestionCategory("risk", ("risk", "more likely", "higher chance", "vulnerable"),
                      "Who is most at risk from {topic}?"),
]

GENERIC_TEMPLATE = "What should I know about {topic}?"


def classify_claim(claim: str) -> QuestionCategory | None:
    lowered = claim.lower()
    for category in QUESTION_CATEGORIES:
        if any(trigger in lowered for trigger in category.trigger_words):
            return category
    return None


def generate_question_template(claim: str, topic_display_name: str) -> str:
    category = classify_claim(claim)
    template = category.template if category else GENERIC_TEMPLATE
    return template.format(topic=topic_display_name)


# --- LLM-backed question generation (optional) -------------------------------

LLM_SYSTEM_PROMPT = (
    "You turn one factual health sentence into a single short, natural, "
    "consumer-style English question that the sentence directly and fully "
    "answers. Rules: output ONLY the question, nothing else. The question "
    "must end with a question mark. Do not add any fact, number, or claim "
    "that is not already in the sentence. Do not mention 'the sentence' or "
    "'the text'. Write as an ordinary member of the public would ask a "
    "question, not a medical professional."
)


def generate_question_llm(claim: str, model: str = "claude-haiku-4-5", max_retries: int = 4) -> str:
    """Generate a question via the Anthropic API. Raises on failure -- callers
    should catch and fall back to generate_question_template."""
    import os
    import random
    import time

    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env (see .env.example)
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=100,
                system=LLM_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": claim}],
            )
            text = next((b.text for b in response.content if b.type == "text"), "").strip()
            if not text.endswith("?"):
                raise ValueError(f"LLM did not return a question: {text!r}")
            return text
        except anthropic.RateLimitError as e:
            last_exc = e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                last_exc = e
            else:
                raise
        delay = min(1.0 * (2 ** attempt) + random.uniform(0, 1), 20.0)
        time.sleep(delay)
    raise RuntimeError(f"generate_question_llm exhausted retries: {last_exc}")


def generate_question(claim: str, topic_display_name: str, backend: str = "template") -> tuple[str, str]:
    """Returns (question, backend_actually_used). backend_actually_used lets
    callers log when an 'llm' request silently fell back to 'template'."""
    if backend == "template":
        return generate_question_template(claim, topic_display_name), "template"
    if backend == "llm":
        try:
            return generate_question_llm(claim), "llm"
        except Exception:
            return generate_question_template(claim, topic_display_name), "template_fallback"
    raise ValueError(f"unknown question generation backend: {backend!r}")
