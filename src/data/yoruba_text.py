"""Yoruba text normalisation, character-set validation, and diacritic metrics.

Build spec section 1.6. This is flagged in the spec as "the most error-prone
code in the repo" — every function here is covered by tests/test_yoruba_text.py
against a hand-written fixture (tests/fixtures/yoruba_text_fixture.json).

Standard Yoruba orthography, decomposed (NFD) view:
  - 18 consonant phonemes: b d f g gb h j k l m n p r s ṣ t w y
    (`gb` is a two-letter digraph; `ṣ` is Latin 's' + COMBINING DOT BELOW in NFD)
  - 7 oral vowels: a e ẹ i o ọ u
    (`ẹ`, `ọ` are 'e'/'o' + COMBINING DOT BELOW in NFD)
  - nasal vowels are vowel+n sequences (an, ẹn, in, un, ọn) — not distinct letters
  - tone: high = COMBINING ACUTE ACCENT (U+0301), low = COMBINING GRAVE ACCENT
    (U+0300), mid = unmarked (occasionally COMBINING MACRON U+0304 is used to
    mark mid tone explicitly in pedagogical text; accepted but not required)
  - syllabic nasals ń / ḿ: base 'n'/'m' + a tone-mark combining accent

Because Unicode has no single precomposed codepoint for a vowel carrying both
an underdot *and* a tone mark (e.g. ẹ̀), the only lossless representation is
NFD-style decomposition (base letter + one or more combining marks). All
validation below therefore normalises to NFD internally, even though NFC is
what gets written to disk (spec 1.6: "Unicode NFC normalisation on all Yorùbá
text") — NFC still composes the single-mark cases (à, ẹ, ṣ, ...) into their
precomposed codepoints; only the doubly-marked vowels remain base+combining
in NFC too, since no precomposed codepoint exists for them.
"""
from __future__ import annotations

import re
import unicodedata as ud
from dataclasses import dataclass, field

# --- Allowed character inventory (checked after NFD decomposition) ---------

YORUBA_CONSONANT_BASE = "bdfghjklmnprstwy"  # 16 plain letters; 'gb' = g+b digraph
YORUBA_VOWEL_BASE = "aeiou"                  # ẹ/ọ appear here as 'e'/'o' + dot-below

ALLOWED_BASE_LOWER = frozenset(YORUBA_CONSONANT_BASE + YORUBA_VOWEL_BASE)
ALLOWED_BASE = frozenset(ALLOWED_BASE_LOWER | {c.upper() for c in ALLOWED_BASE_LOWER})

COMBINING_GRAVE = "̀"       # low tone
COMBINING_ACUTE = "́"       # high tone (also on syllabic ń/ḿ)
COMBINING_MACRON = "̄"      # optional explicit mid tone
COMBINING_DOT_BELOW = "̣"   # underdot: ẹ, ọ, ṣ
ALLOWED_COMBINING = frozenset({COMBINING_GRAVE, COMBINING_ACUTE, COMBINING_MACRON, COMBINING_DOT_BELOW})

# Letters that standard Yoruba orthography does not use.
NON_YORUBA_LATIN = frozenset("cqvxzCQVXZ")

# Health-domain acronyms/loanwords that Yoruba consumer-health text
# conventionally keeps in their original Latin form rather than transliterating
# (mirrors real-world usage in Yoruba WHO/NPHCDA materials). These are the
# only reason a strict-mode record would otherwise be wrongly flagged, since
# most English clinical loanwords already happen to use letters Yoruba also
# uses (e.g. "AIDS" is coincidentally all Yoruba-legal letters) -- it is
# specifically 'c'/'v'/'q'/'x'/'z'-bearing acronyms like HIV/COVID that need
# this explicit allowance. Extend via configs/data.yaml-driven tooling if a
# project needs a different accepted list; this is deliberately small.
ACCEPTED_LOANWORDS = frozenset({"hiv", "aids", "covid", "covid-19", "prep", "art"})
_LOANWORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in ACCEPTED_LOANWORDS) + r")\b", re.IGNORECASE,
)


def _mask_loanwords(text: str) -> str:
    """Replaces each accepted loanword occurrence with same-length spaces so
    position indices into the rest of the string are unaffected."""
    return _LOANWORD_PATTERN.sub(lambda m: " " * len(m.group(0)), text)

# Punctuation, whitespace, and digits are always permitted (question marks,
# commas, numerals in dosage-free contexts, etc.) regardless of strict mode.
ALLOWED_OTHER = frozenset(" \t\n\r.,;:!?'‘’\"“”()-–—/%&0123456789")


def normalize_nfc(text: str) -> str:
    """Canonicalise text to Unicode NFC. Call this on every Yoruba string before
    it is written to disk, compared, or hashed."""
    return ud.normalize("NFC", text)


def strip_diacritics(text: str) -> str:
    """Remove all combining marks (tone marks and underdots), returning the
    bare-letter form used for `instruction_nodia`. NFC in, NFC out."""
    nfd = ud.normalize("NFD", text)
    stripped = "".join(ch for ch in nfd if ud.category(ch) != "Mn")
    return ud.normalize("NFC", stripped)


@dataclass
class CharsetValidationResult:
    is_valid: bool
    invalid_chars: list[str] = field(default_factory=list)
    invalid_positions: list[int] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid


def validate_charset(text: str, strict: bool = True, allow_loanwords: bool = False) -> CharsetValidationResult:
    """Check every character of `text` against the permitted Yoruba set.

    strict=True (default): only the 18 consonants / 7 vowels (+ digraph `gb`),
        the four combining marks above, and common punctuation/digits/whitespace
        are allowed. Any Latin letter outside that set (c, q, v, x, z, or any
        non-Latin script) is flagged — use this for records that must be pure
        Yoruba (e.g. `instruction`, `output` fields).
    strict=False: any alphabetic character is accepted in addition to the
        strict set. Use this only where deliberate code-switching or a
        transliterated clinical term is expected, e.g. free-text notes fields.
    allow_loanwords=True: additionally accepts whole-word occurrences of
        ACCEPTED_LOANWORDS (HIV, COVID, ...) before checking the rest of the
        text strictly. Use this for real consumer-health Yoruba text, which
        conventionally keeps these acronyms in Latin form rather than
        transliterating them.
    """
    scan_text = _mask_loanwords(text) if allow_loanwords else text
    nfd = ud.normalize("NFD", scan_text)
    invalid_chars: list[str] = []
    invalid_positions: list[int] = []
    for i, ch in enumerate(nfd):
        if ch in ALLOWED_BASE or ch in ALLOWED_COMBINING or ch in ALLOWED_OTHER:
            continue
        if not strict and ch.isalpha():
            continue
        invalid_chars.append(ch)
        invalid_positions.append(i)
    return CharsetValidationResult(
        is_valid=len(invalid_chars) == 0,
        invalid_chars=invalid_chars,
        invalid_positions=invalid_positions,
    )


def is_legal_yoruba(text: str, strict: bool = True, allow_loanwords: bool = False) -> bool:
    """Convenience boolean wrapper around validate_charset, used by import
    validators (04b_import_postedit.py, 05b_import_validation.py)."""
    return validate_charset(text, strict=strict, allow_loanwords=allow_loanwords).is_valid


def _tokenize(text: str) -> list[str]:
    return normalize_nfc(text).split()


def diacritic_accuracy(hyp: str, ref: str) -> float:
    """Proportion of hypothesis tokens whose full (diacritised) form matches the
    reference, computed over the tokens that align at all once diacritics are
    stripped.

    Algorithm (documented explicitly since the spec does not fix one):
      1. Tokenise hyp and ref on whitespace (NFC-normalised).
      2. Run difflib.SequenceMatcher over the diacritic-stripped token
         sequences to find aligned ("equal") token spans — this locates which
         hyp tokens correspond to which ref tokens even under reordering or
         insertions/deletions elsewhere in the sentence.
      3. Among aligned token pairs (i.e. tokens the system got lexically right
         once diacritics are removed), compute the fraction whose *original*
         diacritised form is identical.

    This isolates diacritic-restoration quality from lexical-choice errors:
    a token the model got wrong lexically was never going to have its
    diacritics scored right or wrong, so it is excluded from the denominator.
    Returns 0.0 (with denominator 0) if no tokens align at all.
    """
    import difflib

    hyp_tokens = _tokenize(hyp)
    ref_tokens = _tokenize(ref)
    hyp_bare = [strip_diacritics(t).lower() for t in hyp_tokens]
    ref_bare = [strip_diacritics(t).lower() for t in ref_tokens]

    matcher = difflib.SequenceMatcher(a=hyp_bare, b=ref_bare, autojunk=False)
    aligned = 0
    correct = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for hi, rj in zip(range(i1, i2), range(j1, j2)):
            aligned += 1
            if hyp_tokens[hi] == ref_tokens[rj]:
                correct += 1

    if aligned == 0:
        return 0.0
    return correct / aligned


def validate_record_text_fields(record: dict, fields: list[str], strict: bool = True) -> dict[str, CharsetValidationResult]:
    """Run validate_charset over a subset of a JSONL record's fields. Returns a
    dict mapping field name -> CharsetValidationResult for any field present."""
    results = {}
    for field_name in fields:
        value = record.get(field_name)
        if isinstance(value, str) and value:
            results[field_name] = validate_charset(value, strict=strict)
    return results
