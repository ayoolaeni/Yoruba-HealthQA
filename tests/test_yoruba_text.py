import json
import unicodedata as ud
from pathlib import Path

import pytest

from src.data.yoruba_text import (
    diacritic_accuracy,
    is_legal_yoruba,
    normalize_nfc,
    strip_diacritics,
    validate_charset,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "yoruba_text_fixture.json"


@pytest.fixture(scope="module")
def fixture():
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_nfc_is_idempotent(fixture):
    for text in fixture["nfc_idempotent_cases"]:
        assert normalize_nfc(text) == text


def test_nfc_composes_decomposed_input(fixture):
    # Construct a genuinely NFD-decomposed form at runtime (safer than hand
    # -typing combining-character byte sequences into a JSON fixture) and
    # confirm normalize_nfc recomposes it back to the canonical NFC form.
    for text in fixture["nfc_idempotent_cases"]:
        nfd_form = ud.normalize("NFD", text)
        assert normalize_nfc(nfd_form) == text


def test_strip_diacritics(fixture):
    for case in fixture["strip_diacritics_cases"]:
        assert strip_diacritics(case["input"]) == case["expected_nodia"]


def test_strip_diacritics_is_nfc(fixture):
    for case in fixture["strip_diacritics_cases"]:
        result = strip_diacritics(case["input"])
        assert result == normalize_nfc(result)


def test_charset_valid_strict(fixture):
    for text in fixture["charset_valid_strict"]:
        result = validate_charset(text, strict=True)
        assert result.is_valid, f"expected valid, got invalid chars {result.invalid_chars!r} in {text!r}"
        assert is_legal_yoruba(text, strict=True)


def test_charset_invalid_strict(fixture):
    for text in fixture["charset_invalid_strict"]:
        result = validate_charset(text, strict=True)
        assert not result.is_valid, f"expected invalid due to non-Yoruba Latin letters in {text!r}"
        assert not is_legal_yoruba(text, strict=True)


def test_charset_invalid_strict_becomes_valid_when_lenient(fixture):
    # strict=False accepts any alphabetic char (deliberate code-switching mode)
    for text in fixture["charset_invalid_strict"]:
        assert is_legal_yoruba(text, strict=False)


def test_charset_strict_rejects_hiv_without_loanword_allowance():
    assert not is_legal_yoruba("Àwọn tí ó ní HIV nílò ìtọ́jú.", strict=True)


def test_charset_strict_accepts_hiv_with_loanword_allowance():
    assert is_legal_yoruba("Àwọn tí ó ní HIV nílò ìtọ́jú.", strict=True, allow_loanwords=True)


def test_charset_loanword_allowance_still_rejects_real_english_contamination():
    result = validate_charset("HIV can be caused by unsafe practices.", strict=True, allow_loanwords=True)
    assert not result.is_valid  # "caused" still has a 'c', loanword allowance doesn't blanket-permit English


def test_diacritic_accuracy(fixture):
    for case in fixture["diacritic_accuracy_cases"]:
        got = diacritic_accuracy(case["hyp"], case["ref"])
        assert got == pytest.approx(case["expected"]), case["description"]


def test_diacritic_accuracy_is_symmetric_denominator_only_not_score():
    # Swapping hyp/ref need not give the same score in general, but a perfect
    # match must score 1.0 regardless of which side is "hyp".
    text = "Kí ni àwọn àmì àrùn ibà?"
    assert diacritic_accuracy(text, text) == 1.0
