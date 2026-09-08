"""Clinical term extraction and canonical-rendering enforcement (build spec
1.4). Pure string/regex logic so it is unit-testable without any ML deps.
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from src.data.yoruba_text import normalize_nfc, strip_diacritics

# A seed clinical-term vocabulary (English). Extraction looks for any of
# these appearing (case-insensitively, word-boundary) inside answer_en text;
# the point is to surface which specific answers use which term, so the
# researcher can standardise the Yoruba rendering *before* post-editing
# starts, per build spec: "enforce it: a checker flags any record using a
# non-canonical rendering of a listed term." Extend this list as new
# clinical vocabulary appears in the corpus.
SEED_CLINICAL_TERMS_EN = [
    "malaria", "typhoid", "tuberculosis", "TB", "HIV", "AIDS", "hypertension",
    "blood pressure", "diabetes", "insulin", "glucose", "vaccine", "vaccination",
    "immunisation", "immunization", "antibiotic", "antimalarial", "antiretroviral",
    "ART", "PrEP", "dosage", "prescription", "diagnosis", "symptom", "fever",
    "mosquito", "parasite", "bacteria", "virus", "infection", "contagious",
    "pregnancy", "prenatal", "postnatal", "malnutrition", "nutrition", "anaemia",
    "anemia", "dehydration", "sanitation", "hygiene",
    # Added after real machine-translation output review surfaced these as
    # high-risk: NLLB either mistranslated them outright (breastfeeding ->
    # "oyún"/pregnancy) or rendered them inconsistently across records
    # (measles). Flagging these for the researcher's canonical_yo column is
    # more important than most of the seed list above.
    "breastfeeding", "measles", "stunting", "stunted", "wastewater", "sludge",
    "cholera", "depression", "mental health", "maternal mortality",
    "pneumonia", "diarrhoea", "diarrhea", "obesity", "overweight",
]

TERMINOLOGY_CSV_COLUMNS = ["term_en", "frequency", "canonical_yo", "notes"]


def extract_term_frequencies(texts: list[str], vocabulary: list[str] | None = None) -> Counter:
    vocabulary = vocabulary or SEED_CLINICAL_TERMS_EN
    counts: Counter = Counter()
    patterns = {term: re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in vocabulary}
    for text in texts:
        for term, pattern in patterns.items():
            if pattern.search(text):
                counts[term] += 1
    return counts


def write_terminology_csv(counts: Counter, out_path: str | Path) -> int:
    """Writes data/terminology.csv for the researcher to fill in canonical_yo.
    Rows are pre-populated with term_en/frequency; canonical_yo/notes are left
    blank for the researcher, per build spec 1.4."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TERMINOLOGY_CSV_COLUMNS)
        writer.writeheader()
        for term, freq in counts.most_common():
            writer.writerow({"term_en": term, "frequency": freq, "canonical_yo": "", "notes": ""})
    return len(counts)


def load_canonical_terminology(csv_path: str | Path) -> dict[str, str]:
    """Loads the researcher-completed terminology.csv into
    {term_en_lower: canonical_yo}, skipping rows left blank (not yet decided)."""
    canonical: dict[str, str] = {}
    # utf-8-sig, not utf-8: Excel's "CSV UTF-8 (Comma delimited)" Save As
    # writes a byte-order-mark at the start of the file. Plain utf-8 leaves
    # that BOM attached to the first header cell ("term_en" becomes
    # "﻿term_en"), which silently breaks every row's lookup -- verified
    # by reproducing an Excel-saved file and getting an empty result back.
    # utf-8-sig strips the BOM if present and behaves like plain utf-8 if not.
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            canonical_yo = (row.get("canonical_yo") or "").strip()
            term_en = (row.get("term_en") or "").strip()
            if canonical_yo and term_en:
                canonical[term_en.lower()] = normalize_nfc(canonical_yo)
    return canonical


def check_record_terminology(answer_en: str, yoruba_text: str, canonical: dict[str, str]) -> list[str]:
    """Flag terms for human review: for each seed clinical term present in
    `answer_en`, check whether `yoruba_text` (the record's Yoruba
    question+answer) contains the canonical rendering registered for that
    term in data/terminology.csv. Returns the term_en values whose canonical
    rendering is missing -- meaning the record may have used a non-canonical
    synonym, dropped the term, or mistranslated it, per build spec 1.4:
    "enforce it: a checker flags any record using a non-canonical rendering
    of a listed term."

    This is a presence check, not a semantic one: it cannot tell a correct
    non-canonical synonym from an actual error, which is why it flags for
    human review rather than auto-rejecting.
    """
    present_terms = extract_term_frequencies([answer_en], vocabulary=list(canonical.keys()))
    bare_yo = strip_diacritics(yoruba_text).lower()
    return [term for term in present_terms if strip_diacritics(canonical[term]).lower() not in bare_yo]
