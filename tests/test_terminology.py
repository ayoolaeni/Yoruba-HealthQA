import csv

from src.data.terminology import (
    check_record_terminology,
    extract_term_frequencies,
    load_canonical_terminology,
    write_terminology_csv,
)


def test_extract_term_frequencies_counts_case_insensitively():
    texts = [
        "Malaria is spread by mosquitoes.",
        "MALARIA symptoms include fever.",
        "Typhoid is a bacterial infection.",
    ]
    counts = extract_term_frequencies(texts, vocabulary=["malaria", "typhoid", "diabetes"])
    assert counts["malaria"] == 2
    assert counts["typhoid"] == 1
    assert counts["diabetes"] == 0


def test_extract_term_frequencies_respects_word_boundaries():
    # "TB" should not match inside "STABLE" or similar
    texts = ["The patient is stable.", "TB is treatable."]
    counts = extract_term_frequencies(texts, vocabulary=["TB"])
    assert counts["TB"] == 1


def test_write_terminology_csv_round_trip(tmp_path):
    counts = extract_term_frequencies(
        ["Malaria causes fever.", "Malaria is preventable.", "Typhoid causes fever too."],
        vocabulary=["malaria", "typhoid", "fever"],
    )
    out_path = tmp_path / "terminology.csv"
    n = write_terminology_csv(counts, out_path)
    assert n == 3

    with open(out_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_term = {r["term_en"]: r for r in rows}
    assert by_term["malaria"]["frequency"] == "2"
    assert by_term["fever"]["frequency"] == "2"
    assert by_term["malaria"]["canonical_yo"] == ""  # left blank for researcher


def test_load_canonical_terminology_skips_blank_rows(tmp_path):
    csv_path = tmp_path / "terminology.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["term_en", "frequency", "canonical_yo", "notes"])
        writer.writerow(["malaria", "10", "ibà", ""])
        writer.writerow(["typhoid", "5", "", ""])  # not yet decided by researcher

    canonical = load_canonical_terminology(csv_path)
    assert canonical == {"malaria": "ibà"}


def test_load_canonical_terminology_handles_excel_utf8_bom(tmp_path):
    # Regression test: Excel's "CSV UTF-8 (Comma delimited)" Save As option
    # (the one a researcher must use to keep Yoruba diacritics intact)
    # writes a byte-order-mark at the start of the file. Confirmed this
    # silently broke every row's lookup before load_canonical_terminology
    # switched to utf-8-sig.
    csv_path = tmp_path / "terminology.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["term_en", "frequency", "canonical_yo", "notes"])
        writer.writerow(["measles", "52", "Kitipi", ""])

    canonical = load_canonical_terminology(csv_path)
    assert canonical == {"measles": "Kitipi"}


def test_check_record_terminology_flags_missing_canonical_rendering():
    canonical = {"malaria": "ibà"}
    # Yoruba text uses the canonical rendering -- no flag
    flags = check_record_terminology("Malaria is common here.", "Ibà wọ́pọ̀ níbí.", canonical)
    assert flags == []

    # Yoruba text does NOT contain "iba" anywhere -- flagged for review
    flags = check_record_terminology("Malaria is common here.", "Àrùn kan wọ́pọ̀ níbí.", canonical)
    assert flags == ["malaria"]


def test_check_record_terminology_ignores_terms_not_present_in_english():
    canonical = {"malaria": "ibà", "typhoid": "typhoid"}
    flags = check_record_terminology("This is about hypertension.", "Nípa ẹ̀jẹ̀ríru.", canonical)
    assert flags == []  # neither malaria nor typhoid was mentioned in answer_en
