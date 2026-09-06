from src.data.claim_extraction import (
    generate_question,
    generate_question_template,
    is_boilerplate,
    split_into_claims,
)


def test_is_boilerplate_flags_known_nav_lines():
    assert is_boilerplate("Skip to main content")
    assert is_boilerplate("Credits")
    assert is_boilerplate("4 December 2025")
    assert not is_boilerplate("Malaria is a life-threatening disease spread by mosquitoes.")


def test_split_into_claims_extracts_sentences_and_drops_boilerplate():
    raw_text = (
        "Malaria\n"
        "Skip to main content\n"
        "Key facts\n"
        "Malaria is a life-threatening disease spread to humans by some types of mosquitoes. "
        "It is mostly found in tropical countries. It is preventable and curable.\n"
        "Credits\n"
    )
    claims = split_into_claims(raw_text)
    assert any("life-threatening disease" in c for c in claims)
    assert any("preventable and curable" in c for c in claims)
    assert not any("Skip to main content" in c for c in claims)
    assert not any(c.strip() == "Credits" for c in claims)


def test_split_into_claims_drops_medlineplus_style_link_and_citation_noise():
    # Regression test: real MedlinePlus pages emit one line per related-topic
    # link and per citation attribution; these are the right length and start
    # with a capital letter but are not sentences and must not become claims.
    raw_text = (
        "High blood pressure can be treated with lifestyle changes and medicines.\n"
        "A1C Test and Race/Ethnicity\n"
        "(National Library of Medicine)\n"
        "High Blood Pressure Stage 2\n"
        "(Centers for Disease Control and Prevention)\n"
    )
    claims = split_into_claims(raw_text)
    assert claims == ["High blood pressure can be treated with lifestyle changes and medicines."]


def test_split_into_claims_repairs_inline_tag_line_wraps():
    # Regression test: an inline <a>/<em> tag inside a sentence can make
    # BeautifulSoup emit a stray newline mid-sentence. The continuation line
    # starts lowercase, which is the signal used to rejoin it.
    raw_text = "If you travel\nto these countries, you are at risk of malaria.\n"
    claims = split_into_claims(raw_text)
    assert claims == ["If you travel to these countries, you are at risk of malaria."]


def test_split_into_claims_drops_question_mark_link_titles():
    # Regression test: related-article link titles can end in "?" and are
    # not usable as a declarative answer.
    raw_text = (
        "Blood pressure is the force of your blood pushing against artery walls.\n"
        "High Blood Pressure and Cold Remedies: Which Are Safe?\n"
    )
    claims = split_into_claims(raw_text)
    assert claims == ["Blood pressure is the force of your blood pushing against artery walls."]


def test_split_into_claims_drops_too_short_and_too_long():
    raw_text = "Ok.\n" + ("A very long run-on sentence. " * 30) + "\n"
    claims = split_into_claims(raw_text)
    assert "Ok." not in claims
    assert all(len(c) <= 400 for c in claims)


def test_generate_question_template_symptoms():
    claim = "Common symptoms of malaria are fever, chills, and headache."
    q = generate_question_template(claim, "malaria")
    assert q == "What are the symptoms of malaria?"


def test_generate_question_template_prevention():
    claim = "Malaria can be prevented by avoiding mosquito bites and with medicines."
    q = generate_question_template(claim, "malaria")
    assert q == "How can malaria be prevented?"


def test_generate_question_template_generic_fallback():
    claim = "Malaria was first described thousands of years ago."
    q = generate_question_template(claim, "malaria")
    assert q == "What should I know about malaria?"


def test_generate_question_dispatches_template_backend():
    claim = "Malaria is treated with antimalarial medicines."
    q, backend_used = generate_question(claim, "malaria", backend="template")
    assert backend_used == "template"
    assert q == "How is malaria treated?"


def test_generate_question_rejects_unknown_backend():
    import pytest

    with pytest.raises(ValueError):
        generate_question("x", "malaria", backend="nonsense")
