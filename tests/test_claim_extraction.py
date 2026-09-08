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


def test_split_into_claims_drops_us_gov_site_chrome():
    # Regression test: MedlinePlus/.gov pages emit a fixed "secure connection"
    # notice that fuses into a plausible-looking pseudo-sentence.
    raw_text = (
        "Malaria is a life-threatening disease spread by mosquitoes.\n"
        ") or https:// means you've safely connected to the .gov website.\n"
        "A .gov website belongs to an official government organization in the United States.\n"
    )
    claims = split_into_claims(raw_text)
    assert claims == ["Malaria is a life-threatening disease spread by mosquitoes."]


def test_split_into_claims_drops_citation_list_entries():
    raw_text = (
        "Tuberculosis disease is treated with special antibiotics.\n"
        "Article: Modeling Treatment Response in Tuberculosis Early Bactericidal Activity Trials.\n"
    )
    claims = split_into_claims(raw_text)
    assert claims == ["Tuberculosis disease is treated with special antibiotics."]


def test_split_into_claims_drops_medlineplus_standing_disclaimers():
    raw_text = (
        "Diabetes can damage blood vessels in the heart, eyes, kidneys and nerves.\n"
        "The information on this site should not be used as a substitute for professional medical care or advice.\n"
        "MedlinePlus also links to health information from non-government Web sites.\n"
    )
    claims = split_into_claims(raw_text)
    assert claims == ["Diabetes can damage blood vessels in the heart, eyes, kidneys and nerves."]


def test_split_into_claims_drops_freeform_bibliography_entries():
    raw_text = (
        "Depression is different from usual mood fluctuations.\n"
        "Seattle: Institute for Health Metrics and Evaluation; 2024 (https://example.org/data, accessed 13 August 2025).\n"
    )
    claims = split_into_claims(raw_text)
    assert claims == ["Depression is different from usual mood fluctuations."]


def test_split_into_claims_drops_bibliography_entries():
    # Regression test: each line below is a real reference-list entry that
    # leaked through into the dataset before these filters were added.
    good = "Malaria can also cause anaemia."
    bad_lines = [
        "Evans-Lacko S, Aguilar-Gaxiola S, Al-Hamzawi A, et al.",
        "(3) Mental health atlas 2020.",
        "Geneva: World Health Organization; 2021 (https://iris.who.int/handle/10665/345946).",
        "Licence: CC BY-NC-SA 3.0 IGO.",
        "2021;14(Suppl 1) (https://doi.org/10.1080/16549716.2021.1974677).",
        "Mekong Malaria Elimination Programme webpage.",
        "Reprod Health 18, 216 (2021).",
    ]
    raw_text = good + "\n" + "\n".join(bad_lines) + "\n"
    claims = split_into_claims(raw_text)
    assert claims == [good]


def test_split_into_claims_merges_comma_led_continuation():
    # Regression test: an inline link boundary can drop the line break right
    # before a comma rather than mid-word; the fragment must be reattached
    # to its subject clause, not kept as a standalone claim.
    raw_text = "The WHO Global Technical Strategy for malaria\n, updated in 2021, provides a technical framework.\n"
    claims = split_into_claims(raw_text)
    assert claims == ["The WHO Global Technical Strategy for malaria , updated in 2021, provides a technical framework."]


def test_split_into_claims_drops_fragment_starting_with_comma_when_unmergeable():
    # If a comma-led fragment has no preceding buffer to merge into (e.g. it
    # is the very first line of the extracted text), it must still be
    # dropped rather than kept as a subject-less claim.
    raw_text = ", updated in 2021, provides a technical framework for all countries.\n"
    claims = split_into_claims(raw_text)
    assert claims == []


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
