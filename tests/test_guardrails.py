import pytest

from src.serve.guardrails import (
    DISCLAIMER_YO,
    G1_REFUSAL_YO,
    G2_REDIRECT_YO,
    GuardrailOutcome,
    apply_disclaimer,
    clinical_boundary_check,
    guarded_response,
    scope_check,
)

# --- G1 scope check ----------------------------------------------------------

G1_POSITIVE_HEALTH_QUESTIONS = [
    "Kí ni àwọn àmì àrùn ibà?",
    "Bawo ni a se le dena typhoid?",
    "Kini ajesara toye ki omo mi gba?",
    "Se hiv le tan nipase ifenumo?",
    "Kilode ti won fi n so pe suga le fa afọju?",
]

G1_NEGATIVE_OUT_OF_SCOPE_QUESTIONS = [
    "Kini owo dola loni?",
    "Se won le so fun mi bi mo se le ko football daadaa?",
    "Kilode ti aye fi n yi ka oorun?",
    "Bawo ni mo se le ra foonu tuntun?",
]


@pytest.mark.parametrize("question", G1_POSITIVE_HEALTH_QUESTIONS)
def test_g1_accepts_health_questions(question):
    assert scope_check(question) is True


@pytest.mark.parametrize("question", G1_NEGATIVE_OUT_OF_SCOPE_QUESTIONS)
def test_g1_rejects_non_health_questions(question):
    assert scope_check(question) is False


# --- G2 clinical-boundary check ----------------------------------------------

G2_POSITIVE_BOUNDARY_QUESTIONS = [
    "Iwon oogun wo ni mo gbodo mu fun iba?",
    "Melo ni mo gbodo mu ninu paracetamol?",
    "Se mo ni typhoid, so fun mi arun wo ni mo ni?",
    "Ko oogun fun mi fun iba.",
    "What dosage of amoxicillin should I give my child?",
]

G2_NEGATIVE_NON_BOUNDARY_HEALTH_QUESTIONS = [
    "Kí ni àwọn àmì àrùn ibà?",
    "Bawo ni a se le dena typhoid?",
    "Kini ajesara to ye ki omo mi gba?",
]


@pytest.mark.parametrize("question", G2_POSITIVE_BOUNDARY_QUESTIONS)
def test_g2_flags_dosage_and_diagnosis_requests(question):
    assert clinical_boundary_check(question) is True


@pytest.mark.parametrize("question", G2_NEGATIVE_NON_BOUNDARY_HEALTH_QUESTIONS)
def test_g2_does_not_flag_ordinary_health_questions(question):
    assert clinical_boundary_check(question) is False


# --- G3 disclaimer ------------------------------------------------------------

def test_g3_disclaimer_is_appended():
    answer = "Àmì àrùn ibà ni ibà gbígbóná àti orí fífọ́."
    result = apply_disclaimer(answer)
    assert result.startswith(answer)
    assert DISCLAIMER_YO in result


def test_g3_disclaimer_present_on_every_outcome_path():
    def fake_generate(_q):
        return "Ìdáhùn àpẹẹrẹ."

    outcomes = [
        guarded_response("Kilode ti aye fi n yi ka oorun?", fake_generate),  # G1
        guarded_response("Iwon oogun wo ni mo gbodo mu fun iba?", fake_generate),  # G2
        guarded_response("Kí ni àwọn àmì àrùn ibà?", fake_generate),  # answered
    ]
    for outcome in outcomes:
        assert DISCLAIMER_YO in outcome.text


# --- orchestration ------------------------------------------------------------

def test_guarded_response_out_of_scope_does_not_call_model():
    calls = []

    def fake_generate(q):
        calls.append(q)
        return "should not be reached"

    result = guarded_response("Kilode ti aye fi n yi ka oorun?", fake_generate)
    assert result.outcome == GuardrailOutcome.OUT_OF_SCOPE
    assert G1_REFUSAL_YO in result.text
    assert calls == []


def test_guarded_response_clinical_boundary_does_not_call_model():
    calls = []

    def fake_generate(q):
        calls.append(q)
        return "should not be reached"

    result = guarded_response("Ko oogun fun mi fun iba.", fake_generate)
    assert result.outcome == GuardrailOutcome.CLINICAL_BOUNDARY
    assert G2_REDIRECT_YO in result.text
    assert calls == []


def test_guarded_response_answers_in_scope_safe_question():
    def fake_generate(q):
        return "Àmì àrùn ibà ni ibà gbígbóná."

    result = guarded_response("Kí ni àwọn àmì àrùn ibà?", fake_generate)
    assert result.outcome == GuardrailOutcome.ANSWERED
    assert "Àmì àrùn ibà ni ibà gbígbóná." in result.text
