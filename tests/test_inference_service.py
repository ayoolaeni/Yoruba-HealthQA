import pytest

from src.serve.guardrails import DISCLAIMER_YO, G1_REFUSAL_YO, G2_REDIRECT_YO, GuardrailOutcome
from src.serve.inference import InferenceService


class FakeBackend:
    def __init__(self, response="Àmì àrùn ibà ni ibà gbígbóná."):
        self.response = response
        self.calls = []

    def generate(self, question, decoding):
        self.calls.append((question, decoding))
        return self.response


def test_inference_service_loads_decoding_params_from_eval_config():
    backend = FakeBackend()
    service = InferenceService(eval_config_path="configs/eval.yaml", backend=backend)
    assert service.decoding["max_new_tokens"] == 256
    assert service.decoding["do_sample"] is False


def test_inference_service_answers_in_scope_question():
    backend = FakeBackend()
    service = InferenceService(eval_config_path="configs/eval.yaml", backend=backend)
    result = service.answer("Kí ni àwọn àmì àrùn ibà?")
    assert result.outcome == GuardrailOutcome.ANSWERED
    assert backend.calls  # model was actually invoked
    assert DISCLAIMER_YO in result.text


def test_inference_service_blocks_out_of_scope_question_without_calling_model():
    backend = FakeBackend()
    service = InferenceService(eval_config_path="configs/eval.yaml", backend=backend)
    result = service.answer("Kilode ti aye fi n yi ka oorun?")
    assert result.outcome == GuardrailOutcome.OUT_OF_SCOPE
    assert G1_REFUSAL_YO in result.text
    assert backend.calls == []


def test_inference_service_blocks_dosage_question_without_calling_model():
    backend = FakeBackend()
    service = InferenceService(eval_config_path="configs/eval.yaml", backend=backend)
    result = service.answer("Iwon oogun wo ni mo gbodo mu fun iba?")
    assert result.outcome == GuardrailOutcome.CLINICAL_BOUNDARY
    assert G2_REDIRECT_YO in result.text
    assert backend.calls == []


def test_inference_service_raises_clear_error_with_no_backend():
    service = InferenceService(eval_config_path="configs/eval.yaml")
    with pytest.raises(RuntimeError, match="no model backend loaded"):
        service.answer("Kí ni àwọn àmì àrùn ibà?")
