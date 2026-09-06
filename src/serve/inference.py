"""Phase 5 -- Inference service (build spec section 7).

Wires together, in order: G1 scope check -> G2 clinical-boundary check ->
model generation (only reached if both guardrails pass) -> G3 disclaimer.
Uses the SAME prompt template as training (src/model/prompt.py) and fixed
decoding parameters from configs/eval.yaml, so what is evaluated is exactly
what a user gets from app/app.py.

The model backend is loaded lazily and only on first real generation request,
so importing/instantiating InferenceService for guardrail-only testing (see
tests/test_guardrails.py) never requires a GPU or model weights.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from src.model.prompt import build_prompt
from src.serve.guardrails import GuardedResponse, guarded_response


class ModelBackend:
    """Loads a base model + optional LoRA adapter for generation. Requires
    torch/transformers/peft and, for a quantised base, bitsandbytes+CUDA."""

    def __init__(self, base_model: str, adapter_path: str | None = None, load_in_4bit: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {}
        if load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model
        self.model.eval()

    def generate(self, question: str, decoding: dict) -> str:
        import torch

        prompt = build_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=decoding.get("max_new_tokens", 256),
                do_sample=decoding.get("do_sample", False),
                temperature=decoding.get("temperature", 0.0) or None,
                num_beams=decoding.get("num_beams", 1),
                repetition_penalty=decoding.get("repetition_penalty", 1.0),
            )
        text = self.tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()


class InferenceService:
    def __init__(self, eval_config_path: str | Path = "configs/eval.yaml", backend: ModelBackend | None = None):
        with open(eval_config_path, encoding="utf-8") as f:
            eval_cfg = yaml.safe_load(f)
        self.decoding = eval_cfg["decoding"]
        self._backend = backend  # None until load_backend() or one is injected (e.g. in tests)

    def load_backend(self, base_model: str, adapter_path: str | None = None) -> None:
        self._backend = ModelBackend(base_model, adapter_path=adapter_path)

    def _generate(self, question: str) -> str:
        if self._backend is None:
            raise RuntimeError("InferenceService has no model backend loaded -- call load_backend() first, "
                                "or inject one via the constructor for testing")
        return self._backend.generate(question, self.decoding)

    def answer(self, question: str) -> GuardedResponse:
        return guarded_response(question, self._generate)
