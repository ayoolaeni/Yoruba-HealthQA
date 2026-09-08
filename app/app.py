#!/usr/bin/env python
"""Phase 8 -- Gradio prototype (build spec section 10).

Yoruba text box in, answer out, guardrails (G1/G2/G3) always active,
disclaimer shown. Mobile-friendly, single-page, no heavy assets (NFR7).
Logs interactions with NO personally identifying information (FR8): each
line logged is {timestamp, question_sha256, guardrail_outcome,
response_char_len, latency_seconds} -- never the raw question/answer text,
never an IP address, never a session/device identifier.

Configure the model via environment variables (see .env.example):
    YHQA_BASE_MODEL      -- required to actually generate answers
    YHQA_ADAPTER_PATH     -- optional LoRA adapter (system M); omitted -> base model

Without YHQA_BASE_MODEL set, the app still starts and demonstrates the
guardrails (G1/G2 refusals work with no model loaded at all), but shows a
clear "model not configured" message instead of a generated answer.

Usage:
    python app/app.py
"""
from __future__ import annotations

import hashlib
import json
import os

# Must be set before `import gradio` anywhere in the process for Gradio to
# pick it up. Using the env var instead of launch(analytics_enabled=...) is
# also version-proof -- that launch() kwarg has been added, renamed, and
# removed across Gradio releases; this env var has not.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv

    load_dotenv()  # reads a .env file in the working directory, if present
except ImportError:
    pass  # python-dotenv not installed -- fine if config is passed via real env vars instead

from src.serve.guardrails import GuardrailOutcome
from src.serve.inference import InferenceService
from src.utils.logging import get_logger

logger = get_logger("app")

INTERACTION_LOG_PATH = Path("runs/app_interactions.log")


def log_interaction(question: str, outcome: GuardrailOutcome, response_text: str, latency_s: float) -> None:
    """Appends one JSONL line. No raw text, IP, or session identifier is
    ever written -- only a one-way hash of the question (lets you count
    repeat questions without recovering their content) plus coarse metadata."""
    INTERACTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question_sha256": hashlib.sha256(question.strip().encode("utf-8")).hexdigest(),
        "guardrail_outcome": outcome.value,
        "response_char_len": len(response_text),
        "latency_seconds": round(latency_s, 3),
    }
    with open(INTERACTION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_service() -> InferenceService:
    service = InferenceService(eval_config_path="configs/eval.yaml")
    base_model = os.environ.get("YHQA_BASE_MODEL")
    if base_model:
        adapter_path = os.environ.get("YHQA_ADAPTER_PATH") or None
        logger.info(f"loading model backend: base={base_model} adapter={adapter_path}")
        service.load_backend(base_model, adapter_path=adapter_path)
    else:
        logger.warning("YHQA_BASE_MODEL not set -- guardrails (G1/G2) work, but in-scope questions "
                        "will show a 'model not configured' message instead of a generated answer")
    return service


def make_answer_fn(service: InferenceService):
    def answer_fn(question: str) -> str:
        question = (question or "").strip()
        if not question:
            return "Jọ̀wọ́ tẹ ìbéèrè kan sí àyè náà."  # "please type a question"

        start = time.time()
        try:
            result = service.answer(question)
            text = result.text
            outcome = result.outcome
        except RuntimeError:
            # G3 (disclaimer) must appear on every answer, including this
            # "no model configured" placeholder -- apply_disclaimer() is the
            # same helper guarded_response() uses for real answers, so this
            # path is not a way to accidentally skip FR6.
            from src.serve.guardrails import apply_disclaimer

            text = apply_disclaimer(
                "Àdáṣe àpẹẹrẹ yìí kò tíì ní ẹ̀rọ ìdáhùn tí ó ń ṣiṣẹ́ (kò sí àwòṣe tí a ti gbé kalẹ̀). "
                "Àmọ́ àyẹ̀wò àyè-iṣẹ́ (guardrails) ń ṣiṣẹ́ dáadáa."
            )
            outcome = GuardrailOutcome.ANSWERED  # not a guardrail decision; the model backend is just absent
        latency = time.time() - start
        log_interaction(question, outcome, text, latency)
        return text

    return answer_fn


#  One tap each during a live demo covers all three guardrail paths
#  (G1 refusal, G2 redirection, G3 disclaimer-on-a-real-answer) without
#  relying on typing Yoruba correctly on stage. Each has been verified
#  against the real guardrail logic in tests/test_guardrails.py.
EXAMPLE_QUESTIONS = [
    ["Kí ni àwọn àmì àrùn ibà?"],           # in-scope health question
    ["Iwon oogun wo ni mo gbodo mu fun iba?"],  # G2: dosage/diagnosis boundary
    ["Kilode ti aye fi n yi ka oorun?"],    # G1: off-topic, out of scope
]


def build_demo():
    import gradio as gr

    service = build_service()
    answer_fn = make_answer_fn(service)
    model_loaded = os.environ.get("YHQA_BASE_MODEL") is not None
    status_text = (
        "🟢 Ẹ̀rọ ìdáhùn ti ń ṣiṣẹ́ — a máa dá ìdáhùn tòótọ́ padà."
        if model_loaded else
        "🟡 Ẹ̀rọ ìdáhùn kò tíì gbé kalẹ̀ (àpẹẹrẹ nìkan ni yìí) — àmọ́ gbogbo àyẹ̀wò ààbò ń ṣiṣẹ́ dáadáa."
    )

    with gr.Blocks(title="Yorùbá HealthQA") as demo:
        gr.Markdown("## Yorùbá HealthQA — Ìbéèrè Ìlera Rẹ Lédè Yorùbá")
        gr.Markdown(
            "Tẹ ìbéèrè rẹ nípa ìlera sí àyè yìí lédè Yorùbá. Ẹ̀rọ yìí kì í ṣe dókítà — "
            "kò sì rọ́pò ìmọ̀ràn ilé-ìwòsàn tàbí ti dókítà."
        )
        gr.Markdown(status_text)
        question_box = gr.Textbox(label="Ìbéèrè rẹ", placeholder="Kí ni àwọn àmì àrùn ibà?", lines=2)
        submit_btn = gr.Button("Fi ránṣẹ́", variant="primary")
        answer_box = gr.Textbox(label="Ìdáhùn", lines=8, interactive=False)

        gr.Examples(
            examples=EXAMPLE_QUESTIONS,
            inputs=question_box,
            label="Àpẹẹrẹ ìbéèrè (tẹ ọ̀kan láti dán an wò)",
        )

        submit_btn.click(fn=answer_fn, inputs=question_box, outputs=answer_box)
        question_box.submit(fn=answer_fn, inputs=question_box, outputs=answer_box)

    return demo


if __name__ == "__main__":
    import gradio as gr

    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                theme=gr.themes.Soft(),
                css=".gradio-container {max-width: 640px !important}")
