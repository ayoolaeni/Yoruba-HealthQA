"""The single source of truth for the instruction-tuning prompt template.

Build spec Phase 4: "Prompt template — identical in training and inference,
defined once in src/model/prompt.py." Both scripts/09_train.py and
src/serve/inference.py import from here; never inline the template elsewhere.

The template is deliberately simple and marker-based so that the boundary
between "prompt" (system + user turn) and "answer" (assistant turn) is
unambiguous — 09_train.py masks the loss so it is computed on answer tokens
only, which requires knowing exactly where the answer starts in the tokenised
sequence.
"""
from __future__ import annotations

from dataclasses import dataclass

SYSTEM_PROMPT_YO = (
    "Ìwọ ni Olùrànlọ́wọ́ Ìlera Yorùbá, ẹ̀rọ kan tí ó ń dáhùn ìbéèrè ìlera "
    "lédè Yorùbá nìkan fún àwọn aráàlú. Fún ìdáhùn tí ó ṣe kúkúrú, tí ó "
    "bọ́gbọ́n mu, tí ìjìnlẹ̀ rẹ̀ sì bá ìmọ̀ ìṣègùn tí a fìdí rẹ̀ múlẹ̀ mu. "
    "Ìwọ kì í ṣe dókítà: má ṣe fúnni ní àyẹ̀wò àrùn pàtó tàbí ìwọ̀n òògùn "
    "kankan; kàkà bẹ́ẹ̀, kí o gba aláìsàn níyànjú láti lọ sí ọ̀dọ̀ "
    "akọ́ṣẹ́mọṣẹ́ ìlera fún irú ọ̀ràn bẹ́ẹ̀."
)

USER_TAG = "### Ìbéèrè:"
ASSISTANT_TAG = "### Ìdáhùn:"


@dataclass
class PromptParts:
    prompt: str   # system + user turn + assistant tag, NO answer text
    full: str     # prompt + answer text (answer NOT yet including eos token)


def build_prompt(question: str, system_prompt: str = SYSTEM_PROMPT_YO) -> str:
    """Prompt text up to and including the assistant tag, with no answer.
    Used for inference (generation starts right after this) and for computing
    the loss-mask boundary during training."""
    return f"{system_prompt}\n\n{USER_TAG} {question.strip()}\n{ASSISTANT_TAG} "


def build_training_example(question: str, answer: str, system_prompt: str = SYSTEM_PROMPT_YO) -> PromptParts:
    """Full training example. Tokenise `.prompt` and `.full` separately (both
    with add_special_tokens consistent with your tokenizer's chat setup);
    the length difference in token ids marks where to start computing loss.
    See scripts/09_train.py:mask_prompt_loss for the exact masking code."""
    prompt = build_prompt(question, system_prompt=system_prompt)
    full = f"{prompt}{answer.strip()}"
    return PromptParts(prompt=prompt, full=full)


def mask_prompt_loss(tokenizer, prompt: str, full: str, max_length: int) -> dict:
    """Tokenise `full` and return input_ids/attention_mask/labels with the
    prompt span's labels set to -100 (ignored by the loss), so gradient only
    flows through answer tokens. Shared by 09_train.py's dataset collator and
    any future re-tuning script — keep masking logic in exactly one place.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
    full_enc = tokenizer(full, add_special_tokens=True, truncation=True, max_length=max_length)
    full_ids = full_enc["input_ids"]

    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = list(full_ids)
    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": full_ids,
        "attention_mask": full_enc["attention_mask"],
        "labels": labels,
    }
