"""Tokeniser fertility analysis for Phase 2 (build spec section 4).

Fertility = mean number of subword tokens per whitespace-delimited word.
Computed separately for a Yoruba sample and a comparable English sample so
the ratio (yoruba_fertility / english_fertility) is directly comparable
across candidate base models -- a higher ratio means the tokeniser
fragments Yoruba more than it fragments English, a proxy for how much
under-represented Yoruba was in the tokeniser's training data.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FertilityResult:
    model_name: str
    n_words: int
    n_tokens: int
    fertility: float  # n_tokens / n_words


def compute_fertility(tokenizer, text: str, model_name: str = "") -> FertilityResult:
    words = text.split()
    if not words:
        raise ValueError("compute_fertility requires non-empty text")
    n_tokens = 0
    for word in words:
        n_tokens += len(tokenizer.encode(word, add_special_tokens=False))
    return FertilityResult(model_name=model_name, n_words=len(words), n_tokens=n_tokens,
                            fertility=n_tokens / len(words))


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)
