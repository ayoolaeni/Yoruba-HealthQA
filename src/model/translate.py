"""NLLB-derived English -> Yoruba machine translation, content-hash cached.

Used by scripts/03_translate.py. Requires `transformers`/`torch` (heavy, GPU
recommended but CPU works for pilot-scale batches) -- imported lazily so
importing this module for its cache helpers doesn't require them.

NLLB-200 language codes: English = "eng_Latn", Yoruba = "yor_Latn".
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TranslationCache:
    """A flat JSON file mapping sha256(source_text) -> translated_text, so an
    unchanged source string is never re-translated (build spec 1.3)."""

    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path)
        self._data: dict[str, str] = {}
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                self._data = json.load(f)

    def get(self, text: str) -> str | None:
        return self._data.get(content_hash(text))

    def put(self, text: str, translation: str) -> None:
        self._data[content_hash(text)] = translation

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def __len__(self) -> int:
        return len(self._data)


class NllbTranslator:
    """Thin wrapper around an NLLB-derived seq2seq checkpoint for
    English -> Yoruba batch translation."""

    SRC_LANG = "eng_Latn"
    TGT_LANG = "yor_Latn"

    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", device: str | None = None):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                f"{e}. NllbTranslator needs torch+transformers, which are in requirements.txt "
                f"but NOT in requirements-dev.txt (that subset is CPU-only, no-GPU tooling for "
                f"Phase 1 dataset scripts). Install requirements.txt on this machine, or run "
                f"scripts/03_translate.py on a GPU machine instead."
            ) from e

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=self.SRC_LANG)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def translate_batch(self, texts: list[str], max_new_tokens: int = 256, batch_size: int = 16) -> list[str]:
        import torch

        outputs: list[str] = []
        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(self.TGT_LANG)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_new_tokens=max_new_tokens,
                )
            outputs.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return outputs


def translate_with_cache(texts: list[str], translator: "NllbTranslator", cache: TranslationCache) -> list[str]:
    """Translate `texts`, skipping any whose sha256 is already in `cache`."""
    to_translate_idx = [i for i, t in enumerate(texts) if t and cache.get(t) is None]
    if to_translate_idx:
        fresh = translator.translate_batch([texts[i] for i in to_translate_idx])
        for i, translation in zip(to_translate_idx, fresh):
            cache.put(texts[i], translation)

    return [cache.get(t) if t else "" for t in texts]
