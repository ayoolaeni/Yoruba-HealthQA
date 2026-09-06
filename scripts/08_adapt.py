#!/usr/bin/env python
"""Phase 3 -- Continued pre-training (language adaptation), optional
(build spec section 5).

Mixes monolingual Yoruba text with high-quality English educational text
(mixture ratio from configs/model.yaml language_adaptation.mixture_ratio_yoruba,
per Buzaaba et al., 2025) and continues pre-training the selected base model
with a causal-LM objective (not the instruction-tuning objective -- that is
09_train.py). Requires a GPU; this is the most compute-heavy stage in the
pipeline.

IMPORTANT (build spec section 5): this stage must be EVALUATED, not assumed.
09_train.py is expected to be run TWICE -- once from this script's adapted
checkpoint, once from the raw base model -- and scripts/10_evaluate.py
reports both (B1-adapted vs B1-raw, etc.) so a negative result ("adaptation
did not help") is reported rather than hidden.

Usage:
    python scripts/08_adapt.py --model-config configs/model.yaml
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import MissingInputError, get_logger
from src.utils.manifest import RunManifest
from src.utils.seeding import seed_everything

logger = get_logger("08_adapt")


def load_mixed_corpus(yoruba_glob: str, english_glob: str, mixture_ratio_yoruba: float, seed: int) -> list[str]:
    import random

    yo_files = sorted(glob.glob(yoruba_glob))
    en_files = sorted(glob.glob(english_glob))
    if not yo_files:
        raise MissingInputError(yoruba_glob, stage="08_adapt",
                                 hint="place monolingual Yoruba .txt files matching this glob "
                                      "(see configs/model.yaml language_adaptation.yoruba_corpus_glob)")
    if not en_files:
        raise MissingInputError(english_glob, stage="08_adapt",
                                 hint="place English educational .txt files matching this glob "
                                      "(see configs/model.yaml language_adaptation.english_corpus_glob)")

    yo_lines = [line.strip() for f in yo_files for line in Path(f).read_text(encoding="utf-8").splitlines() if line.strip()]
    en_lines = [line.strip() for f in en_files for line in Path(f).read_text(encoding="utf-8").splitlines() if line.strip()]

    rng = random.Random(seed)
    rng.shuffle(yo_lines)
    rng.shuffle(en_lines)

    # Sample lines so the final mixture matches mixture_ratio_yoruba as closely
    # as whichever corpus is limiting allows, without repeating any line
    # (a single pass over each corpus, never more). The naive
    # "cap each corpus at ratio * (Ny + Ne)" approach silently collapses
    # toward 50/50 whenever one corpus is much smaller than the other, since
    # (Ny + Ne) is fixed regardless of the requested ratio -- instead, size
    # the *total* mixture as the largest total for which both per-language
    # line counts stay within their corpus, i.e.
    # total = min(Ny / ratio, Ne / (1 - ratio)).
    if mixture_ratio_yoruba <= 0:
        n_yo_target, n_en_target = 0, len(en_lines)
    elif mixture_ratio_yoruba >= 1:
        n_yo_target, n_en_target = len(yo_lines), 0
    else:
        total_target = min(len(yo_lines) / mixture_ratio_yoruba, len(en_lines) / (1 - mixture_ratio_yoruba))
        n_yo_target = min(len(yo_lines), round(total_target * mixture_ratio_yoruba))
        n_en_target = min(len(en_lines), round(total_target * (1 - mixture_ratio_yoruba)))

    mixed = yo_lines[:n_yo_target] + en_lines[:n_en_target]
    rng.shuffle(mixed)
    return mixed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--out-dir", default="models/adapted")
    args = parser.parse_args()

    import yaml

    model_config_path = Path(args.model_config)
    if not model_config_path.exists():
        raise MissingInputError(str(model_config_path), stage="08_adapt")
    with open(model_config_path, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    adapt_cfg = model_cfg["language_adaptation"]
    if not adapt_cfg.get("enabled", True):
        logger.info("language_adaptation.enabled is false in configs/model.yaml -- nothing to do")
        return

    base_model = model_cfg.get("selected_base_model")
    if not base_model:
        raise ValueError("configs/model.yaml selected_base_model is null -- run scripts/07_fertility.py "
                          "and have the researcher pick a base model first")

    seed = seed_everything(model_cfg.get("seed", 42))

    with RunManifest(stage="08_adapt", config={"base_model": base_model, "adapt_cfg": adapt_cfg}, seed=seed) as run:
        corpus_lines = load_mixed_corpus(
            adapt_cfg["yoruba_corpus_glob"], adapt_cfg["english_corpus_glob"],
            adapt_cfg["mixture_ratio_yoruba"], seed,
        )
        logger.info(f"mixed corpus: {len(corpus_lines)} lines "
                    f"(target Yoruba ratio {adapt_cfg['mixture_ratio_yoruba']:.0%})")

        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = Dataset.from_dict({"text": corpus_lines})

        def tokenize_fn(batch):
            return tokenizer(batch["text"], truncation=True, max_length=adapt_cfg["block_size"])

        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=str(out_dir),
            max_steps=adapt_cfg["max_steps"],
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            seed=seed,
            bf16=torch.cuda.is_available(),
            report_to=[],
        )
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, data_collator=collator)
        trainer.train()
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        run.record_output(str(out_dir))
        logger.info(f"saved adapted checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
