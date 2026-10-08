#!/usr/bin/env python
"""Emergency CPU-only training fallback -- NOT part of the formal build spec
pipeline (that is scripts/09_train.py: 4-bit QLoRA, requires CUDA).

Written because on presentation day, no GPU was available and the two
GPU-trained candidates both failed for real, verified reasons:
  - Qwen2.5-1.5B-Instruct: fluent-looking but semantically empty in Yoruba
    (base model was never trained on real Yoruba text at scale).
  - bigscience/bloomz-560m zero-shot: produces genuine Yoruba words (its
    ROOTS-corpus pretraining did include some real Yoruba via Masakhane)
    but has no health-domain knowledge, so answers are irrelevant.

This script fine-tunes bloomz-560m -- the only candidate that showed any
real Yoruba fluency -- on the SAME data/final_demo split and the SAME
prompt template as the formal pipeline (src/model/prompt.py), but skips
4-bit quantisation (bitsandbytes' quantised kernels require CUDA; there is
no GPU here) and trains in plain fp32 on CPU instead. This is a genuine
LoRA fine-tune on genuine data -- just without the quantisation and without
CUDA, because those aren't available today.

Usage:
    python scripts/09b_train_cpu_fallback.py --max-steps 5   # timing probe
    python scripts/09b_train_cpu_fallback.py                 # full run
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import read_jsonl
from src.model.prompt import build_training_example, mask_prompt_loss
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

logger = get_logger("09b_train_cpu_fallback")

BASE_MODEL = "bigscience/bloomz-560m"


def build_hf_dataset(records, tokenizer, max_length: int):
    from datasets import Dataset

    def to_example(r):
        parts = build_training_example(r["instruction"], r["output"])
        return mask_prompt_loss(tokenizer, prompt=parts.prompt, full=parts.full, max_length=max_length)

    return Dataset.from_list([to_example(r) for r in records])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/final_demo")
    parser.add_argument("--run-dir", default="models/bloomz_cpu_adapter")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="cap steps for a quick timing probe")
    parser.add_argument("--max-seq-length", type=int, default=256)
    args = parser.parse_args()

    seed_everything(42)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

    data_dir = Path(args.data_dir)
    train_records = list(read_jsonl(data_dir / "train.jsonl"))
    logger.info(f"loaded {len(train_records)} train records from {data_dir}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)  # fp32, CPU -- no bnb quantisation available
    logger.info(f"loaded base model in {time.time() - t0:.1f}s")

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["query_key_value"],  # BLOOM's fused qkv projection name
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_ds = build_hf_dataset(train_records, tokenizer, args.max_seq_length)

    run_dir = Path(args.run_dir)
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=42,
        use_cpu=True,
    )
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, data_collator=collator)

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    logger.info(f"training finished in {elapsed:.1f}s -- train_loss={result.training_loss:.4f}")

    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))
    logger.info(f"adapter saved to {run_dir}")


if __name__ == "__main__":
    main()
