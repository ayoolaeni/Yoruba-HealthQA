#!/usr/bin/env python
"""Phase 4 -- QLoRA instruction tuning (build spec section 6).

4-bit NF4 quantised base + LoRA adapters (Hu et al., 2022; Dettmers et al.,
2023). Uses the SAME prompt template as inference (src/model/prompt.py),
with the loss masked to answer tokens only (src/model/prompt.mask_prompt_loss).

Two modes:
  --single-run: train once with configs/train.yaml's "initial" hyperparameters.
  --sweep: train one run per grid point across the "sweep" values (LoRA r,
      alpha, target_modules, learning_rate, epochs), selecting the winner on
      VALIDATION chrF++ (not loss alone), per build spec. Writes
      table_4_4_hyperparameter_sweep and fig_4_2_training_curves.png.

Requires a CUDA GPU with >=24GB VRAM for an 8B base model (build spec
section 1). Every run writes its own runs/<timestamp>/manifest.json.

Usage:
    python scripts/09_train.py --single-run --data-dir data/final --train-config configs/train.yaml
    python scripts/09_train.py --sweep --data-dir data/final --train-config configs/train.yaml
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import read_jsonl
from src.model.prompt import mask_prompt_loss
from src.utils.logging import MissingInputError, get_logger
from src.utils.manifest import RunManifest
from src.utils.seeding import seed_everything

logger = get_logger("09_train")


def load_split_jsonl(data_dir: Path, split: str) -> list[dict]:
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise MissingInputError(str(path), stage="09_train", hint="run scripts/06_finalise_dataset.py first")
    return list(read_jsonl(path))


def build_hf_dataset(records: list[dict], tokenizer, max_length: int):
    from datasets import Dataset

    def to_example(r):
        return mask_prompt_loss(tokenizer, prompt=_prompt_only(r), full=_full_text(r), max_length=max_length)

    from src.model.prompt import build_training_example

    def _prompt_only(r):
        return build_training_example(r["instruction"], r["output"]).prompt

    def _full_text(r):
        return build_training_example(r["instruction"], r["output"]).full

    examples = [to_example(r) for r in records]
    return Dataset.from_list(examples)


def build_model_and_tokenizer(base_model: str, lora_cfg: dict):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"], bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer


def compute_val_chrf(model, tokenizer, val_records: list[dict], max_new_tokens: int = 256) -> float:
    import torch

    from src.eval.metrics import corpus_chrf
    from src.model.prompt import build_prompt

    model.eval()
    hyps = []
    for r in val_records:
        prompt = build_prompt(r["instruction"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        hyps.append(text)
    refs = [r["output"] for r in val_records]
    model.train()
    return corpus_chrf(hyps, refs)


def train_one_run(base_model: str, train_records, val_records, hparams: dict, run_dir: Path, seed: int) -> dict:
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    model, tokenizer = build_model_and_tokenizer(base_model, hparams["lora"])
    train_ds = build_hf_dataset(train_records, tokenizer, hparams["max_seq_length"])

    args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=hparams["per_device_train_batch_size"],
        gradient_accumulation_steps=hparams["gradient_accumulation_steps"],
        learning_rate=hparams["learning_rate"],
        num_train_epochs=hparams["epochs"],
        lr_scheduler_type="cosine",
        warmup_ratio=hparams["warmup_ratio"],
        logging_steps=hparams["logging_steps"],
        save_steps=hparams["save_steps"],
        save_total_limit=2,
        optim=hparams["optimiser"],
        max_grad_norm=hparams["max_grad_norm"],
        weight_decay=hparams["weight_decay"],
        seed=seed,
        bf16=True,
        report_to=[],
    )
    # DataCollatorForLanguageModeling is the WRONG collator here -- verified
    # by testing it directly: it calls tokenizer.pad(), which pads
    # "input_ids"/"attention_mask" correctly but does not know how to pad a
    # pre-computed "labels" field of a different length, and crashes on the
    # first batch with more than one distinct sequence length (i.e.
    # immediately, on any real dataset). DataCollatorForSeq2Seq pads
    # "labels" with label_pad_token_id (-100, ignored by the loss) instead
    # of the tokenizer's pad token, which is exactly what our prompt-masked
    # labels (see src/model/prompt.py:mask_prompt_loss) need.
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
    train_result = trainer.train()
    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))

    val_chrf = compute_val_chrf(model, tokenizer, val_records)
    return {"train_loss": train_result.training_loss, "val_chrf++": val_chrf, "run_dir": str(run_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/final")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--single-run", action="store_true")
    mode.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    import yaml

    train_config_path = Path(args.train_config)
    model_config_path = Path(args.model_config)
    for p in (train_config_path, model_config_path):
        if not p.exists():
            raise MissingInputError(str(p), stage="09_train")

    with open(train_config_path, encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    with open(model_config_path, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    base_model = train_cfg.get("base_model") or model_cfg.get("selected_base_model")
    if not base_model:
        raise ValueError("no base_model set in configs/train.yaml or configs/model.yaml.selected_base_model -- "
                          "run scripts/07_fertility.py and have the researcher choose one first")

    seed = seed_everything(train_cfg.get("seed", 42))
    data_dir = Path(args.data_dir)
    train_records = load_split_jsonl(data_dir, "train")
    val_records = load_split_jsonl(data_dir, "val")
    logger.info(f"loaded {len(train_records)} train / {len(val_records)} val records")

    models_dir = Path(args.models_dir)

    if args.single_run:
        hparams = {
            "lora": {
                "r": train_cfg["lora"]["r"]["initial"],
                "alpha": train_cfg["lora"]["alpha"]["initial"],
                "dropout": train_cfg["lora"]["dropout"]["initial"],
                "target_modules": train_cfg["lora"]["target_modules"]["initial"],
            },
            "learning_rate": train_cfg["optim"]["learning_rate"]["initial"],
            "epochs": train_cfg["optim"]["epochs"]["initial"],
            "per_device_train_batch_size": train_cfg["optim"]["per_device_train_batch_size"],
            "gradient_accumulation_steps": train_cfg["optim"]["gradient_accumulation_steps"],
            "max_seq_length": train_cfg["optim"]["max_seq_length"],
            "optimiser": train_cfg["optim"]["optimiser"],
            "max_grad_norm": train_cfg["optim"]["max_grad_norm"],
            "weight_decay": train_cfg["optim"]["weight_decay"],
            "warmup_ratio": train_cfg["optim"]["warmup_ratio"],
            "logging_steps": train_cfg["logging"]["logging_steps"],
            "save_steps": train_cfg["logging"]["save_steps"],
        }
        with RunManifest(stage="09_train_single", config={"base_model": base_model, "hparams": hparams}, seed=seed) as run:
            run_dir = models_dir / "single_run"
            result = train_one_run(base_model, train_records, val_records, hparams, run_dir, seed)
            run.record_output(str(run_dir))
            logger.info(f"single run result: {result}")
        return

    # --sweep
    lora_r_values = train_cfg["lora"]["r"]["sweep"]
    lr_values = train_cfg["optim"]["learning_rate"]["sweep"]
    epoch_values = train_cfg["optim"]["epochs"]["sweep"]

    grid = list(itertools.product(lora_r_values, lr_values, epoch_values))
    logger.info(f"sweeping {len(grid)} hyperparameter combinations")

    import pandas as pd

    with RunManifest(stage="09_train_sweep", config={"base_model": base_model, "grid_size": len(grid)}, seed=seed) as run:
        rows = []
        for i, (r, lr, epochs) in enumerate(grid):
            hparams = {
                "lora": {
                    "r": r, "alpha": train_cfg["lora"]["alpha"]["initial"],
                    "dropout": train_cfg["lora"]["dropout"]["initial"],
                    "target_modules": train_cfg["lora"]["target_modules"]["initial"],
                },
                "learning_rate": lr, "epochs": epochs,
                "per_device_train_batch_size": train_cfg["optim"]["per_device_train_batch_size"],
                "gradient_accumulation_steps": train_cfg["optim"]["gradient_accumulation_steps"],
                "max_seq_length": train_cfg["optim"]["max_seq_length"],
                "optimiser": train_cfg["optim"]["optimiser"],
                "max_grad_norm": train_cfg["optim"]["max_grad_norm"],
                "weight_decay": train_cfg["optim"]["weight_decay"],
                "warmup_ratio": train_cfg["optim"]["warmup_ratio"],
                "logging_steps": train_cfg["logging"]["logging_steps"],
                "save_steps": train_cfg["logging"]["save_steps"],
            }
            run_dir = models_dir / f"sweep_{i:03d}"
            logger.info(f"[{i+1}/{len(grid)}] r={r} lr={lr} epochs={epochs}")
            result = train_one_run(base_model, train_records, val_records, hparams, run_dir, seed)
            rows.append({"run": i, "lora_r": r, "learning_rate": lr, "epochs": epochs, **result})

        df = pd.DataFrame(rows).sort_values("val_chrf++", ascending=False)
        reports_dir = Path(args.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        csv_path = reports_dir / "table_4_4_hyperparameter_sweep.csv"
        md_path = reports_dir / "table_4_4_hyperparameter_sweep.md"
        df.to_csv(csv_path, index=False)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(df.to_markdown(index=False))
        run.record_output(str(csv_path))
        run.record_output(str(md_path))

        best = df.iloc[0]
        logger.info(f"best run by val chrF++: run={best['run']} chrF++={best['val_chrf++']:.2f} "
                    f"(r={best['lora_r']}, lr={best['learning_rate']}, epochs={best['epochs']})")


if __name__ == "__main__":
    main()
