#!/usr/bin/env python
"""Phase 6 -- Automatic evaluation across all systems (build spec section 8).

Two sub-stages, run separately so generation (slow, needs each system's
model/API) and scoring (fast, pure metrics) can be iterated on independently:

  --generate SYSTEM_ID: generate answers for the held-out test set from one
      system (B1/B2/B3/B4/M as configured in configs/eval.yaml) and write
      reports/generations/<SYSTEM_ID>.jsonl. Each system has its own generator
      function below -- add a new one and register it in GENERATORS to add a
      system.
  --score: once reports/generations/<id>.jsonl exists for every system in
      configs/eval.yaml, computes chrF++ (primary), BLEU, ROUGE-L, BERTScore,
      AfriCOMET (if configured), diacritic accuracy, and language consistency
      for each system against the test set references, paired-bootstraps
      every baseline against M, and writes table_4_5/4_6 + fig_4_3.

This script never fabricates a system's output: --generate fails loudly if
the system's required model/API/checkpoint is not configured, and --score
fails loudly if a system's generation file is missing.

Usage:
    python scripts/10_evaluate.py --generate M --test-jsonl data/final/test.jsonl \
        --adapter-path models/single_run
    python scripts/10_evaluate.py --score --test-jsonl data/final/test.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import read_jsonl, write_jsonl
from src.data.yoruba_text import diacritic_accuracy
from src.utils.logging import MissingInputError, get_logger
from src.utils.manifest import RunManifest
from src.utils.seeding import seed_everything

logger = get_logger("10_evaluate")


# --- per-system generators ----------------------------------------------------

def generate_finetuned(records: list[dict], eval_cfg: dict, sys_cfg: dict, args) -> list[str]:
    from src.serve.inference import ModelBackend
    from src.model.prompt import build_prompt

    adapter_path = args.adapter_path or sys_cfg.get("adapter_path")
    if not adapter_path:
        raise ValueError("system M requires --adapter-path (or configs/eval.yaml systems.M.adapter_path)")
    base_model = args.base_model
    if not base_model:
        raise ValueError("system M requires --base-model")
    backend = ModelBackend(base_model, adapter_path=adapter_path)
    return [backend.generate(r["instruction"], eval_cfg["decoding"]) for r in records]


def generate_base_zero_shot(records: list[dict], eval_cfg: dict, sys_cfg: dict, args) -> list[str]:
    from src.serve.inference import ModelBackend

    base_model = args.base_model
    if not base_model:
        raise ValueError("system B1 requires --base-model")
    backend = ModelBackend(base_model, adapter_path=None)
    return [backend.generate(r["instruction"], eval_cfg["decoding"]) for r in records]


def generate_base_few_shot(records: list[dict], eval_cfg: dict, sys_cfg: dict, args) -> list[str]:
    from src.model.prompt import SYSTEM_PROMPT_YO, ASSISTANT_TAG, USER_TAG
    from src.serve.inference import ModelBackend
    from src.data.schema import read_jsonl as _read

    train_path = Path(args.train_jsonl) if args.train_jsonl else None
    if not train_path or not train_path.exists():
        raise MissingInputError(str(train_path or "--train-jsonl"), stage="10_evaluate",
                                 hint="system B2 needs --train-jsonl to draw few-shot exemplars from")
    train_records = list(_read(train_path))[: sys_cfg.get("n_shots", 4)]
    shots = "\n\n".join(
        f"{USER_TAG} {r['instruction']}\n{ASSISTANT_TAG} {r['output']}" for r in train_records
    )
    base_model = args.base_model
    if not base_model:
        raise ValueError("system B2 requires --base-model")
    backend = ModelBackend(base_model, adapter_path=None)

    outputs = []
    for r in records:
        prompt = f"{SYSTEM_PROMPT_YO}\n\n{shots}\n\n{USER_TAG} {r['instruction']}\n{ASSISTANT_TAG} "
        import torch

        inputs = backend.tokenizer(prompt, return_tensors="pt").to(backend.model.device)
        with torch.no_grad():
            generated = backend.model.generate(**inputs, max_new_tokens=eval_cfg["decoding"]["max_new_tokens"],
                                                do_sample=False)
        text = backend.tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(text.strip())
    return outputs


def generate_translate_pivot(records: list[dict], eval_cfg: dict, sys_cfg: dict, args) -> list[str]:
    from src.model.translate import NllbTranslator

    english_model = sys_cfg.get("english_model")
    if not english_model:
        raise ValueError("system B3 requires configs/eval.yaml systems.B3.english_model to be set")
    mt = NllbTranslator(model_name=sys_cfg["mt_model"])
    questions_en = mt.translate_batch([r["instruction"] for r in records])  # yo -> en (reverse direction model call)

    from src.serve.inference import ModelBackend

    backend = ModelBackend(english_model, adapter_path=None)
    answers_en = [backend.generate(q, eval_cfg["decoding"]) for q in questions_en]
    answers_yo = mt.translate_batch(answers_en)
    return answers_yo


def generate_proprietary(records: list[dict], eval_cfg: dict, sys_cfg: dict, args) -> list[str]:
    provider = sys_cfg.get("provider")
    if provider != "anthropic":
        raise ValueError(f"system B4: only provider 'anthropic' is implemented; "
                          f"configs/eval.yaml systems.B4.provider={provider!r}")
    import anthropic

    from src.model.prompt import SYSTEM_PROMPT_YO

    client = anthropic.Anthropic()
    model_name = sys_cfg.get("model_name")
    if not model_name:
        raise ValueError("system B4 requires configs/eval.yaml systems.B4.model_name to be set")

    outputs = []
    for r in records:
        response = client.messages.create(
            model=model_name, max_tokens=eval_cfg["decoding"]["max_new_tokens"],
            system=SYSTEM_PROMPT_YO, messages=[{"role": "user", "content": r["instruction"]}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        outputs.append(text.strip())
    return outputs


GENERATORS = {
    "B1": generate_base_zero_shot,
    "B2": generate_base_few_shot,
    "B3": generate_translate_pivot,
    "B4": generate_proprietary,
    "M": generate_finetuned,
}


def cmd_generate(args, eval_cfg: dict) -> None:
    system_id = args.generate
    if system_id not in GENERATORS:
        raise ValueError(f"unknown system '{system_id}', expected one of {list(GENERATORS)}")
    sys_cfg = eval_cfg["systems"][system_id]

    test_path = Path(args.test_jsonl)
    if not test_path.exists():
        raise MissingInputError(str(test_path), stage="10_evaluate")
    records = list(read_jsonl(test_path))

    seed = seed_everything(eval_cfg.get("seed", 42))
    with RunManifest(stage=f"10_evaluate_generate_{system_id}", config={"system": system_id}, seed=seed) as run:
        logger.info(f"[{system_id}] generating {len(records)} answers ({sys_cfg['description']})")
        outputs = GENERATORS[system_id](records, eval_cfg, sys_cfg, args)
        out_path = Path(args.generations_dir) / f"{system_id}.jsonl"
        rows = [{"id": r["id"], "instruction": r["instruction"], "reference": r["output"], "hypothesis": h}
                for r, h in zip(records, outputs)]
        write_jsonl(rows, out_path)
        run.record_output(str(out_path))
        logger.info(f"[{system_id}] wrote {len(rows)} generations -> {out_path}")


def cmd_score(args, eval_cfg: dict) -> None:
    from src.eval.metrics import (
        corpus_africomet, corpus_bertscore, corpus_bleu, corpus_chrf, corpus_rouge_l,
        language_consistency, paired_bootstrap, sentence_chrf_scores,
    )

    test_path = Path(args.test_jsonl)
    if not test_path.exists():
        raise MissingInputError(str(test_path), stage="10_evaluate")
    test_records = {r["id"]: r for r in read_jsonl(test_path)}

    generations = {}
    for system_id in eval_cfg["systems"]:
        gen_path = Path(args.generations_dir) / f"{system_id}.jsonl"
        if not gen_path.exists():
            raise MissingInputError(str(gen_path), stage="10_evaluate",
                                     hint=f"run --generate {system_id} first")
        rows = list(read_jsonl(gen_path))
        missing = set(test_records) - {r["id"] for r in rows}
        if missing:
            raise ValueError(f"{gen_path} is missing generations for {len(missing)} test ids -- regenerate")
        generations[system_id] = {r["id"]: r for r in rows}

    seed = seed_everything(eval_cfg.get("seed", 42))
    test_ids = sorted(test_records)
    refs = [test_records[i]["output"] for i in test_ids]

    import pandas as pd

    with RunManifest(stage="10_evaluate_score", config={"n_test": len(test_ids)}, seed=seed) as run:
        metrics_rows = []
        per_sentence_chrf: dict[str, list[float]] = {}
        for system_id in eval_cfg["systems"]:
            hyps = [generations[system_id][i]["hypothesis"] for i in test_ids]
            sent_chrf = sentence_chrf_scores(hyps, refs)
            per_sentence_chrf[system_id] = sent_chrf
            bleu_score, bleu_sig = corpus_bleu(hyps, refs)
            bertscore = corpus_bertscore(hyps, refs)
            africomet = corpus_africomet(hyps, refs, [test_records[i]["instruction"] for i in test_ids],
                                          eval_cfg.get("africomet", {}).get("checkpoint_path"))
            row = {
                "system": system_id,
                "description": eval_cfg["systems"][system_id]["description"],
                "chrf++": sum(sent_chrf) / len(sent_chrf),
                "bleu": bleu_score,
                "bleu_signature": bleu_sig,
                "rouge_l": corpus_rouge_l(hyps, refs),
                "bertscore_f1": bertscore if bertscore is not None else "SKIPPED (bert-score not installed)",
                "africomet": africomet if africomet is not None else "SKIPPED (no checkpoint configured)",
                "diacritic_accuracy": sum(diacritic_accuracy(h, r) for h, r in zip(hyps, refs)) / len(hyps),
                "language_consistency": language_consistency(hyps),
            }
            metrics_rows.append(row)
            logger.info(f"[{system_id}] chrF++={row['chrf++']:.2f} BLEU={row['bleu']:.2f} "
                        f"ROUGE-L={row['rouge_l']:.3f} diacritic_acc={row['diacritic_accuracy']:.3f} "
                        f"lang_consistency={row['language_consistency']:.3f}")

        metrics_df = pd.DataFrame(metrics_rows)
        reports_dir = Path(args.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        csv_p = reports_dir / "table_4_5_automatic_metrics.csv"
        md_p = reports_dir / "table_4_5_automatic_metrics.md"
        metrics_df.to_csv(csv_p, index=False)
        with open(md_p, "w", encoding="utf-8") as f:
            f.write(metrics_df.to_markdown(index=False))
        run.record_output(str(csv_p))
        run.record_output(str(md_p))

        # --- significance: every baseline vs M, on chrF++ ---
        sig_cfg = eval_cfg["significance"]
        compare_against = sig_cfg["compare_against"]
        if compare_against not in per_sentence_chrf:
            raise ValueError(f"significance.compare_against={compare_against!r} has no generations")
        sig_rows = []
        for system_id in eval_cfg["systems"]:
            if system_id == compare_against:
                continue
            result = paired_bootstrap(per_sentence_chrf[system_id], per_sentence_chrf[compare_against],
                                       n_resamples=sig_cfg["n_resamples"], ci=sig_cfg["ci"], seed=seed)
            sig_rows.append({
                "baseline": system_id, "compared_to": compare_against, "metric": "chrf++",
                "mean_diff": result.mean_diff, "ci_low": result.ci_low, "ci_high": result.ci_high,
                "p_value": result.p_value, "effect_size_cohens_d": result.effect_size,
                "n_resamples": result.n_resamples,
            })
        sig_df = pd.DataFrame(sig_rows)
        csv_p = reports_dir / "table_4_6_significance.csv"
        md_p = reports_dir / "table_4_6_significance.md"
        sig_df.to_csv(csv_p, index=False)
        with open(md_p, "w", encoding="utf-8") as f:
            f.write(sig_df.to_markdown(index=False))
        run.record_output(str(csv_p))
        run.record_output(str(md_p))

        # --- fig_4_3: metric comparison bar chart ---
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
            ax.bar(metrics_df["system"], metrics_df["chrf++"])
            ax.set_ylabel("chrF++")
            ax.set_title("Automatic evaluation: chrF++ by system")
            fig.tight_layout()
            fig_path = reports_dir / "fig_4_3_metric_comparison.png"
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            run.record_output(str(fig_path))
        except ImportError:
            logger.warning("matplotlib not installed -- skipped fig_4_3_metric_comparison.png")

        logger.info("evaluation scoring complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--test-jsonl", default="data/final/test.jsonl")
    parser.add_argument("--train-jsonl", default="data/final/train.jsonl", help="used by system B2 (few-shot)")
    parser.add_argument("--base-model", default=None, help="required for B1/B2/M")
    parser.add_argument("--adapter-path", default=None, help="LoRA adapter path for system M")
    parser.add_argument("--generations-dir", default="reports/generations")
    parser.add_argument("--reports-dir", default="reports")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", choices=list(GENERATORS), default=None)
    mode.add_argument("--score", action="store_true")
    args = parser.parse_args()

    import yaml

    eval_config_path = Path(args.eval_config)
    if not eval_config_path.exists():
        raise MissingInputError(str(eval_config_path), stage="10_evaluate")
    with open(eval_config_path, encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    if args.generate:
        cmd_generate(args, eval_cfg)
    else:
        cmd_score(args, eval_cfg)


if __name__ == "__main__":
    main()
