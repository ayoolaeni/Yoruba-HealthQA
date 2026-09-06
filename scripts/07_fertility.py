#!/usr/bin/env python
"""Phase 2 -- Tokeniser fertility analysis (build spec section 4).

For each candidate base model in configs/model.yaml, loads its tokenizer and
computes mean subword tokens per word on a common Yoruba sample and on
comparable English text, plus the ratio. This is what drives base-model
selection -- configs/model.yaml.selected_base_model must stay null until
this has actually run and produced numbers; do not hand-pick a base model.

Sample text: --yoruba-sample / --english-sample point at plain text files
(one sample per line or a single blob is fine -- fertility is computed over
whitespace-split words in the whole file). These must be real text the
researcher supplies (comparable register/length in both languages); this
script does not fabricate sample sentences.

Usage:
    python scripts/07_fertility.py --yoruba-sample data/raw/fertility_yo_sample.txt \
        --english-sample data/raw/fertility_en_sample.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.fertility import compute_fertility, load_tokenizer
from src.utils.logging import MissingInputError, get_logger
from src.utils.manifest import RunManifest
from src.utils.seeding import seed_everything

logger = get_logger("07_fertility")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yoruba-sample", required=True)
    parser.add_argument("--english-sample", required=True)
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    yo_path, en_path = Path(args.yoruba_sample), Path(args.english_sample)
    for p in (yo_path, en_path):
        if not p.exists():
            raise MissingInputError(str(p), stage="07_fertility",
                                     hint="supply a real Yoruba/English text sample -- these are not generated")

    model_config_path = Path(args.model_config)
    if not model_config_path.exists():
        raise MissingInputError(str(model_config_path), stage="07_fertility")
    with open(model_config_path, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    seed = seed_everything(model_cfg.get("seed", 42))
    yo_text = yo_path.read_text(encoding="utf-8")
    en_text = en_path.read_text(encoding="utf-8")

    with RunManifest(stage="07_fertility", config={"yoruba_sample": str(yo_path), "english_sample": str(en_path)},
                      seed=seed) as run:
        rows = []
        for candidate in model_cfg["candidates"]:
            name = candidate["name"]
            logger.info(f"loading tokenizer for {name}")
            try:
                tokenizer = load_tokenizer(name)
            except Exception as e:
                logger.error(f"failed to load tokenizer for {name}: {e}")
                continue
            yo_result = compute_fertility(tokenizer, yo_text, model_name=name)
            en_result = compute_fertility(tokenizer, en_text, model_name=name)
            rows.append({
                "model": name,
                "params_b": candidate.get("params_b"),
                "licence": candidate.get("licence"),
                "yoruba_words": yo_result.n_words,
                "yoruba_tokens": yo_result.n_tokens,
                "yoruba_fertility": round(yo_result.fertility, 4),
                "english_words": en_result.n_words,
                "english_tokens": en_result.n_tokens,
                "english_fertility": round(en_result.fertility, 4),
                "fertility_ratio_yo_over_en": round(yo_result.fertility / en_result.fertility, 4),
            })

        if not rows:
            raise RuntimeError("no candidate tokenizer loaded successfully -- check network access and model names")

        df = pd.DataFrame(rows).sort_values("fertility_ratio_yo_over_en")
        reports_dir = Path(args.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        csv_path = reports_dir / "table_4_3_tokeniser_fertility.csv"
        md_path = reports_dir / "table_4_3_tokeniser_fertility.md"
        df.to_csv(csv_path, index=False)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(df.to_markdown(index=False))
        run.record_output(str(csv_path))
        run.record_output(str(md_path))

        fig_path = reports_dir / "fig_4_1_fertility.png"
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
            x = range(len(df))
            width = 0.35
            ax.bar([i - width / 2 for i in x], df["yoruba_fertility"], width, label="Yoruba")
            ax.bar([i + width / 2 for i in x], df["english_fertility"], width, label="English")
            ax.set_xticks(list(x))
            ax.set_xticklabels(df["model"], rotation=30, ha="right")
            ax.set_ylabel("mean subword tokens per word")
            ax.set_title("Tokeniser fertility: Yoruba vs English")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            run.record_output(str(fig_path))
        except ImportError:
            logger.warning("matplotlib not installed -- skipped fig_4_1_fertility.png")

        logger.info(f"wrote fertility table for {len(rows)} candidate(s) -> {csv_path}")
        best = df.iloc[0]
        logger.info(f"lowest Yoruba/English fertility ratio: {best['model']} "
                    f"({best['fertility_ratio_yo_over_en']}) -- NOT auto-selected; "
                    f"researcher fills configs/model.yaml.selected_base_model")


if __name__ == "__main__":
    main()
