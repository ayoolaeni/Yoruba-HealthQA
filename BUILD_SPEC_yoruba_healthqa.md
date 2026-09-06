# BUILD SPEC — Yorùbá Healthcare Question-Answering System

**Hand this whole file to your coding agent as the first message.**

---

## 0. Context you must read before writing any code

This repository implements the artefact for a Master of Information Technology dissertation:
*Development and Evaluation of an Instruction-Tuned Large Language Model for Yorùbá Healthcare
Question Answering.* Chapter 3 of the dissertation has already specified the methodology. Your job
is to implement that specification exactly, and to emit the tables and figures that Chapter 4 will
report.

### Non-negotiable rules

1. **Never invent data.** No synthetic "example" results, no placeholder metrics, no fabricated
   dataset rows committed as if they were real. If a step cannot run because an input is missing,
   fail loudly with a clear error naming the missing file. A fabricated number in this repository
   becomes academic misconduct in the dissertation.
2. **Everything is seeded and logged.** Every script sets and records a random seed. Every run
   writes a JSON manifest to `runs/<timestamp>/` containing config, git commit hash, package
   versions, seed, and wall-clock duration.
3. **No metric is computed on data the model trained on.** Splitting happens once, early, and is
   frozen. Assert this in code.
4. **Every output artefact is a dissertation artefact.** Scripts write tables as CSV *and* as
   Markdown, and figures as 300-dpi PNG, into `reports/`. Name them after the thesis object they
   become (e.g. `table_4_3_automatic_metrics.csv`).
5. **Ask before assuming.** If this spec is ambiguous on something that would change results,
   stop and ask rather than guessing.

### Division of labour — what you CANNOT do

You are building the machinery. The human researcher supplies the human judgement:

| The agent builds | The researcher supplies |
|---|---|
| Ingestion, translation, normalisation pipelines | Source documents and licence checks |
| Post-editing and validation *tooling* (export/import) | The actual post-editing and clinical validation |
| Blinded rating sheet generator and agreement stats | The clinician panel and their ratings |
| Training, inference, evaluation code | GPU budget decisions, final scientific claims |

Do not attempt to substitute an LLM for the clinical validator or the human evaluation panel. If
you are tempted to auto-generate ratings "for testing", write them to `data/dev_fixtures/` with
`FIXTURE_DO_NOT_REPORT` in every filename.

---

## 1. Environment

- Python 3.11
- Single NVIDIA GPU, ≥24 GB VRAM (A100 40 GB ideal; 4-bit QLoRA on an 8B model fits in 24 GB)
- CUDA 12.x

```
torch, transformers, peft, bitsandbytes, accelerate, datasets, trl
sacrebleu, rouge-score, bert-score, unbabel-comet
pandas, numpy, scipy, statsmodels, krippendorff, matplotlib
gradio, pyyaml, tqdm, python-dotenv, rich
```

Pin every version in `requirements.txt` and record `pip freeze` in each run manifest.

---

## 2. Repository structure

```
yoruba-healthqa/
├── README.md
├── requirements.txt
├── configs/
│   ├── data.yaml            # sources, topics, split ratios
│   ├── model.yaml           # base model candidates, adaptation settings
│   ├── train.yaml           # LoRA + optimiser hyperparameters
│   └── eval.yaml            # metrics, baselines, decoding params
├── src/
│   ├── data/                # phases 1
│   ├── model/               # phases 2–4
│   ├── serve/               # phase 5
│   ├── eval/                # phases 6–7
│   └── utils/               # seeding, logging, manifests, yoruba text utils
├── scripts/                 # thin CLI entry points, one per pipeline stage
├── app/                     # phase 8 Gradio prototype
├── data/
│   ├── raw/ interim/ validated/ final/ dev_fixtures/
├── models/                  # adapters + adapted checkpoints (gitignored)
├── runs/                    # per-run manifests, logs, checkpoints
├── reports/                 # tables + figures for Chapter 4
└── tests/
```

Every stage is a separate CLI script so it can be re-run independently. Make the whole pipeline
reproducible via `make all` or a single `scripts/run_pipeline.sh`.

---

## 3. Phase 1 — Dataset construction

Implements Chapter 3, Section 3.6. Target: **1,500–3,000 validated QA pairs.** Do not aim higher;
the methodology explicitly prefers curated over large.

### 1.1 Ingest (`scripts/01_ingest.py`)
Pull consumer health content from the sources listed in `configs/data.yaml` (WHO fact sheets,
NCDC advisories, FMoH/NPHCDA materials, MedlinePlus, the AfriMed-QA consumer-query subset).
Record per-document: URL, retrieval date, licence, SHA-256 of the raw text. Refuse to ingest a
source with no recorded licence. Output `data/raw/<source>/*.json`.

### 1.2 Segment and construct questions (`scripts/02_build_qa.py`)
Split documents into atomic claims. Generate consumer-style English questions per claim. Tag each
record with a `topic` from the fixed topic list in `configs/data.yaml` (malaria, typhoid, TB,
HIV, hypertension, diabetes, maternal health, child health, immunisation, nutrition, WASH, …).
Assign a `claim_id` — **this is the unit that splitting will later respect.**

Also provide `scripts/02b_import_elicited.py` to load researcher-collected Yorùbá questions from a
CSV and match them to source-grounded answers, marking unmatched ones `question_type=out_of_scope`.

### 1.3 Translate (`scripts/03_translate.py`)
Machine-translate English → Yorùbá using an NLLB-derived model. Batch, cache by content hash,
and never re-translate an unchanged string. Output goes to `data/interim/`, flagged
`post_edited: false`.

### 1.4 Terminology list (`scripts/03b_terminology.py`)
Extract recurring clinical terms, emit `data/terminology.csv` for the researcher to fix Yorùbá
renderings **before** post-editing starts. Once returned, enforce it: a checker flags any record
using a non-canonical rendering of a listed term.

### 1.5 Post-editing round-trip (`scripts/04_export_for_postedit.py` / `04b_import_postedit.py`)
Export to spreadsheet with columns: `id | topic | question_en | answer_en | question_yo_mt |
answer_yo_mt | question_yo_final | answer_yo_final | editor_id | notes`. Import validates that
required columns are filled and characters are legal Yorùbá.

### 1.6 Normalisation (`src/data/yoruba_text.py`)
- Unicode **NFC** normalisation on all Yorùbá text.
- Character-set validator against the permitted Yorùbá set (18 consonants incl. `gb`, `ṣ`;
  7 oral vowels incl. `ẹ`, `ọ`; nasal vowels; tone marks; syllabic nasals `ḿ`, `ń`).
- `strip_diacritics()` to produce `instruction_nodia`.
- `diacritic_accuracy(hyp, ref)` — used later in evaluation.
Unit-test these on a hand-written fixture file. They are the most error-prone code in the repo.

### 1.7 Validation tooling (`scripts/05_export_validation.py` / `05b_import_validation.py`)
Two separate exports: **linguistic** (native speakers) and **clinical** (health workers).
- Clinical: 100% of records.
- Linguistic: 100% if capacity allows, else a stratified random sample of ≥30%.
- Double-annotate ≥10% for agreement; compute **Cohen's kappa**, interpret per Landis & Koch (1977).
Failed records route to a correction queue and are re-checked. Log the discard rate.

### 1.8 Filter and split (`scripts/06_finalise_dataset.py`)
- Near-duplicate removal via character n-gram similarity (threshold in config; report how many removed).
- Safety filter: records containing dosage / prescription / emergency-procedure content are
  **removed from the answer set and rewritten as refusal examples**.
- Split 80/10/10, **stratified by topic, grouped by `claim_id`**. Assert zero `claim_id` overlap
  across splits and fail the build if violated.

### Record schema (JSONL) — matches Table 3.5 of the dissertation
```json
{
  "id": "yhqa-000123",
  "claim_id": "who-malaria-007",
  "instruction": "Kí ni àwọn àmì àrùn ibà?",
  "instruction_nodia": "Ki ni awon ami arun iba?",
  "input": "",
  "output": "<diacritised Yorùbá answer>",
  "topic": "malaria",
  "question_type": "derived | elicited | out_of_scope",
  "source_en": "<English source text>",
  "source_ref": "WHO Malaria Fact Sheet, 2024",
  "validated_ling": true,
  "validated_clin": true
}
```

**Emits:** `table_4_1_dataset_composition` (records per topic, per split, per question type),
`table_4_2_validation_outcomes` (validated / corrected / discarded counts, kappa).

---

## 4. Phase 2 — Tokeniser fertility analysis (`scripts/07_fertility.py`)

For each candidate base model, compute mean subword tokens per word on a common Yorùbá sample and
on comparable English text. Report both, plus the ratio.

**Emits:** `table_4_3_tokeniser_fertility`, `fig_4_1_fertility.png`.

This drives base-model selection. Candidates: open-weight instruction-capable decoder-only models
≤8B with documented multilingual coverage and a licence permitting redistribution of derived
weights. Record the licence of each in the table.

---

## 5. Phase 3 — Language adaptation (optional, `scripts/08_adapt.py`)

Continued pre-training on monolingual Yorùbá text mixed with high-quality English educational text
(per Buzaaba et al., 2025). Mixture ratio is a config parameter.

**This stage must be evaluated, not assumed.** Train two instruction-tuned models — one on the
adapted checkpoint, one on the raw base — and report both. If adaptation does not help, that is a
publishable negative finding; report it rather than hiding it.

---

## 6. Phase 4 — Instruction tuning (`scripts/09_train.py`)

QLoRA: 4-bit NF4 quantised base, LoRA adapters (Hu et al., 2022; Dettmers et al., 2023).

Prompt template — **identical in training and inference**, defined once in `src/model/prompt.py`:
a Yorùbá system instruction, the user question, then the answer. **Mask the loss so it is computed
on answer tokens only.**

Start from Table 3.6 of the dissertation:

| Parameter | Initial | Sweep |
|---|---|---|
| Quantisation | 4-bit NF4 | fixed |
| LoRA rank r | 16 | 8, 16, 32, 64 |
| LoRA alpha | 32 | yes |
| LoRA dropout | 0.05 | fixed |
| Target modules | attention + FFN projections | yes |
| Learning rate | 2e-4 | 1e-4, 2e-4, 5e-4 |
| Scheduler | cosine + warmup | fixed |
| Epochs | 3 | 2–5 |
| Effective batch | 16 (grad accum) | fixed |
| Max seq length | 1024 | fixed |
| Optimiser | paged AdamW 8-bit | fixed |
| Early stopping | on val loss | fixed |

Sweep against **validation chrF++**, not loss alone. Log every run to `runs/`.

**Emits:** `table_4_4_hyperparameter_sweep`, `fig_4_2_training_curves.png`.

---

## 7. Phase 5 — Inference service and guardrails (`src/serve/`)

Three guardrails, in the **service layer**, not left to learned behaviour (Chapter 3, §3.7.4):

- **G1 scope check** — question outside the health domain → Yorùbá refusal (FR4).
- **G2 clinical-boundary check** — diagnosis / dosage / prescription request → Yorùbá redirection
  to a qualified professional (FR5).
- **G3 disclaimer** — fixed Yorùbá referral disclaimer appended to every answer (FR6).

Each guardrail is independently unit-tested with positive and negative cases. Decoding parameters
are fixed in `configs/eval.yaml` and reported.

---

## 8. Phase 6 — Automatic evaluation (`scripts/10_evaluate.py`)

Generate answers for the held-out test set from all systems:

| ID | System |
|---|---|
| B1 | Base model, zero-shot, Yorùbá prompt |
| B2 | Base model, few-shot in-context |
| B3 | Translate → English model → translate back |
| B4 | Strong proprietary model, zero-shot Yorùbá |
| **M** | **This study's fine-tuned model** |

Metrics: **chrF++ (primary)**, BLEU via sacreBLEU with reported signature (Post, 2018), ROUGE-L,
BERTScore, AfriCOMET if a checkpoint is available. Plus the two task-specific measures:

- **Diacritic accuracy** — proportion of generated tokens whose diacritics match the reference.
- **Language consistency** — proportion of tokens identified as Yorùbá rather than English.

Significance: **paired bootstrap resampling** (≥1000 samples) for every pairwise comparison against
M. Report effect sizes and 95% CIs alongside p-values.

**Emits:** `table_4_5_automatic_metrics`, `table_4_6_significance`, `fig_4_3_metric_comparison.png`,
and per-system generation files in `reports/generations/`.

---

## 9. Phase 7 — Human evaluation tooling (`scripts/11_human_eval_*.py`)

- **Export**: stratified random sample of ≥150 test questions; for each, all system responses in
  **randomised order with system identity concealed**. One sheet per rater. Store the unblinding
  key separately in `reports/.blinding_key.json` (gitignored).
- Rubric per Table 3.8: factual correctness, completeness, linguistic fluency, diacritic accuracy,
  cultural appropriateness (all 5-point Likert) + **potential for harm** (None/Low/Moderate/Severe,
  categorical, never averaged).
- **Import & analyse**: unblind, aggregate, compute **Fleiss' kappa** (harm) and
  **Krippendorff's alpha** (ordinal dimensions). Non-parametric test for related samples across
  systems.
- **Harm report**: list every Moderate/Severe response *individually* with question, answer and
  rater comment. This is a headline result, not an appendix.

**Emits:** `table_4_7_human_ratings`, `table_4_8_agreement`, `table_4_9_harm_incidents`,
`fig_4_4_human_ratings.png`.

---

## 10. Phase 8 — Prototype (`app/`)

Gradio app: Yorùbá text box in, answer out, guardrails active, disclaimer shown, mobile-friendly
and light on bandwidth (NFR7). Logs interactions with **no personally identifying information**
(FR8). Include a screenshot script for `fig_4_5_prototype.png`.

---

## 11. Phase 9 — Error analysis (`scripts/12_error_analysis.py`)

Classify the lowest-scoring responses into: factual error, omission, terminological error,
diacritic error, code-switching, disfluency, refusal failure, hallucination. Report distribution
by topic and by error type.

**Emits:** `table_4_10_error_taxonomy`, `fig_4_6_errors_by_topic.png`.

---

## 12. Chapter 4 outline this repository must support

| Section | Content | Fed by |
|---|---|---|
| 4.1 | Introduction | — |
| 4.2 | Dataset construction results | Tables 4.1–4.2 |
| 4.3 | Base model selection | Table 4.3, Fig 4.1 |
| 4.4 | Training implementation | Table 4.4, Fig 4.2 |
| 4.5 | Prototype implementation | Fig 4.5 |
| 4.6 | Automatic evaluation results | Tables 4.5–4.6, Fig 4.3 |
| 4.7 | Human evaluation results | Tables 4.7–4.9, Fig 4.4 |
| 4.8 | Error analysis | Table 4.10, Fig 4.6 |
| 4.9 | Discussion | all of the above |

Write 4.9 yourself, in prose, once the numbers exist. It must answer: did native-language tuning
beat the translate-test pipeline (B3)? How far below the proprietary baseline (B4) are we, and
does that gap matter for the use case? Did language adaptation help? Where does the system remain
unsafe?

---

## 13. Definition of done

- [ ] `scripts/run_pipeline.sh` runs end to end on a fresh clone given raw inputs
- [ ] Zero `claim_id` overlap across splits, asserted in code
- [ ] Every table and figure in §12 exists in `reports/`
- [ ] Every result traceable to a `runs/<timestamp>/manifest.json`
- [ ] Guardrails G1–G3 have passing unit tests
- [ ] `src/data/yoruba_text.py` has passing tests on a hand-written fixture
- [ ] No fixture or placeholder data outside `data/dev_fixtures/`
- [ ] README documents how to reproduce every number

---

## 14. Suggested order of work

Build **Phase 1 skeleton → Phase 4 → Phase 6** first on a seed set of ~200 pairs. This is
Increment 1 from the dissertation: prove the pipeline runs end to end and produces scores before
investing weeks in data. Only then scale the dataset (Increment 2), then add guardrails, UI and
human evaluation (Increment 3).
