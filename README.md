# Yorùbá Healthcare Question-Answering System

Artefact repository for the MIT dissertation *Development and Evaluation of an
Instruction-Tuned Large Language Model for Yorùbá Healthcare Question
Answering*. This README documents how to reproduce every table and figure
Chapter 4 reports (definition-of-done: "README documents how to reproduce
every number").

## What is, and isn't, automated here

Per the build spec's division of labour:

| The pipeline automates | A human must do |
|---|---|
| Ingestion, claim segmentation, MT, normalisation | Supply source documents + confirm their licences |
| Post-edit/validation *spreadsheets* (export/import/checks) | The actual post-editing and clinical/linguistic validation |
| Blinded human-eval rating-sheet generation + all agreement statistics | Sit on the clinician/native-speaker panel and provide ratings |
| Training, inference, automatic evaluation, error-taxonomy tooling | GPU budget decisions; confirming/rejecting suggested error-type flags; final scientific claims |

No script in this repo fabricates a data point, a rating, or an error-type
label. Where a step needs human judgement, the corresponding script exports a
spreadsheet, stops, and only continues once a companion `*_import*` script
reads back a **completed** sheet (validated on import -- an incomplete or
malformed sheet is rejected with a specific error, never silently accepted).

## Environment

```bash
python -m venv .venv
# CPU-only subset -- enough to run Phase 1 dataset scripts, all tests, and
# scripts/07_fertility.py:
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt

# Full stack (GPU machine, CUDA 12.x, >=24GB VRAM) -- needed for
# 03_translate.py (or run it on requirements-dev.txt + torch/transformers/
# sentencepiece only, CPU works but is slow), 08_adapt.py, 09_train.py,
# 10_evaluate.py's B1/B2/B3/M systems:
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY` if you use the
LLM-backed question-generation backend (`02_build_qa.py --question-backend
llm`) or the B4 proprietary baseline in `10_evaluate.py`.

Run the test suite: `.venv/Scripts/python.exe -m pytest tests/ -v` (all pure
logic -- yoruba_text, guardrails, metrics, agreement stats, dataset
splitting/dedup, human-eval blinding -- runs with `requirements-dev.txt`
only, no GPU, no network).

## Reproducing Chapter 4

Run `scripts/run_pipeline.sh` (or step through it manually below). It runs
every fully-automatic stage and stops with explicit instructions at each
point that needs a human, then picks back up on re-run.

### 4.2 Dataset construction (Tables 4.1-4.2)

```bash
python scripts/01_ingest.py                 # refuses any source with no recorded licence
python scripts/02_build_qa.py               # claim segmentation + question generation
python scripts/02b_import_elicited.py --csv data/raw/elicited_questions.csv   # optional
python scripts/03_translate.py              # NLLB MT, content-hash cached
python scripts/03b_terminology.py --extract # -> data/terminology.csv; researcher fills canonical_yo
python scripts/03b_terminology.py --check --terminology data/terminology.csv --in data/interim/04b_import_postedit.jsonl
python scripts/04_export_for_postedit.py
# --- human: post-edit data/interim/postedit_batch_01.xlsx ---
python scripts/04b_import_postedit.py --edited data/interim/postedit_batch_01.xlsx --working-jsonl data/interim/03_translate.jsonl
python scripts/05_export_validation.py --kind clinical --out data/interim/validation_clinical.xlsx
python scripts/05_export_validation.py --kind linguistic --out data/interim/validation_linguistic.xlsx
# --- human: clinical + linguistic validation ---
python scripts/05b_import_validation.py --edited data/interim/validation_clinical.xlsx \
    --working-jsonl data/interim/04b_import_postedit.jsonl --kind clinical \
    --out data/interim/05b_clinical_validated.jsonl --correction-queue data/interim/correction_queue_clinical.jsonl
python scripts/05b_import_validation.py --edited data/interim/validation_linguistic.xlsx \
    --working-jsonl data/interim/04b_import_postedit.jsonl --kind linguistic \
    --out data/interim/05b_linguistic_validated.jsonl --correction-queue data/interim/correction_queue_linguistic.jsonl
```

**Merging validation passes.** `06_finalise_dataset.py` requires records that
passed BOTH clinical and linguistic validation on the same file
(`validated_clin` and `validated_ling` both `true`). Merge the two outputs by
`id` (a record with `validated_clin=true` from the clinical pass and
`validated_ling=true` from the linguistic pass wins on that field):

```bash
python - <<'PY'
import json
clin = {json.loads(l)["id"]: json.loads(l) for l in open("data/interim/05b_clinical_validated.jsonl", encoding="utf-8")}
ling = {json.loads(l)["id"]: json.loads(l) for l in open("data/interim/05b_linguistic_validated.jsonl", encoding="utf-8")}
merged = []
for rid in set(clin) | set(ling):
    rec = {**ling.get(rid, {}), **clin.get(rid, {})}
    rec["validated_clin"] = clin.get(rid, {}).get("validated_clin", False)
    rec["validated_ling"] = ling.get(rid, {}).get("validated_ling", False)
    merged.append(rec)
with open("data/interim/05b_all_validated.jsonl", "w", encoding="utf-8") as f:
    for r in merged:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
PY

python scripts/06_finalise_dataset.py --in data/interim/05b_all_validated.jsonl
```

Emits `reports/table_4_1_dataset_composition.{csv,md}` and
`reports/table_4_2_validation_outcomes.{csv,md}`, plus
`data/final/{train,val,test}.jsonl` with **zero `claim_id` overlap across
splits**, asserted in `src/data/dedup_and_split.py:split_by_claim_group` --
the build fails loudly (an `AssertionError`) if this is ever violated.

### 4.3 Base model selection (Table 4.3, Fig 4.1)

```bash
python scripts/07_fertility.py --yoruba-sample <real Yoruba sample .txt> --english-sample <real English sample .txt>
```

Then set `configs/model.yaml:selected_base_model` by hand from the results --
this script never auto-selects.

### 4.4 Training (Table 4.4, Fig 4.2) -- GPU required

```bash
python scripts/08_adapt.py                # optional; both an adapted and a
                                            # raw run of 09_train.py must be
                                            # evaluated in 10_evaluate.py, per spec
python scripts/09_train.py --single-run    # Table 3.6 "initial" hyperparameters
python scripts/09_train.py --sweep         # full grid, selected on val chrF++
```

### 4.5 Prototype (Fig 4.5)

```bash
export YHQA_BASE_MODEL=...      # your selected base model
export YHQA_ADAPTER_PATH=...    # the trained LoRA adapter
python app/app.py
python app/screenshot.py        # -> reports/fig_4_5_prototype.png
```

### 4.6 Automatic evaluation (Tables 4.5-4.6, Fig 4.3)

```bash
for SYS in B1 B2 B3 B4 M; do
  python scripts/10_evaluate.py --generate $SYS --base-model ... --adapter-path ...
done
python scripts/10_evaluate.py --score
```

### 4.7 Human evaluation (Tables 4.7-4.9, Fig 4.4)

```bash
python scripts/11_human_eval_export.py --n-raters 3
# --- human: raters complete reports/human_eval/rater_*.xlsx ---
python scripts/11b_human_eval_import.py
```

`reports/.blinding_key.json` is written alongside the rater sheets and is
gitignored -- never send it to a rater.

### 4.8 Error analysis (Table 4.10, Fig 4.6)

```bash
python scripts/12_error_analysis.py --export --n 100
# --- human: reviewer confirms/rejects SUGGESTED flags, fills in
#     factual_error / terminological_error / refusal_failure / hallucination ---
python scripts/12_error_analysis.py --aggregate
```

`--aggregate` refuses to run (loud error) if any `SUGGESTED` cell was never
resolved by a reviewer -- an unconfirmed heuristic never silently becomes a
reported number.

## Repository layout

See `BUILD_SPEC_yoruba_healthqa.md` for the full specification this
repository implements. `src/` holds all reusable logic (thoroughly unit
tested in `tests/`); `scripts/` are thin CLI entry points over it; every run
writes a manifest to `runs/<timestamp>_<stage>/manifest.json` with the git
commit, config, seed, package versions, and duration, so every number in
Chapter 4 is traceable to one specific run.

## Known limitations of this build

- **`src/serve/guardrails.py` is a keyword-based scope/boundary check**, not
  a learned classifier -- by design (build spec 3.7.4: guardrails must not be
  left to learned behaviour). Extend `HEALTH_KEYWORDS_NODIA` /
  `CLINICAL_BOUNDARY_KEYWORDS_NODIA` from real query logs as they accumulate.
- **`02_build_qa.py`'s default question-generation backend is template-based**
  (six clinical-content categories + a generic fallback), not an LLM. Pass
  `--question-backend llm` (needs `ANTHROPIC_API_KEY`) for more natural
  phrasing; it falls back to the template automatically on any API error.
- **Near-duplicate removal and the safety filter are threshold/keyword-based**
  (`configs/data.yaml`); tune `dedup.similarity_threshold` and
  `safety_filter_terms` against real data before trusting the discard counts
  in `table_4_2`.
