#!/usr/bin/env bash
# End-to-end pipeline orchestrator (build spec definition-of-done item 1).
#
# This script runs every FULLY AUTOMATIC stage back-to-back, and STOPS with
# clear instructions at every stage that genuinely requires a human (source
# licence checks, post-editing, clinical/linguistic validation, human
# evaluation rating, GPU training decisions) -- per the build spec's
# division of labour, this script does not simulate or fabricate those
# steps. Re-run it after completing each manual step; it picks up wherever
# the last completed stage's output already exists.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi

log() { echo "[run_pipeline] $*"; }
stop_for_human() {
  echo
  echo "=================================================================="
  echo "MANUAL STEP REQUIRED -- $1"
  echo "=================================================================="
  echo "$2"
  echo
  echo "Re-run scripts/run_pipeline.sh once this is done."
  exit 0
}

log "Phase 1.1: ingest"
"$PYTHON" scripts/01_ingest.py
if [ -z "$(find data/raw -name '*.json' 2>/dev/null)" ]; then
  stop_for_human "no sources with a recorded licence" \
    "Edit configs/data.yaml and record a licence for at least one source (ncdc_advisories, fmoh_nphcda, afrimed_qa are all licence: null out of the box), or supply cleared manual text under data/raw/<source>/manual/."
fi

log "Phase 1.2: build QA pairs from ingested claims"
"$PYTHON" scripts/02_build_qa.py

if [ -f "data/raw/elicited_questions.csv" ]; then
  log "Phase 1.2b: import researcher-elicited Yoruba questions"
  "$PYTHON" scripts/02b_import_elicited.py --csv data/raw/elicited_questions.csv
fi

if [ ! -f "data/interim/03_translate.jsonl" ]; then
  log "Phase 1.3: machine-translate to Yoruba (this can be slow on CPU)"
  if ! "$PYTHON" scripts/03_translate.py; then
    stop_for_human "torch/transformers not installed" \
      "scripts/03_translate.py needs the full requirements.txt (torch, transformers), not just requirements-dev.txt. Install it (ideally on a GPU machine -- CPU works but is slow) and re-run."
  fi
fi

if [ ! -f "data/terminology.csv" ] || ! grep -q '[^,]' <(cut -d, -f3 data/terminology.csv | tail -n +2); then
  "$PYTHON" scripts/03b_terminology.py --extract
  stop_for_human "clinical terminology needs canonical Yoruba renderings" \
    "Open data/terminology.csv and fill in the canonical_yo column for every clinical term before post-editing starts (build spec 1.4)."
fi

if [ ! -f "data/interim/postedit_batch_01.xlsx" ]; then
  log "Phase 1.5: export for post-editing"
  "$PYTHON" scripts/04_export_for_postedit.py
fi

if [ ! -f "data/interim/04b_import_postedit.jsonl" ]; then
  stop_for_human "post-editing" \
    "Have a Yoruba speaker complete data/interim/postedit_batch_01.xlsx (question_yo_final / answer_yo_final columns), then run:
    $PYTHON scripts/04b_import_postedit.py --edited data/interim/postedit_batch_01.xlsx --working-jsonl data/interim/03_translate.jsonl"
fi

if [ ! -f "data/interim/05b_clinical_validated.jsonl" ] || [ ! -f "data/interim/05b_linguistic_validated.jsonl" ]; then
  log "Phase 1.7: export validation sheets"
  "$PYTHON" scripts/05_export_validation.py --kind clinical --out data/interim/validation_clinical.xlsx
  "$PYTHON" scripts/05_export_validation.py --kind linguistic --out data/interim/validation_linguistic.xlsx
  stop_for_human "clinical and linguistic validation" \
    "Have health workers complete data/interim/validation_clinical.xlsx and native Yoruba speakers complete data/interim/validation_linguistic.xlsx, then run 05b_import_validation.py for each --kind (see the script's docstring)."
fi

log "Phase 1.8: finalise dataset (dedup, safety filter, split)"
# Merge the two validation passes: a record only reaches the final dataset
# if it passed BOTH (06_finalise_dataset.py checks validated_clin AND
# validated_ling on each record, so the two validation outputs must first be
# merged onto one working set -- see README for the merge step).
if [ ! -f "data/interim/05b_all_validated.jsonl" ]; then
  stop_for_human "merge clinical + linguistic validation results" \
    "See README.md 'Merging validation passes' for the one-liner that combines data/interim/05b_clinical_validated.jsonl and data/interim/05b_linguistic_validated.jsonl into data/interim/05b_all_validated.jsonl."
fi
"$PYTHON" scripts/06_finalise_dataset.py --in data/interim/05b_all_validated.jsonl

log "Phase 2: tokeniser fertility"
if [ -f "data/raw/fertility_yo_sample.txt" ] && [ -f "data/raw/fertility_en_sample.txt" ]; then
  "$PYTHON" scripts/07_fertility.py --yoruba-sample data/raw/fertility_yo_sample.txt --english-sample data/raw/fertility_en_sample.txt
else
  stop_for_human "fertility sample text" \
    "Supply data/raw/fertility_yo_sample.txt and data/raw/fertility_en_sample.txt (real, comparable-length Yoruba/English text), then re-run."
fi

stop_for_human "base model selection, then GPU training/evaluation" \
  "Review reports/table_4_3_tokeniser_fertility.csv and set configs/model.yaml selected_base_model. Then, on a GPU machine:
    $PYTHON scripts/08_adapt.py                 # optional language adaptation
    $PYTHON scripts/09_train.py --single-run    # or --sweep
    $PYTHON scripts/10_evaluate.py --generate <SYSTEM_ID> ...   # for each of B1/B2/B3/B4/M
    $PYTHON scripts/10_evaluate.py --score
    $PYTHON scripts/11_human_eval_export.py ... # then have raters complete the sheets
    $PYTHON scripts/11b_human_eval_import.py ...
    $PYTHON scripts/12_error_analysis.py --export ... # then have a reviewer complete the sheet
    $PYTHON scripts/12_error_analysis.py --aggregate ..."
