
PYTHON ?= .venv/Scripts/python.exe

.PHONY: test venv install-dev ingest build-qa fertility clean-runs pipeline

venv:
	python -m venv .venv

install-dev: venv
	$(PYTHON) -m pip install -r requirements-dev.txt

install: venv
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

ingest:
	$(PYTHON) scripts/01_ingest.py

build-qa:
	$(PYTHON) scripts/02_build_qa.py

fertility:
	$(PYTHON) scripts/07_fertility.py --yoruba-sample data/raw/fertility_yo_sample.txt --english-sample data/raw/fertility_en_sample.txt

clean-runs:
	rm -rf runs/*

# Full pipeline (Increment 1: skeleton run on whatever raw data exists).
# Each stage after 04b requires a human in the loop (post-editing,
# validation) and is NOT run automatically here -- see scripts/run_pipeline.sh
# for the exact manual checkpoints.
pipeline: install-dev
	bash scripts/run_pipeline.sh
