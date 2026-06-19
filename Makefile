.PHONY: demo demo-novel train eval test lint help

PYTHON := .venv/bin/python

help:
	@echo "SENTINEL — make targets"
	@echo ""
	@echo "  make demo        End-to-end triage on a known sample (offline)"
	@echo "  make demo-novel  Same demo + the novel-organism escalation path"
	@echo "  make train       Train + evaluate the XGBoost model"
	@echo "  make eval        Run the full evaluation (train if needed)"
	@echo "  make test        Run the pytest suite (8 tests)"
	@echo "  make lint        Ruff linter check"
	@echo ""
	@echo "First time: ./run.sh  (creates .venv, installs deps, trains, runs demo)"

demo: _need_model
	$(PYTHON) -m sentinel.demo

demo-novel: _need_model
	$(PYTHON) -m sentinel.demo --show-novel

train: _need_data
	$(PYTHON) -m sentinel.model
	$(PYTHON) -m sentinel.explain
	$(PYTHON) -m sentinel.novelty

eval: train
	@echo "--- Evaluation complete: see metrics/ for plots and metrics.json ---"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check sentinel/ tests/ || true

_need_data:
	@test -f data/sentinel_dataset.csv || \
	  (echo "Dataset missing — run: $(PYTHON) -m sentinel.fetch_data" && exit 1)

_need_model:
	@test -f models/xgb_model.json || \
	  (echo "Model not trained — run: make train" && exit 1)
