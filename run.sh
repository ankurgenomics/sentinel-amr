#!/usr/bin/env bash
# One-command setup + full pipeline for SENTINEL.
# Usage: ./run.sh
set -euo pipefail

PY=python3.12
command -v uv >/dev/null 2>&1 || { echo "Please install uv (https://docs.astral.sh/uv/)"; exit 1; }

echo "[1/6] Create venv (Python 3.12)"
uv venv --python 3.12 .venv

echo "[2/6] Install dependencies"
uv pip install --python .venv -r requirements.txt

# macOS: XGBoost needs the OpenMP runtime.
if [[ "$(uname)" == "Darwin" ]]; then
  if ! ls /opt/homebrew/opt/libomp/lib/libomp.dylib >/dev/null 2>&1 \
     && ! ls /usr/local/opt/libomp/lib/libomp.dylib >/dev/null 2>&1; then
    echo "[note] installing libomp via brew (required by XGBoost on macOS)"
    brew install libomp || echo "[warn] brew install libomp failed; install it manually"
  fi
fi

echo "[3/6] Train + evaluate model (group-aware holdout)"
.venv/bin/python -m sentinel.model

echo "[4/6] SHAP explainability"
.venv/bin/python -m sentinel.explain

echo "[5/6] Calibrate novelty detector"
.venv/bin/python -m sentinel.novelty

echo "[6/6] End-to-end agent demo"
.venv/bin/python -m sentinel.demo --show-novel

echo
echo "Done. Reports in output/, plots + metrics in metrics/."
echo "Run tests with:  .venv/bin/python -m pytest -q"
