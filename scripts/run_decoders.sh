#!/bin/bash
# Reproduces all baseline results on whatever sessions are in data/raw/
# Usage: ./scripts/run_decoders.sh

set -euo pipefail

echo "==> Training Ridge..."
python scripts/train_ridge.py

echo "==> Training Kalman..."
python scripts/train_kalman.py

echo "==> Training GRU..."
python scripts/train_gru.py

echo "==> Training TCN..."
python scripts/train_tcn.py

echo "==> Training Transformer..."
python scripts/train_transformer.py

echo "==> Evaluating and generating plots..."
python scripts/evaluate.py

echo "==> Done. Results in results/, weights in saved_models/."