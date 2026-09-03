#!/usr/bin/env bash

set -euo pipefail

device="${1:-auto}"
python_bin="${PYTHON_BIN:-python}"
mmbert_model="${MMBERT_MODEL:-jhu-clsp/mmBERT-base}"
xlmr_model="${XLMR_MODEL:-FacebookAI/xlm-roberta-base}"

for seed in 42 1337; do
  output_dir="artifacts/experiments/mmbert-bilou"
  if [[ "$seed" == "1337" ]]; then
    output_dir="artifacts/experiments/mmbert-bilou-seed1337"
  fi
  "$python_bin" -m baseline.train \
    --train data/train.jsonl \
    --dev data/dev.jsonl \
    --output-dir "$output_dir" \
    --model-name "$mmbert_model" \
    --epochs 1 \
    --batch-size 8 \
    --learning-rate 5e-5 \
    --max-length 256 \
    --stride 64 \
    --seed "$seed" \
    --tag-scheme bilou \
    --mask-partial-window-entities \
    --attn-implementation sdpa \
    --eval-attn-implementation eager \
    --device "$device"
done

"$python_bin" -m baseline.train \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --output-dir artifacts/experiments/xlm-roberta-base \
  --model-name "$xlmr_model" \
  --epochs 3 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --learning-rate 3e-5 \
  --max-length 256 \
  --stride 64 \
  --seed 42 \
  --device "$device"

"$python_bin" -m statistical.crf \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --output-dir artifacts/statistical/crf \
  --fixed-confidence-threshold 0

echo "Final base models are ready. Run scripts/run_final_inference.sh next."
