#!/usr/bin/env bash

set -euo pipefail

input_path="${1:-data/dev.jsonl}"
output_path="${2:-artifacts/final/predictions.jsonl}"
device="${3:-auto}"
python_bin="${PYTHON_BIN:-python}"
work_dir="${FINAL_WORK_DIR:-artifacts/final/sources}"

mkdir -p "$work_dir" "$(dirname "$output_path")"

"$python_bin" -m baseline.predict \
  --model-dir artifacts/experiments/mmbert-bilou/model \
  --input "$input_path" \
  --output "$work_dir/seed42_viterbi.jsonl" \
  --decoder viterbi \
  --secondary-output "$work_dir/seed42_argmax.jsonl" \
  --secondary-decoder argmax \
  --batch-size 16 \
  --device "$device"

"$python_bin" -m baseline.predict \
  --model-dir artifacts/experiments/mmbert-bilou-seed1337/model \
  --input "$input_path" \
  --output "$work_dir/seed1337_viterbi.jsonl" \
  --decoder viterbi \
  --secondary-output "$work_dir/seed1337_argmax.jsonl" \
  --secondary-decoder argmax \
  --batch-size 16 \
  --device "$device"

"$python_bin" -m baseline.predict \
  --model-dir artifacts/experiments/xlm-roberta-base/model \
  --input "$input_path" \
  --output "$work_dir/xlmr.jsonl" \
  --batch-size 16 \
  --device "$device"

"$python_bin" -m statistical.predict \
  --model-dir artifacts/statistical/crf \
  --input "$input_path" \
  --output "$work_dir/crf.jsonl" \
  --min-confidence 0

"$python_bin" -m statistical.predict_lexicon \
  --train data/train.jsonl \
  --input "$input_path" \
  --config final/lexicon_configs.json \
  --variant suffix \
  --output "$work_dir/suffix_lexicon.jsonl"

"$python_bin" -m statistical.predict_lexicon \
  --train data/train.jsonl \
  --input "$input_path" \
  --config final/lexicon_configs.json \
  --variant exact \
  --output "$work_dir/exact_lexicon.jsonl"

"$python_bin" scripts/apply_span_stacker.py \
  --bundle final/stacker.joblib \
  --train data/train.jsonl \
  --input "$input_path" \
  --predictions \
    "$work_dir/seed42_viterbi.jsonl" \
    "$work_dir/seed42_argmax.jsonl" \
    "$work_dir/seed1337_viterbi.jsonl" \
    "$work_dir/seed1337_argmax.jsonl" \
    "$work_dir/xlmr.jsonl" \
    "$work_dir/crf.jsonl" \
    "$work_dir/suffix_lexicon.jsonl" \
    "$work_dir/exact_lexicon.jsonl" \
  --output "$output_path"

echo "Final predictions: $output_path"
