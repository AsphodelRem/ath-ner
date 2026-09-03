#!/bin/bash
#SBATCH --job-name=ner-aug
#SBATCH --partition=a100
#SBATCH --error="/userspace/kms/Code/ner-uz/logs/aug_%A_%a.err"
#SBATCH --output="/userspace/kms/Code/ner-uz/logs/aug_%A_%a.log"
#SBATCH --open-mode=append
#SBATCH --time=08:00:00
#SBATCH --array=0-2%1
#
# ==========================================================================
# Аугментации: регистр, подмена сущностей и обе вместе.
#
#   sbatch --nodelist=ngpu06 tools/slurm/augment.sbatch
#   sbatch --array=1 tools/slurm/augment.sbatch          # только подмена
#
# Контроль — уже посчитанный artifacts/glot500-s42 (micro F1 0.8853):
# конфигурация здесь та же, отличаются только флаги аугментации.
#
# Ставим на Glot500, а не на large: разброс по сидам у Glot500 0,01-0,02
# пункта, у large до 0,88 — там эффект утонет в шуме.
#
# Перед запуском скопировать на кластер два файла:
#   scp solution/augment.py solution/train.py <хост>:<проект>/solution/
# и проверить: python3 -m solution.augment
# ==========================================================================

# ---------------------------------------------------------------- переменные
project_root="/userspace/kms/Code/ner-uz"
miniconda_dir="/userspace/kms/miniconda3"
conda_env="ner"
cache_dir="/userspace/kms/.cache"
model="/userspace/kms/models/glot500-base"
logs_dir="${project_root}/logs"
log_file="${logs_dir}/aug_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

mkdir -p "${logs_dir}"

# ------------------------------------------------------------------ окружение
source "${miniconda_dir}/etc/profile.d/conda.sh"
conda activate "${conda_env}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="${cache_dir}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${cache_dir}/torch_cache"
export XDG_CACHE_HOME="${cache_dir}/xdg_cache"
export DO_NOT_TRACK=1
export ANONYMIZED_TELEMETRY=False
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${project_root}"

# ------------------------------------------------------------- конфигурации
# имя | флаги аугментации
EXPERIMENTS=(
  "glot500-aug-case|--augment-case 0.3"
  "glot500-aug-swap|--augment-swap 0.3 --augment-swap-share 0.5"
  "glot500-aug-both|--augment-case 0.3 --augment-swap 0.3 --augment-swap-share 0.5"
)
IFS='|' read -r RUN_NAME AUG <<< "${EXPERIMENTS[${SLURM_ARRAY_TASK_ID}]}"

echo "===================================================" >> "${log_file}"
echo "Задача ${SLURM_ARRAY_TASK_ID}: ${RUN_NAME}" >> "${log_file}"
echo "Аугментация: ${AUG}" >> "${log_file}"
echo "Старт: $(date)" >> "${log_file}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> "${log_file}" 2>&1
echo "===================================================" >> "${log_file}"

if [ -f "artifacts/${RUN_NAME}/dev_metrics.json" ]; then
  echo "=== ${RUN_NAME} уже завершён, пропускаю ===" >> "${log_file}"
  exit 0
fi

# Веса и корректность аугментаций проверяем до обучения: нода изолирована,
# а сдвиг координат сломал бы метрику молча.
python3 -c "
from transformers import AutoConfig, AutoTokenizer
import sys
try:
    AutoConfig.from_pretrained('${model}')
    t = AutoTokenizer.from_pretrained('${model}', use_fast=True)
except OSError as error:
    sys.exit(f'нет весов в ${model}: {type(error).__name__}')
if not t.is_fast:
    sys.exit('нужен fast tokenizer')
print(f'веса на месте, словарь {t.vocab_size}')
" >> "${log_file}" 2>&1 || { echo "=== Остановлено: веса ===" >> "${log_file}"; exit 1; }

python3 -m solution.augment >> "${log_file}" 2>&1 \
  || { echo "=== Остановлено: аугментации не прошли самопроверку ===" >> "${log_file}"; exit 1; }

# ------------------------------------------------------------------- обучение
# ${AUG} намеренно без кавычек: нужно разбиение на отдельные аргументы.
echo "=== Обучение ${RUN_NAME} ===" >> "${log_file}"
python3 -m solution.train \
    --model-name "${model}" \
    --tag-scheme bilou --max-length 512 --stride 128 \
    --no-gradient-checkpointing \
    --viterbi --transition-weight 0.25 --transition-mode conditional \
    ${AUG} \
    --batch-size 32 --gradient-accumulation-steps 1 \
    --learning-rate 3e-5 --epochs 10 --patience 3 --eval-batch-size 32 \
    --seed 42 \
    --comet offline --comet-project uzbek-ner \
    --comet-tag augment --comet-tag "${RUN_NAME}" \
    --output-dir "artifacts/${RUN_NAME}" >> "${log_file}" 2>&1
status=$?
echo "=== ${RUN_NAME} завершён, код ${status} ===" >> "${log_file}"
grep -a "Лучший результат" "${log_file}" | tail -1 >> "${log_file}"

# ---------------------------------------------------------------------- итоги
# Общий F1 здесь недостаточен: разбор ошибок предсказывает рост ORG и GEO
# при возможном проседании NAME, и это надо видеть отдельно.
echo "=== Сводка ===" >> "${log_file}"
python3 tools/summarize_runs.py artifacts >> "${log_file}" 2>&1

if [ -f "artifacts/${RUN_NAME}/dev_predictions.jsonl" ] \
   && [ -f artifacts/glot500-s42/dev_predictions.jsonl ]; then
  echo "=== Срезы: ${RUN_NAME} против контроля glot500-s42 ===" >> "${log_file}"
  python3 tools/evaluate_slices.py \
      --predictions artifacts/glot500-s42/dev_predictions.jsonl \
                    "artifacts/${RUN_NAME}/dev_predictions.jsonl" >> "${log_file}" 2>&1
  echo "=== F1 по классам ===" >> "${log_file}"
  python3 -c "
import json, sys
for name in ('glot500-s42', '${RUN_NAME}'):
    m = json.load(open(f'artifacts/{name}/dev_metrics.json'))
    by = m['by_label']
    print(f\"  {name:<22} micro {m['micro']['f1']:.4f} | \"
          + '  '.join(f'{k} {by[k][\"f1\"]:.4f}' for k in ('ORG','NAME','GEO')))
" >> "${log_file}" 2>&1
fi

echo "Финиш: $(date)" >> "${log_file}"
