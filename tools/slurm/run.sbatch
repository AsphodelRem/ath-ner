#!/bin/bash
#SBATCH --job-name=ner
#SBATCH --partition=a100
#SBATCH --error="/userspace/kms/Code/ner-uz/logs/ner_%A_%a.err"
#SBATCH --output="/userspace/kms/Code/ner-uz/logs/ner_%A_%a.log"
#SBATCH --open-mode=append
#SBATCH --time=08:00:00
#SBATCH --array=0-8%1
#
# ==========================================================================
# Все эксперименты в одном файле. Группа выбирается диапазоном --array,
# так что разные группы можно поставить на разные карты:
#
#   sbatch --nodelist=ngpu06 --array=0-3%1 tools/slurm/run.sbatch   # энкодеры
#   sbatch --nodelist=ngpu07 --array=4-5%1 tools/slurm/run.sbatch   # ёмкость
#   sbatch --nodelist=ngpu08 --array=6-8%1 tools/slurm/run.sbatch   # данные
#
#   sbatch --array=1 tools/slurm/run.sbatch                # один прогон
#   sbatch tools/slurm/run.sbatch                          # всё подряд
# ---------------------------------------------------------------- переменные
project_root="/userspace/kms/Code/ner-uz"
miniconda_dir="/userspace/kms/miniconda3"
conda_env="ner"
cache_dir="/userspace/kms/.cache"
models_dir="/userspace/kms/models"
logs_dir="${project_root}/logs"
log_file="${logs_dir}/ner_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

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
# Лучшее на сегодня: large-s42 = 0.8987, glot500-s42 = 0.8853, xlmr-s42 = 0.8772.
BASE_ARGS=(--tag-scheme bilou --max-length 512 --stride 128
           --viterbi --transition-weight 0.25 --transition-mode conditional
           --patience 3 --eval-batch-size 32
           --comet offline --comet-project uzbek-ner)

# индекс | имя прогона | каталог модели | дополнительные аргументы
EXPERIMENTS=(
  "xlmv-s42|${models_dir}/xlm-v-base|--seed 42 --learning-rate 3e-5 --epochs 10 --batch-size 32 --gradient-accumulation-steps 1"
  "rembert-s42|${models_dir}/rembert|--seed 42 --learning-rate 1e-5 --epochs 12 --batch-size 16 --gradient-accumulation-steps 2"
  "labse-s42|${models_dir}/labse|--seed 42 --learning-rate 3e-5 --epochs 10 --batch-size 32 --gradient-accumulation-steps 1"
  "bgem3-s42|${models_dir}/bge-m3|--seed 42 --learning-rate 1e-5 --epochs 12 --batch-size 16 --gradient-accumulation-steps 2"
  "large-s43|${models_dir}/xlm-roberta-large|--seed 43 --learning-rate 1e-5 --epochs 12 --batch-size 32 --gradient-accumulation-steps 1"
  "large-s44|${models_dir}/xlm-roberta-large|--seed 44 --learning-rate 1e-5 --epochs 12 --batch-size 32 --gradient-accumulation-steps 1"
  "glot500-s44|${models_dir}/glot500-base|--seed 44 --learning-rate 3e-5 --epochs 10 --batch-size 32 --gradient-accumulation-steps 1"
  "glot500-argmax|${models_dir}/glot500-base|@NO_VITERBI --seed 42 --learning-rate 3e-5 --epochs 10 --batch-size 32 --gradient-accumulation-steps 1"
  "large-external|${models_dir}/xlm-roberta-large|@TWO_STAGE --seed 42 --learning-rate 1e-5 --epochs 12 --batch-size 32 --gradient-accumulation-steps 1"
)

IFS='|' read -r RUN_NAME MODEL EXTRA <<< "${EXPERIMENTS[${SLURM_ARRAY_TASK_ID}]}"

echo "===================================================" >> "${log_file}"
echo "Задача ${SLURM_ARRAY_TASK_ID}: ${RUN_NAME}" >> "${log_file}"
echo "Модель: ${MODEL}" >> "${log_file}"
echo "Старт:  $(date)" >> "${log_file}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader >> "${log_file}" 2>&1
python3 -c "import torch; print('torch', torch.__version__, '| bf16', torch.cuda.is_bf16_supported())" >> "${log_file}" 2>&1
echo "===================================================" >> "${log_file}"

if [ -f "artifacts/${RUN_NAME}/dev_metrics.json" ]; then
  echo "=== ${RUN_NAME} уже завершён, пропускаю ===" >> "${log_file}"
  exit 0
fi

python3 -c "
import sys
from transformers import AutoConfig, AutoTokenizer
name = '${MODEL}'
try:
    config = AutoConfig.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
except OSError as error:
    sys.exit(f'нет весов в {name}: {type(error).__name__}. Скачайте на ноде с сетью.')
if not tokenizer.is_fast:
    sys.exit(f'{name}: нужен fast tokenizer, иначе не будет offset_mapping')
print(f'веса на месте: словарь {tokenizer.vocab_size}, окно {config.max_position_embeddings}')
" >> "${log_file}" 2>&1 || {
  echo "=== Остановлено: проблема с весами ===" >> "${log_file}"
  exit 1
}

case "${EXTRA}" in
  @NO_VITERBI*)
    # Собираем аргументы заново, без флагов Витерби.
    python3 -m solution.train --model-name "${MODEL}" \
        --tag-scheme bilou --max-length 512 --stride 128 \
        --patience 3 --eval-batch-size 32 \
        --comet offline --comet-project uzbek-ner --comet-tag "${RUN_NAME}" \
        ${EXTRA#@NO_VITERBI} \
        --output-dir "artifacts/${RUN_NAME}" >> "${log_file}" 2>&1
    ;;
  @TWO_STAGE*)
    if [ ! -f "artifacts/${RUN_NAME}-warmup/dev_metrics.json" ]; then
      echo "=== Стадия 1: прогрев на внешнем корпусе ===" >> "${log_file}"
      python3 -m solution.train --model-name "${MODEL}" \
          --train data/external/external_train.jsonl --dev data/dev.jsonl \
          "${BASE_ARGS[@]}" --seed 42 --learning-rate 5e-6 --epochs 2 \
          --batch-size 32 --gradient-accumulation-steps 1 \
          --comet-tag external --comet-tag warmup \
          --output-dir "artifacts/${RUN_NAME}-warmup" >> "${log_file}" 2>&1
    fi
    echo "=== Стадия 2: дообучение на выборке кейса ===" >> "${log_file}"
    python3 -m solution.train --model-name "artifacts/${RUN_NAME}-warmup/model" \
        "${BASE_ARGS[@]}" ${EXTRA#@TWO_STAGE} --comet-tag external --comet-tag stage2 \
        --output-dir "artifacts/${RUN_NAME}" >> "${log_file}" 2>&1
    ;;
  *)
    python3 -m solution.train --model-name "${MODEL}" \
        "${BASE_ARGS[@]}" ${EXTRA} --comet-tag "${RUN_NAME}" \
        --output-dir "artifacts/${RUN_NAME}" >> "${log_file}" 2>&1
    ;;
esac
status=$?

echo "=== ${RUN_NAME} завершён, код ${status} ===" >> "${log_file}"
grep -a "Лучший результат" "${log_file}" | tail -1 >> "${log_file}"

# Сводка после каждого прогона: удобно смотреть, не дожидаясь всей группы.
python3 tools/summarize_runs.py artifacts >> "${log_file}" 2>&1

echo "Финиш: $(date)" >> "${log_file}"
