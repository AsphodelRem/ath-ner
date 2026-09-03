#!/bin/bash
#SBATCH --job-name=ner-mlm
#SBATCH --partition=a100
#SBATCH --error="/userspace/kms/Code/ner-uz/logs/mlm_%A_%a.err"
#SBATCH --output="/userspace/kms/Code/ner-uz/logs/mlm_%A_%a.log"
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --array=0-2%1
#
# ==========================================================================
# Доменная адаптация энкодера и дообучение под NER.
#
#   0: MLM на узбекском корпусе
#   1: NER на адаптированном энкодере, сид 42
#   2: то же, сид 43
#
#   sbatch --nodelist=ngpu06 tools/slurm/pretrain.sbatch
#   sbatch --nodelist=ngpu07 --export=ALL,BASE=rembert tools/slurm/pretrain.sbatch
#   sbatch --array=1-2%1 tools/slurm/pretrain.sbatch      # MLM уже готов
#
# %1 обязателен: задачи 1 и 2 используют результат задачи 0.
# ==========================================================================

# ---------------------------------------------------------------- переменные
project_root="/userspace/kms/Code/ner-uz"
miniconda_dir="/userspace/kms/miniconda3"
conda_env="ner"
cache_dir="/userspace/kms/.cache"
models_dir="/userspace/kms/models"
logs_dir="${project_root}/logs"
log_file="${logs_dir}/mlm_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

# База выбирается при постановке: --export=ALL,BASE=rembert
# RemBERT для MLM почти вдвое дороже: 32 слоя против 24 и 990M параметров
# против 814M, потому что выходные эмбеддинги у него 1664-мерные.
BASE="${BASE:-large}"
case "${BASE}" in
  large)   base_model="${models_dir}/xlm-roberta-large"; mlm_batch=16; mlm_accum=2; ner_lr=1e-5 ;;
  rembert) base_model="${models_dir}/rembert";           mlm_batch=8;  mlm_accum=4; ner_lr=1e-5 ;;
  glot500) base_model="${models_dir}/glot500-base";      mlm_batch=32; mlm_accum=1; ner_lr=3e-5 ;;
  *) echo "неизвестная база ${BASE}"; exit 1 ;;
esac
corpus="${project_root}/data/pretrain/corpus.jsonl"
holdout="${project_root}/data/pretrain/corpus.holdout.jsonl"
manifest="${project_root}/data/pretrain/corpus.manifest.json"
mlm_dir="artifacts/mlm-${BASE}"

mkdir -p "${logs_dir}"

# ------------------------------------------------------------------ окружение
source "${miniconda_dir}/etc/profile.d/conda.sh"
conda activate "${conda_env}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Обе переменные явно: HF_HUB_CACHE имеет приоритет над путём из HF_HOME,
# и унаследованное из профиля значение молча перебьёт настройку.
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

echo "===================================================" >> "${log_file}"
echo "Задача ${SLURM_ARRAY_TASK_ID} | база ${BASE} (${base_model})" >> "${log_file}"
echo "Старт: $(date)" >> "${log_file}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader >> "${log_file}" 2>&1
python3 -c "import torch, transformers, accelerate; print('torch', torch.__version__, '| transformers', transformers.__version__, '| accelerate', accelerate.__version__, '| bf16', torch.cuda.is_bf16_supported())" >> "${log_file}" 2>&1 \
  || { echo "=== Остановлено: нет accelerate, Trainer не запустится ===" >> "${log_file}"; exit 1; }
echo "===================================================" >> "${log_file}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    # ------------------------------------------------- стадия 1: адаптация
    if [ -f "${mlm_dir}/model/pretrain_info.json" ]; then
      echo "=== MLM уже готов, пропускаю ===" >> "${log_file}"
      exit 0
    fi
    if [ ! -f "${corpus}" ]; then
      echo "=== Нет корпуса ${corpus} ===" >> "${log_file}"
      echo "Соберите на логин-ноде: python3 tools/prepare_pretrain.py ..." >> "${log_file}"
      exit 1
    fi

    MAX_STEPS="${MAX_STEPS:-$(python3 -c "
import json, sys
try:
    tokens = json.load(open('${manifest}'))['estimated_tokens']
except Exception:
    sys.exit('')
print(max(1000, tokens // (${mlm_batch} * ${mlm_accum} * 512)))
" 2>/dev/null)}"
    if [ -z "${MAX_STEPS}" ]; then
      echo "=== Не удалось прочитать ${manifest}, задайте MAX_STEPS вручную ===" >> "${log_file}"
      exit 1
    fi
    echo "Корпус: $(wc -l < "${corpus}") пассажей | шагов: ${MAX_STEPS}" >> "${log_file}"
    python3 -c "
import json
m = json.load(open('${manifest}'))
print('  токенов:', m['estimated_tokens'], '| симв./токен:', m['chars_per_token_used'],
      '| калибровано:', m.get('chars_per_token_calibrated'))
print('  по графикам:', m['by_script'])
print('  по источникам:', m['by_source'])
" >> "${log_file}" 2>&1

    echo "=== Стадия 1: MLM ===" >> "${log_file}"
    python3 -m solution.pretrain \
        --corpus "${corpus}" \
        --eval-corpus "${holdout}" \
        --model-name "${base_model}" \
        --output-dir "${mlm_dir}" \
        --max-length 512 \
        --batch-size "${mlm_batch}" --gradient-accumulation-steps "${mlm_accum}" \
        --learning-rate 5e-5 --warmup-ratio 0.06 \
        --max-steps "${MAX_STEPS}" \
        --eval-every 1000 --save-every 2000 --log-every 100 \
        --num-workers 8 --seed 42 --resume \
        --comet offline --comet-project uzbek-ner-mlm >> "${log_file}" 2>&1
    status=$?
    echo "=== MLM завершён, код ${status} ===" >> "${log_file}"
    grep -a "held-out" "${log_file}" | tail -3 >> "${log_file}"
    ;;

  1|2)
    # -------------------------------------------- стадия 2: NER поверх MLM
    SEED=$(( 41 + SLURM_ARRAY_TASK_ID ))
    RUN_NAME="mlm-${BASE}-s${SEED}"
    if [ ! -f "${mlm_dir}/model/pretrain_info.json" ]; then
      echo "=== Нет адаптированного энкодера в ${mlm_dir}/model ===" >> "${log_file}"
      echo "Сначала должна отработать задача 0." >> "${log_file}"
      exit 1
    fi
    if [ -f "artifacts/${RUN_NAME}/dev_metrics.json" ]; then
      echo "=== ${RUN_NAME} уже завершён, пропускаю ===" >> "${log_file}"
      exit 0
    fi
    echo "=== Стадия 2: NER, сид ${SEED} ===" >> "${log_file}"
    cat "${mlm_dir}/model/pretrain_info.json" >> "${log_file}"

    python3 -m solution.train \
        --model-name "${mlm_dir}/model" \
        --tag-scheme bilou --max-length 512 --stride 128 \
        --viterbi --transition-weight 0.25 --transition-mode conditional \
        --batch-size 32 --gradient-accumulation-steps 1 \
        --learning-rate "${ner_lr}" --epochs 12 --patience 3 --eval-batch-size 32 \
        --seed "${SEED}" \
        --comet offline --comet-project uzbek-ner \
        --comet-tag mlm --comet-tag "${BASE}" --comet-tag "seed${SEED}" \
        --output-dir "artifacts/${RUN_NAME}" >> "${log_file}" 2>&1
    echo "=== ${RUN_NAME} завершён, код $? ===" >> "${log_file}"
    grep -a "Лучший результат" "${log_file}" | tail -1 >> "${log_file}"

    # Итог подводит последняя задача.
    if [ "${SLURM_ARRAY_TASK_ID}" -eq 2 ]; then
      echo "=== Сводка ===" >> "${log_file}"
      python3 tools/summarize_runs.py artifacts >> "${log_file}" 2>&1
      control="artifacts/large-s42/dev_predictions.jsonl"
      if [ -f "${control}" ] && [ -f "artifacts/${RUN_NAME}/dev_predictions.jsonl" ]; then
        echo "=== Срезы против контроля. Главный вопрос: выросла ли КИРИЛЛИЦА ===" >> "${log_file}"
        python3 tools/evaluate_slices.py \
            --predictions "${control}" \
                          "artifacts/${RUN_NAME}/dev_predictions.jsonl" >> "${log_file}" 2>&1
      fi
    fi
    ;;
esac

echo "Финиш: $(date)" >> "${log_file}"
