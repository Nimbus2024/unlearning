#!/usr/bin/env bash
# Run one unlearn method: train + eval, all outputs under a single run dir.
#
# Usage:
#   ./run_unlearn.sh <method> [train args...]
#
# Example:
#   ./run_unlearn.sh GA --forget_split_ratio 5 --num_epochs 3 --batch_size 1
#   ./run_unlearn.sh PO --forget_split_ratio 5 --num_epochs 3 --batch_size 1
#
# The run dir defaults to results/<METHOD>/<timestamp>/. train.log and eval.log
# are written there; the model goes to <run_dir>/model/ and eval results to
# <run_dir>/<METHOD>_results.json.
set -euo pipefail

# 加载 HF 环境配置(缓存/镜像/xet)
if [ -f "$(dirname "$0")/.env" ]; then
  source "$(dirname "$0")/.env"
fi

METHOD="$1"
shift

TS=$(date +%Y%m%d-%H%M%S)
RUN_DIR="results/${METHOD}/${TS}"
mkdir -p "${RUN_DIR}"

MODEL_ID=${MODEL_ID:-llava-hf/llava-1.5-7b-hf}
# 遗忘基座默认是 SFT 模型 (llava_smu_ft)；all unlearn 方法遗忘的是 SFT 后的知识
VANILLA_DIR=${VANILLA_DIR:-chengyewang/llava_smu_ft}
DATA_SPLIT_DIR=${DATA_SPLIT_DIR:-/root/autodl-tmp/data/UMU-bench}
FORGET_RATIO=${FORGET_RATIO:-5}

echo "=== [$(date +%H:%M:%S)] Run dir: ${RUN_DIR} ==="

# --- Train ---
if [ "${METHOD}" = "GA" ]; then
  python unlearn/GA.py --model_id "${MODEL_ID}" --vanilla_dir "${VANILLA_DIR}" \
    --run_dir "${RUN_DIR}" --data_split_dir "${DATA_SPLIT_DIR}" "$@" \
    2>&1 | tee "${RUN_DIR}/train.log"
elif [ "${METHOD}" = "PO" ]; then
  python unlearn/PO.py --model_id "${MODEL_ID}" --vanilla_dir "${VANILLA_DIR}" \
    --run_dir "${RUN_DIR}" --data_split_dir "${DATA_SPLIT_DIR}" "$@" \
    2>&1 | tee "${RUN_DIR}/train.log"
elif [ "${METHOD}" = "GD" ]; then
  python unlearn/Graddiff.py --model_id "${MODEL_ID}" --vanilla_dir "${VANILLA_DIR}" \
    --run_dir "${RUN_DIR}" --data_split_dir "${DATA_SPLIT_DIR}" "$@" \
    2>&1 | tee "${RUN_DIR}/train.log"
elif [ "${METHOD}" = "KL" ]; then
  python unlearn/KL.py --model_id "${MODEL_ID}" --vanilla_dir "${VANILLA_DIR}" \
    --oracle_model_id "${ORACLE_DIR:-chengyewang/llava_smu_ft}" \
    --run_dir "${RUN_DIR}" --data_split_dir "${DATA_SPLIT_DIR}" "$@" \
    2>&1 | tee "${RUN_DIR}/train.log"
elif [ "${METHOD}" = "NPO" ]; then
  python unlearn/NPO.py --model_id "${MODEL_ID}" --vanilla_dir "${VANILLA_DIR}" \
    --oracle_model_id "${ORACLE_DIR:-chengyewang/llava_smu_ft}" \
    --run_dir "${RUN_DIR}" --data_split_dir "${DATA_SPLIT_DIR}" "$@" \
    2>&1 | tee "${RUN_DIR}/train.log"
elif [ "${METHOD}" = "OURS" ]; then
  # Ours_v2: dynamic-gamma DPO unlearn; base/ref = llava_smu_ft (SFT), processor = llava-1.5-7b-hf
  python unlearn/Ours_v2.py --model_id "${MODEL_ID}" --vanilla_dir "${VANILLA_DIR}" \
    --processor_dir "${PROCESSOR_DIR:-llava-hf/llava-1.5-7b-hf}" \
    --run_dir "${RUN_DIR}" --data_split_dir "${DATA_SPLIT_DIR}" "$@" \
    2>&1 | tee "${RUN_DIR}/train.log"
else
  echo "Unknown method: ${METHOD}" >&2
  exit 1
fi

# --- Eval ---
echo "=== [$(date +%H:%M:%S)] Eval on ${RUN_DIR}/model ==="
python eval.py --model_id "${MODEL_ID}" \
  --cache_path "${RUN_DIR}/model" \
  --forget_ratio "${FORGET_RATIO}" \
  --data_split_dir "${DATA_SPLIT_DIR}" \
  --output_path "${RUN_DIR}" \
  --output_file "${METHOD}_results.json" \
  2>&1 | tee "${RUN_DIR}/eval.log"

echo "=== [$(date +%H:%M:%S)] Done. Results in ${RUN_DIR} ==="
