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
if [ "${METHOD}" = "MAW" ]; then
  RUN_DIR="results/MAW/runs/${TS}"
  TRAIN_LOG="${RUN_DIR}/logs/train.log"
else
  RUN_DIR="results/${METHOD}/${TS}"
  TRAIN_LOG="${RUN_DIR}/train.log"
fi
mkdir -p "${RUN_DIR}" "$(dirname "${TRAIN_LOG}")"
if [ "${METHOD}" = "MAW" ]; then
  ln -sfn "runs/${TS}" results/MAW/latest
fi

MODEL_ID=${MODEL_ID:-llava-hf/llava-1.5-7b-hf}
# 遗忘基座和数据默认使用项目内指向 UMU-bench 的软链接；可通过环境变量覆盖。
VANILLA_DIR=${VANILLA_DIR:-./llava_smu_ft}
DATA_SPLIT_DIR=${DATA_SPLIT_DIR:-./data}
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
  # DDP 多卡训练: NPO_NPROC 默认 4 (4 卡), per-GPU batch = --batch_size, global batch = x NPROC
  accelerate launch --num_processes "${NPO_NPROC:-4}" unlearn/NPO.py --model_id "${MODEL_ID}" --vanilla_dir "${VANILLA_DIR}" \
    --oracle_model_id "${ORACLE_DIR:-chengyewang/llava_smu_ft}" \
    --run_dir "${RUN_DIR}" --data_split_dir "${DATA_SPLIT_DIR}" "$@" \
    2>&1 | tee "${RUN_DIR}/train.log"
elif [ "${METHOD}" = "MAW" ]; then
  # MAW: four ranks, batch size 2/GPU (global batch size 8).
  accelerate launch --num_processes 4 unlearn/MAW.py --model_id "${MODEL_ID}" \
    --vanilla_dir "${VANILLA_DIR}" \
    --processor_dir "${PROCESSOR_DIR:-./llava_smu_ft}" \
    --batch_size 2 --lr 5e-5 --lmbda 1 --num_epochs 5 \
    --run_dir "${RUN_DIR}" --data_split_dir "${DATA_SPLIT_DIR}" "$@" \
    2>&1 | tee "${TRAIN_LOG}"
else
  echo "Unknown method: ${METHOD}" >&2
  exit 1
fi

# --- Eval (vLLM backend, UMU parquet schema) ---
if [ "${METHOD}" = "MAW" ]; then
  mkdir -p "${RUN_DIR}/logs/eval" "${RUN_DIR}/metrics"
  for checkpoint in "${RUN_DIR}"/adapters/epochs/epoch-*; do
    [ -d "${checkpoint}" ] || continue
    epoch=$(basename "${checkpoint}" | sed 's/epoch-//')
    epoch_dir="${RUN_DIR}/metrics/epoch-${epoch}"
    eval_log="${RUN_DIR}/logs/eval/epoch-${epoch}.log"
    echo "=== [$(date +%H:%M:%S)] Eval epoch ${epoch} on ${checkpoint} ===" | tee "${eval_log}"
    set +e
    python eval_vllm.py --model_id "${MODEL_ID}" \
      --cache_path "${checkpoint}" \
      --processor_path "${PROCESSOR_DIR:-./llava_smu_ft}" \
      --data_split_folder "${DATA_SPLIT_DIR}" \
      --task_data "${DATA_SPLIT_DIR}/full_data/train-00000-of-00001.parquet" \
      --test_data "${DATA_SPLIT_DIR}/full_data/train-00000-of-00001.parquet" \
      --celebrity_data "${DATA_SPLIT_DIR}/real_person/train-00000-of-00001.parquet" \
      --output_folder "${epoch_dir}" \
      --output_file "${METHOD}_epoch-${epoch}" \
      --forget_ratio "${FORGET_RATIO}" \
      --batch_size 32 --tensor_parallel_size 1 --max_model_len 4096 \
      2>&1 | tee -a "${eval_log}"
    eval_status=${PIPESTATUS[0]}
    set -e
    # Python finally handles normal exits; this also covers native vLLM aborts.
    rm -rf -- "$(dirname "${checkpoint}")/.$(basename "${checkpoint}")_vllm_merged"
    if [ "${eval_status}" -ne 0 ]; then
      echo "vLLM evaluation failed for epoch ${epoch} (status ${eval_status})" >&2
      exit "${eval_status}"
    fi
  done
else
  echo "=== [$(date +%H:%M:%S)] Eval on ${RUN_DIR}/model ==="
  set +e
  python eval_vllm.py --model_id "${MODEL_ID}" \
    --cache_path "${RUN_DIR}/model" \
    --processor_path "${PROCESSOR_DIR:-./llava_smu_ft}" \
    --data_split_folder "${DATA_SPLIT_DIR}" \
    --task_data "${DATA_SPLIT_DIR}/full_data/train-00000-of-00001.parquet" \
    --test_data "${DATA_SPLIT_DIR}/full_data/train-00000-of-00001.parquet" \
    --celebrity_data "${DATA_SPLIT_DIR}/real_person/train-00000-of-00001.parquet" \
    --output_folder "${RUN_DIR}" \
    --output_file "${METHOD}_results" \
    --forget_ratio "${FORGET_RATIO}" \
    --batch_size 32 --tensor_parallel_size 4 --max_model_len 4096 \
    2>&1 | tee "${RUN_DIR}/eval.log"
  eval_status=${PIPESTATUS[0]}
  set -e
  rm -rf -- "${RUN_DIR}/.model_vllm_merged"
  if [ "${eval_status}" -ne 0 ]; then
    exit "${eval_status}"
  fi
fi

echo "=== [$(date +%H:%M:%S)] Done. Results in ${RUN_DIR} ==="

# 可选: 完成后自动关机 (SHUTDOWN_AFTER=1 时)。日志已落盘, 关机后仍可读。
if [ "${SHUTDOWN_AFTER:-0}" = "1" ]; then
  echo "=== [$(date +%H:%M:%S)] SHUTDOWN_AFTER=1, 3秒后关机 ==="
  sleep 3
  /usr/bin/shutdown
fi
