#!/bin/bash
# PO 冻结版重训（非冻结历史 run 替换；满足有效遗忘条件①）
# 用法: bash run_po_frozen.sh [lr] [epochs]
export PATH=/root/miniconda3/bin:$PATH
export TERM=xterm
export HF_HOME=/root/autodl-tmp/hf
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

LR="${1:-1e-5}"
EPOCHS="${2:-3}"
TS=$(date +%Y%m%d-%H%M%S)
RUN_DIR="/root/autodl-tmp/UMU-bench/cross-modal-diagnosis/results/PO_frozen/${TS}"
mkdir -p "${RUN_DIR}"
echo "RUN_DIR=${RUN_DIR} LR=${LR} EPOCHS=${EPOCHS}" | tee "${RUN_DIR}/run_meta.txt"

cd /root/autodl-tmp/UMU-bench
tmux kill-session -t tune 2>/dev/null
tmux new-session -d -s tune "cd /root/autodl-tmp/UMU-bench && export PATH=/root/miniconda3/bin:\$PATH && export TERM=xterm && export HF_HOME=/root/autodl-tmp/hf && export HF_ENDPOINT=https://hf-mirror.com && export HF_HUB_DISABLE_XET=1 && \
echo ===TRAIN=== && /root/miniconda3/envs/unlearn/bin/python unlearn/PO.py --model_id llava-hf/llava-1.5-7b-hf --vanilla_dir chengyewang/llava_smu_ft --run_dir ${RUN_DIR} --data_split_dir /root/autodl-tmp/data/UMU-bench --forget_split_ratio 5 --batch_size 1 --alpha 1.0 --gamma 1.0 --lr ${LR} --num_epochs ${EPOCHS} --max_length 384 2>&1 | tee ${RUN_DIR}/train.log ; \
echo ===EVAL=== && /root/miniconda3/envs/unlearn/bin/python eval.py --model_id llava-hf/llava-1.5-7b-hf --cache_path ${RUN_DIR}/model --forget_ratio 5 --data_split_dir /root/autodl-tmp/data/UMU-bench --output_path ${RUN_DIR} --output_file PO_frozen_results.json 2>&1 | tee ${RUN_DIR}/eval.log ; \
echo ===DONE_SHUTDOWN=== ; /usr/bin/shutdown"
echo "tmux tune 启动 (RUN_DIR=${RUN_DIR}): $?"
