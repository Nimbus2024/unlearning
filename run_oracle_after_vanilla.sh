#!/usr/bin/env bash
set -uo pipefail

VANILLA_PID=2942
RESULT_FILE=/root/autodl-tmp/UMU-bench/results/vanilla_results.json

echo "[$(date +%H:%M:%S)] watcher 启动：检测 vanilla (PID=$VANILLA_PID) 是否结束"

# 双条件：PID 不存在 或 结果文件已生成，任一满足即认为 vanilla 完成
while true; do
  if ! kill -0 "$VANILLA_PID" 2>/dev/null || [ -f "$RESULT_FILE" ]; then
    break
  fi
  sleep 20
done

# 等结果文件落盘完整（防止文件刚创建还没写全）
for i in 1 2 3 4 5 6; do
  [ -f "$RESULT_FILE" ] && break
  sleep 5
done

echo "[$(date +%H:%M:%S)] vanilla 已结束，启动 oracle 评估"
cd /root/autodl-tmp/UMU-bench
export PATH=/root/miniconda3/bin:$PATH
source /root/autodl-tmp/hf.env

# oracle 评估：失败也继续（用 ; 分隔保证走到关机），日志始终 tee 落盘
python eval.py \
  --model_id llava-hf/llava-1.5-7b-hf \
  --cache_path /root/autodl-tmp/models/llava_smu_ft \
  --forget_ratio 5 \
  --data_split_dir /root/autodl-tmp/data/UMU-bench \
  --output_path /root/autodl-tmp/UMU-bench/results \
  --output_file oracle_results.json \
  2>&1 | tee /root/autodl-tmp/UMU-bench/results/oracle_eval.log
PY_EXIT=${PIPESTATUS[0]}
echo "[$(date +%H:%M:%S)] oracle 评估结束，exit_code=$PY_EXIT，即将关机"
sleep 3
/usr/bin/shutdown
