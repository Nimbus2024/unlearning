#!/usr/bin/env bash
# 训练+eval 进程结束即关机。用 ps|grep -v grep 避免 pgrep -f 自匹配。
set -uo pipefail
echo "[$(date +%H:%M:%S)] 监控 NPO 训练/eval 进程... (不依赖本地)"

for i in $(seq 1 240); do
  # 检测 eval 和 train 进程; grep -v grep 排除命令自身
  if ! ps aux | grep "python eval.py" | grep -v grep >/dev/null 2>&1 && \
     ! ps aux | grep "python unlearn/NPO.py" | grep -v grep >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] 训练/eval 进程均已结束"
    RESULTS=$(ls /root/autodl-tmp/UMU-bench/results/NPO/*/NPO_results.json 2>/dev/null | head -1)
    if [ -n "${RESULTS}" ]; then
      echo "[$(date +%H:%M:%S)] 结果文件: ${RESULTS}"
    else
      echo "[$(date +%H:%M:%S)] 警告: 无结果文件, 但流程已结束"
    fi
    echo "[$(date +%H:%M:%S)] 即将关机"
    sleep 3
    /usr/bin/shutdown
    exit 0
  fi
  sleep 30
done
echo "[$(date +%H:%M:%S)] 警告: 2小时超时, 进程仍在, 不关机"
