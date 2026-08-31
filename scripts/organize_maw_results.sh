#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-results/MAW}
mkdir -p "$ROOT/runs" "$ROOT/incomplete"

organize_run() {
  local src=$1
  local id
  id=$(basename "$src")

  mkdir -p "$src/config" "$src/adapters/final" "$src/adapters/epochs"
  mkdir -p "$src/logs/eval" "$src/logs/tensorboard" "$src/metrics"

  for file in args.json ema_state.json crash_report.json; do
    if [ -f "$src/$file" ]; then
      case "$file" in
        ema_state.json) mv "$src/$file" "$src/config/controller_state.json" ;;
        *) mv "$src/$file" "$src/config/$file" ;;
      esac
    fi
  done

  if [ -d "$src/model" ]; then
    for checkpoint in "$src"/model/checkpoint-epoch-*; do
      [ -d "$checkpoint" ] || continue
      epoch=${checkpoint##*checkpoint-epoch-}
      mv "$checkpoint" "$src/adapters/epochs/epoch-$epoch"
    done
    find "$src/model" -mindepth 1 -maxdepth 1 -exec mv -t "$src/adapters/final" -- {} + 2>/dev/null || true
    rmdir "$src/model" 2>/dev/null || true
  fi

  if [ -d "$src/tensorboard" ]; then
    find "$src/tensorboard" -mindepth 1 -maxdepth 1 -exec mv -t "$src/logs/tensorboard" -- {} + 2>/dev/null || true
    rmdir "$src/tensorboard" 2>/dev/null || true
  fi

  if [ -d "$src/vllm_eval" ]; then
    for log in "$src"/vllm_eval/epoch-*.log; do
      [ -f "$log" ] || continue
      mv "$log" "$src/logs/eval/$(basename "$log")"
    done
    for metrics in "$src"/vllm_eval/epoch-*; do
      [ -d "$metrics" ] || continue
      mv "$metrics" "$src/metrics/$(basename "$metrics")"
    done
    rmdir "$src/vllm_eval" 2>/dev/null || true
  fi

  for file in train.log eval.log pipeline.log eval_watcher.log cleanup_watcher.log; do
    [ -f "$src/$file" ] && mv "$src/$file" "$src/logs/$file"
  done
  find "$src" -mindepth 1 -maxdepth 1 -type f -name '*results.json' \
    -exec mv -t "$src/metrics" -- {} + 2>/dev/null || true

  local expected_epochs=5
  if [ -f "$src/config/args.json" ] && command -v jq >/dev/null 2>&1; then
    expected_epochs=$(jq -r '.num_epochs // 5' "$src/config/args.json")
  fi
  local destination="$ROOT/incomplete/$id"
  if [ -f "$src/adapters/epochs/epoch-${expected_epochs}/adapter_model.safetensors" ]; then
    destination="$ROOT/runs/$id"
  fi
  mv "$src" "$destination"
}

for run in "$ROOT"/20??????-??????; do
  [ -d "$run" ] || continue
  organize_run "$run"
done

# A completed run used to store the last adapter twice. Replace the duplicate
# final directory only when its weight and config are byte-identical.
for run in "$ROOT/runs"/20??????-??????; do
  [ -d "$run" ] || continue
  args="$run/config/args.json"
  expected=$(jq -r '.num_epochs // 5' "$args" 2>/dev/null || printf '5')
  final="$run/adapters/final"
  last="$run/adapters/epochs/epoch-$expected"
  if [ -d "$final" ] && [ ! -L "$final" ] && \
     cmp -s "$final/adapter_model.safetensors" "$last/adapter_model.safetensors" && \
     cmp -s <(jq -S '(.target_modules |= sort)' "$final/adapter_config.json") \
            <(jq -S '(.target_modules |= sort)' "$last/adapter_config.json"); then
    [ ! -f "$final/base_model.json" ] || mv "$final/base_model.json" "$last/base_model.json"
    rm -rf "$final"
    ln -s "epochs/epoch-$expected" "$final"
  fi
done

index="$ROOT/experiments.tsv"
printf 'run_id\tstatus\tlr\tbatch_size\tlambda\tgamma0\ttarget_gap\tgamma_gain\tgamma_min\tgamma_max\trho\tepochs\tevaluated_epochs\tsize\n' > "$index"
for bucket in runs incomplete; do
  for run in "$ROOT/$bucket"/20??????-??????; do
    [ -d "$run" ] || continue
    id=$(basename "$run")
    args="$run/config/args.json"
    if [ -f "$args" ] && command -v jq >/dev/null 2>&1; then
      fields=$(jq -r '[.lr,.batch_size,.lmbda,.gamma0,.target_gap,.gamma_gain,.gamma_min,.gamma_max,.rho,.num_epochs] | map(if . == null then "-" else tostring end) | @tsv' "$args")
    else
      fields=$'-\t-\t-\t-\t-\t-\t-\t-\t-\t-'
    fi
    evaluated=$(find "$run/metrics" -name '*final_evaluation_results.json' 2>/dev/null | wc -l)
    expected=$(jq -r '.num_epochs // 5' "$args" 2>/dev/null || printf '5')
    adapter_count=$(find "$run/adapters/epochs" -name adapter_model.safetensors 2>/dev/null | wc -l)
    if [ "$bucket" = runs ] && [ "$evaluated" -ge "$expected" ]; then
      status=evaluated
    elif [ "$bucket" = runs ]; then
      status=trained
    elif [ "$adapter_count" -gt 0 ]; then
      status=partial
    else
      status=empty
    fi
    size=$(du -sh "$run" | cut -f1)
    printf '%s\t%s\t%s\t%s\t%s\n' "$id" "$status" "$fields" "$evaluated" "$size" >> "$index"
  done
done

latest=$(find "$ROOT/runs" -mindepth 1 -maxdepth 1 -type d -name '20??????-??????' \
  -printf '%f\n' | sort | tail -n 1)
if [ -n "$latest" ]; then
  ln -sfn "runs/$latest" "$ROOT/latest"
fi
