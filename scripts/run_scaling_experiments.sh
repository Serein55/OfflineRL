#!/usr/bin/env bash
# Matched 40k-step experiments. Run in tmux; stop on any failed train/eval.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export NCCL_DEBUG=WARN
mkdir -p logs artifacts
exec 9>artifacts/scaling_pipeline.lock
flock -n 9 || { echo 'Scaling pipeline already running'; exit 1; }
trap 'code=$?; if [ "$code" -ne 0 ]; then echo "failed code=$code at $(date -Is)" > artifacts/scaling_pipeline_state.txt; fi' EXIT
# Require the completed, zero-update diagnostic before launching any training.
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('artifacts/scaling_diagnostic')
assert json.loads((p/'protocol.json').read_text())['updates']==0
s=json.loads((p/'summary.json').read_text())['vanilla_40k']['zscore_adaptive']
assert s['ess']['mean']<15 and s['fraction_ess_below_4']<.05, s
PY
for experiment in arfm_taskz rwr_taskz_a01 rwr_taskz_a05; do
  output="artifacts/${experiment}_uniform_seed42"
  method=arfm; alpha=.1
  if [[ "$experiment" == rwr* ]]; then method=rwr; fi
  if [[ "$experiment" == *a05 ]]; then alpha=.5; fi
  echo "training $experiment at $(date -Is)" > artifacts/scaling_pipeline_state.txt
  args=()
  if [ -f "$output/latest.pt" ]; then args+=(--resume "$output/latest.pt"); fi
  .venv/bin/torchrun --standalone --nproc_per_node=4 scripts/train.py \
    --method "$method" --fixed-alpha "$alpha" --advantage-normalization task_zscore \
    --arfm-lambda 0.0005 --time-sampler uniform --seed 42 --steps 40000 \
    --output "$output" "${args[@]}" >> "logs/train_${experiment}.log" 2>&1
  echo "evaluating $experiment replan=5 at $(date -Is)" > artifacts/scaling_pipeline_state.txt
  pids=()
  suites=(libero_goal libero_spatial libero_object libero_10)
  for gpu in 0 1 2 3; do
    suite="${suites[$gpu]}"
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" .venv/bin/python -u scripts/evaluate.py \
      --checkpoint "$output/latest.pt" --suites "$suite" --seed 42 --rollouts 50 --replan-steps 5 \
      --output "$output/eval_${suite}.jsonl" >> "logs/eval_${experiment}_${suite}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if [ "$failed" -ne 0 ]; then exit 1; fi
  .venv/bin/python scripts/summarize.py "$output"
  .venv/bin/python scripts/report_scaling.py
done
echo "complete at $(date -Is)" > artifacts/scaling_pipeline_state.txt
