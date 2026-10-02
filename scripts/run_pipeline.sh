#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export NCCL_DEBUG=WARN
mkdir -p artifacts logs
trap 'code=$?; if [ "$code" -ne 0 ]; then echo "pipeline failed code=$code at $(date -Is)" > artifacts/pipeline_state.txt; fi' EXIT
if [ ! -f data/processed/manifest.json ]; then
  echo "preprocessing" > artifacts/pipeline_state.txt
  .venv/bin/python -u scripts/preprocess.py --workers 24 > logs/preprocess.log 2>&1
fi
# Matched Uniform baseline and ARFM; each starts from identical base pi0.
for method in vanilla arfm rwr; do
  output="artifacts/${method}_uniform_seed42"
  echo "training $method" > artifacts/pipeline_state.txt
  args=()
  if [ -f "$output/latest.pt" ]; then args+=(--resume "$output/latest.pt"); fi
  .venv/bin/torchrun --standalone --nproc_per_node=4 scripts/train.py \
    --method "$method" --time-sampler uniform --steps 40000 --output "$output" "${args[@]}" > "logs/train_${method}.log" 2>&1
  echo "evaluating $method" > artifacts/pipeline_state.txt
  # One suite per GPU; independent fixed-init-state evaluations.
  pids=()
  suites=(libero_goal libero_spatial libero_object libero_10)
  for gpu in 0 1 2 3; do
    suite="${suites[$gpu]}"
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" .venv/bin/python -u scripts/evaluate.py \
      --checkpoint "$output/latest.pt" --suites "$suite" --output "$output/eval_${suite}.jsonl" > "logs/eval_${method}_${suite}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  .venv/bin/python scripts/summarize.py "$output"
done
echo "complete" > artifacts/pipeline_state.txt
