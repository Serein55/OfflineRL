#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export NCCL_DEBUG=WARN
mkdir -p logs artifacts
exec 9>artifacts/vanilla_delta_pipeline.lock
flock -n 9 || { echo 'Vanilla delta pipeline already running'; exit 1; }
trap 'code=$?; if [ "$code" -ne 0 ]; then echo "failed code=$code at $(date -Is)" > artifacts/vanilla_delta_pipeline_state.txt; fi' EXIT
manifest=data/processed/manifest_extra_delta.json
output=artifacts/vanilla_extra_delta_uniform_seed42
# Validate real forward/backward, checkpoint reload and closed-loop inverse transform
# before starting the independent, freshly initialized 40k-step experiment.
if [ ! -f artifacts/vanilla_delta_smoke_pass.json ]; then
  echo "smoke training at $(date -Is)" > artifacts/vanilla_delta_pipeline_state.txt
  .venv/bin/torchrun --standalone --nproc_per_node=4 scripts/train.py \
    --method vanilla --manifest "$manifest" --steps 2 --save-every 2 --smoke-inference \
    --output artifacts/smoke_extra_delta_seed42 > logs/smoke_extra_delta.log 2>&1
  echo "smoke evaluation at $(date -Is)" > artifacts/vanilla_delta_pipeline_state.txt
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 .venv/bin/python -u scripts/evaluate.py \
    --checkpoint artifacts/smoke_extra_delta_seed42/latest.pt --suites libero_goal \
    --task-ids 0 --rollouts 1 --replan-steps 5 \
    --output artifacts/smoke_extra_delta_seed42/evaluation.jsonl > logs/eval_smoke_extra_delta.log 2>&1
  .venv/bin/python - <<'PY'
import json
from pathlib import Path
rows=[json.loads(s) for s in Path('artifacts/smoke_extra_delta_seed42/evaluation.jsonl').read_text().splitlines()]
assert len(rows)==1 and rows[0]['extra_delta_transform'] and rows[0]['replan_steps']==5
Path('artifacts/vanilla_delta_smoke_pass.json').write_text(json.dumps({'training_steps':2,'closed_loop_rollouts':1,'result':rows[0]},indent=2))
PY
fi
echo "training vanilla extra delta at $(date -Is)" > artifacts/vanilla_delta_pipeline_state.txt
args=()
if [ -f "$output/latest.pt" ]; then args+=(--resume "$output/latest.pt"); fi
.venv/bin/torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --method vanilla --manifest "$manifest" --advantage-normalization none \
  --time-sampler uniform --seed 42 --steps 40000 --output "$output" "${args[@]}" \
  >> logs/train_vanilla_extra_delta.log 2>&1
echo "evaluating vanilla extra delta replan=5 at $(date -Is)" > artifacts/vanilla_delta_pipeline_state.txt
pids=()
suites=(libero_goal libero_spatial libero_object libero_10)
for gpu in 0 1 2 3; do
  suite="${suites[$gpu]}"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" .venv/bin/python -u scripts/evaluate.py \
    --checkpoint "$output/latest.pt" --suites "$suite" --seed 42 --rollouts 50 --replan-steps 5 \
    --output "$output/eval_${suite}.jsonl" >> "logs/eval_vanilla_extra_delta_${suite}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [ "$failed" -ne 0 ]; then exit 1; fi
.venv/bin/python scripts/summarize.py "$output"
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root=Path('artifacts')
old=json.loads((root/'vanilla_uniform_replan5_seed42/results.json').read_text())
new=json.loads((root/'vanilla_extra_delta_uniform_seed42/results.json').read_text())
(root/'vanilla_delta_comparison.json').write_text(json.dumps({'raw':old,'extra_delta':new,'difference_pp':100*(new['average']-old['average'])},indent=2))
PY
echo "complete at $(date -Is)" > artifacts/vanilla_delta_pipeline_state.txt
