#!/usr/bin/env bash
ARFM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ARFM_ROOT
export CUDA_VISIBLE_DEVICES=4,5,6,7
export OPENPI_DATA_HOME="$ARFM_ROOT/checkpoints/openpi_cache"
# Vendored lerobot rejects the deprecated LEROBOT_HOME outright; only HF_LEROBOT_HOME may be set.
unset LEROBOT_HOME
export HF_LEROBOT_HOME="$ARFM_ROOT/data/openpi_lerobot"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export OMP_NUM_THREADS=8
export PYTHONPATH="$ARFM_ROOT/vendor/openpi_official/src:$ARFM_ROOT/vendor/openpi_official/packages/openpi-client/src:$ARFM_ROOT/scripts"
