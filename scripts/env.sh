#!/usr/bin/env bash
ARFM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ARFM_ROOT
export PYTHONPATH="$ARFM_ROOT/src:$ARFM_ROOT/vendor/lerobot/src:$ARFM_ROOT/vendor/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$ARFM_ROOT/configs/libero"
export ARFM_TOKENIZER="$ARFM_ROOT/checkpoints/tokenizer"
export MUJOCO_EGL_DEVICE_ID=0
