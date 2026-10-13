#!/usr/bin/env bash
# Source this file, then use vendor/openpi_official/.venv/bin/python.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/openpi_env.sh"
export PYTHONPATH="$ARFM_ROOT/src:$PYTHONPATH"
