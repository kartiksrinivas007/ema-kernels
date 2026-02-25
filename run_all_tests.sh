#!/usr/bin/env bash
set -euo pipefail

python -m pytest -v -s \
  kernels/tests/ema_ssd_bwd/test_ema_dpx.py \
  kernels/tests/ema_ssd_bwd/test_ema_combined_autograd.py
