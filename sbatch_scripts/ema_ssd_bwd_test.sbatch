#!/bin/bash
#SBATCH --job-name=ema_ssd_bwd_test
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=sbatch_scripts/logs/ema_ssd_bwd_test_%j.out
#SBATCH --error=sbatch_scripts/logs/ema_ssd_bwd_test_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

mkdir -p sbatch_scripts/logs

OUT_FILE="scratch.txt"
{
  echo "Running EMA SSD backward test on $(hostname)"
  date
  python -m pytest -s kernels/tests/ema_ssd_bwd/test_ema_dpx.py
  date
} | tee -a "${OUT_FILE}"
