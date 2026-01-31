#!/bin/bash
#SBATCH --job-name=ema_ssd_fwd
#SBATCH --partition=flame-earlybirds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32G
#SBATCH --qos=earlybird_qos
#SBATCH --time=00:20:00
#SBATCH --output=sbatch_scripts/logs/ema_ssd_fwd_%j.out
#SBATCH --error=sbatch_scripts/logs/ema_ssd_fwd_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

mkdir -p sbatch_scripts/logs

OUT_FILE="scratch.txt"
{
  echo "Running EMA SSD forward on $(hostname)"
  date
  python kernels/new_ema_kernels/ema_ssd_fwd.py
  date
} | tee -a "${OUT_FILE}"
