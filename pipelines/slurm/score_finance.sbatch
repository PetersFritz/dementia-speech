#!/bin/bash
#SBATCH --job-name=finance-status
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/parscratch/users/acp23fp/dementia-speech/logs/finance-status-%j.out

# Run from the repo root: sbatch pipelines/slurm/score_finance.sbatch

module load Anaconda3
source activate dementia-speech-llm

# Strict mode only after conda activation (its MKL scripts break under `set -u`).
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HOME=/mnt/parscratch/users/acp23fp/hf_cache

python pipelines/score_finance.py
