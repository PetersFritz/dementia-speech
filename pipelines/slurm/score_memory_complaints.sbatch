#!/bin/bash
#SBATCH --job-name=memory-complaints
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/parscratch/users/acp23fp/dementia-speech/logs/memory-complaints-%j.out

# Run from the repo root: sbatch pipelines/slurm/score_memory_complaints.sbatch
set -euo pipefail

module load Anaconda3
source activate dementia-speech-llm

export PYTHONNOUSERSITE=1
export HF_HOME=/mnt/parscratch/users/acp23fp/hf_cache

python pipelines/score_memory_complaints.py
