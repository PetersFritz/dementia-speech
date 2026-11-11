#!/bin/bash
#SBATCH --job-name=score-news
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/parscratch/users/acp23fp/dementia-speech/logs/score-news-%j.out

# Run from the repo root: sbatch pipelines/slurm/score_news.sbatch
#
# Needs data/processed/news_events_cache.csv already built
# (pipelines/build_news_event_cache.py, run separately from a login node --
# it needs outbound internet, which this GPU node won't have) and
# data/raw/Process_demographics.csv's assessment_date column already added
# (pipelines/scratch/add_assessment_dates.py).

module load Anaconda3
source activate dementia-speech-llm

# Strict mode only after conda activation (its MKL scripts break under `set -u`).
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HOME=/mnt/parscratch/users/acp23fp/hf_cache
export DEMENTIA_SPEECH_DATA_DIR=/mnt/parscratch/users/acp23fp/dementia-speech/data

python pipelines/score_news.py
