#!/bin/bash
# Copies the "recent news" transcript for each PROCESS participant from
# /mnt/parscratch/users/acp23fp/DementiaSpeech_HPC/PROCESS_Master/files into
# data/raw/text/STN (pipelines/extract_news_transcripts.py). Pure file I/O, no
# GPU/model needed -- unlike a scoring job, this can run on a CPU partition.
#
# Run from the repo root: sbatch pipelines/slurm/extract_news_transcripts.sbatch

#SBATCH --job-name=extract-news-transcripts
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/parscratch/users/acp23fp/dementia-speech/logs/extract-news-transcripts-%j.out

module load Anaconda3
source activate dementia-speech-llm

# Strict mode only after conda activation (its MKL scripts break under `set -u`).
set -euo pipefail

export PYTHONNOUSERSITE=1
export DEMENTIA_SPEECH_DATA_DIR=/mnt/parscratch/users/acp23fp/dementia-speech/data

python pipelines/extract_news_transcripts.py
