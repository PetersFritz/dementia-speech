#!/bin/bash
# Copies the "memory-problems" transcript for each PROCESS participant from
# /mnt/parscratch/users/acp23fp/PROCESS_Master/files into data/raw/text/MEM
# (pipelines/extract_memory_transcripts.py). Pure file I/O, no GPU/model needed --
# unlike score_memory_complaints.sbatch, this can run on a CPU partition.
#
# Run from the repo root: sbatch pipelines/slurm/extract_memory_transcripts.sbatch

#SBATCH --job-name=extract-mem-transcripts
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/parscratch/users/acp23fp/dementia-speech/logs/extract-mem-transcripts-%j.out

module load Anaconda3
source activate dementia-speech-llm

# Strict mode only after conda activation (its MKL scripts break under `set -u`).
set -euo pipefail

export PYTHONNOUSERSITE=1

python pipelines/extract_memory_transcripts.py
