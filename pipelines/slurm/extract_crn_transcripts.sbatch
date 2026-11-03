#!/bin/bash
# Copies the "who-concerned" transcript for each PROCESS participant from
# /mnt/parscratch/users/acp23fp/DementiaSpeech_HPC/PROCESS_Master/files into
# data/raw/text/CRN (pipelines/extract_crn_transcripts.py). Pure file I/O, no
# GPU/model needed -- unlike score_who_concerned.sbatch, this can run on a CPU partition.
#
# Run from the repo root: sbatch pipelines/slurm/extract_crn_transcripts.sbatch

#SBATCH --job-name=extract-crn-transcripts
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/parscratch/users/acp23fp/dementia-speech/logs/extract-crn-transcripts-%j.out

module load Anaconda3
source activate dementia-speech-llm

# Strict mode only after conda activation (its MKL scripts break under `set -u`).
set -euo pipefail

export PYTHONNOUSERSITE=1
export DEMENTIA_SPEECH_DATA_DIR=/mnt/parscratch/users/acp23fp/dementia-speech/data

python pipelines/extract_crn_transcripts.py
