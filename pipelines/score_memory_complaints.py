"""Batch-extract memory complaints from Q2 ('memory-problems') transcripts via a local LLM.

Needs a GPU (loads Qwen2.5-14B-Instruct) -- run through pipelines/slurm/score_memory_complaints.sbatch,
not directly on a login node. The dashboard never calls the model itself: like score_pft_new.py and
score_cookie_theft.py, this writes a CSV once and Shiny just reads it at runtime.
"""
import os
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
os.chdir(project_root)

from dementia_speech.tasks.memory_complaints.extract import extract
from dementia_speech.tasks.memory_complaints.verify import drop_ungrounded

TRANSCRIPT_DIR = Path("data/raw/text/MEM")  # adjust to your path -- Q2 "memory-problems" transcripts
METADATA_PATH = Path("data/raw/meta-info.csv")
OUTPUT_PATH = Path("data/processed/memory_complaints.csv")


def extract_participant_id(filename) -> str:
    parts = filename.stem.split("__")
    return parts[1] if len(parts) >= 2 else filename.stem


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(METADATA_PATH)

    transcript_files = sorted(TRANSCRIPT_DIR.glob("*.txt"))
    print(f"Found {len(transcript_files)} transcripts")

    rows = []
    for i, txt_path in enumerate(transcript_files, 1):
        participant_id = extract_participant_id(txt_path)
        transcript = txt_path.read_text().strip()

        match = metadata[metadata["IDs"].astype(str).str.contains(participant_id)]
        diagnosis = match.iloc[0]["diagnosis"] if len(match) > 0 else "Unknown"

        try:
            result = extract(transcript)
            result = drop_ungrounded(result, transcript)
        except Exception as e:
            print(f"  ERROR on {txt_path.name}: {e}")
            continue

        base_row = {
            "participant_id": participant_id,
            "diagnosis": diagnosis,
            "source_file": txt_path.name,
            "status": result.status,
        }
        if result.findings:
            for finding in result.findings:
                rows.append({
                    **base_row,
                    "domain": finding.domain,
                    "summary": finding.summary,
                    "evidence_quote": finding.evidence_quote,
                })
        else:
            rows.append({**base_row, "domain": None, "summary": None, "evidence_quote": None})

        if i % 20 == 0:
            print(f"  {i}/{len(transcript_files)} done")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(results_df.groupby("participant_id")["status"].first().value_counts())


if __name__ == "__main__":
    main()
