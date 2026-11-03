"""Batch-extract who-is-concerned-about-memory status from Q3 ('who-concerned')
transcripts via a local LLM.

Needs a GPU (loads Qwen2.5-14B-Instruct) -- run through
pipelines/slurm/score_who_concerned.sbatch, not directly on a login node. The dashboard
never calls the model itself: like score_finance.py and score_memory_complaints.py,
this writes a CSV once and Shiny just reads it at runtime.
"""
import os
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
os.chdir(project_root)

from dementia_speech.paths import DATA_DIR
from dementia_speech.tasks.who_concerned.extract import extract
from dementia_speech.tasks.who_concerned.verify import drop_if_ungrounded

TRANSCRIPT_DIR = DATA_DIR / "raw/text/CRN"  # Q3 "who-concerned" transcripts
METADATA_PATH = DATA_DIR / "raw/meta-info.csv"
OUTPUT_PATH = DATA_DIR / "processed/who_concerned.csv"


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
            result = drop_if_ungrounded(result, transcript)
        except Exception as e:
            print(f"  ERROR on {txt_path.name}: {e}")
            continue

        rows.append({
            "participant_id": participant_id,
            "diagnosis": diagnosis,
            "source_file": txt_path.name,
            "status": result.status,
            "who_concerned": result.who_concerned,
            "summary": result.summary,
            "evidence_quote": result.evidence_quote,
        })

        if i % 20 == 0:
            print(f"  {i}/{len(transcript_files)} done")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")
    print("\nStatus:")
    print(results_df["status"].value_counts())
    print("\nWho concerned:")
    print(results_df["who_concerned"].value_counts())


if __name__ == "__main__":
    main()
