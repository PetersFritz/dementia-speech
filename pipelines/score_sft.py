"""Score all SFT (semantic/category fluency) trials from the manually-transcribed
transcripts in data/raw/process/SFT/ -- NOT the Whisper transcripts in
data/raw/text/SFT/, which lack the speaker tags and pause/event annotations
this scoring relies on. See SFT_new/scoring.py for the classification rules.
"""
import os
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
os.chdir(project_root)

from dementia_speech.tasks.SFT_new.scoring import compute_metrics, score_trial

TRANSCRIPT_DIR = Path("data/raw/process/SFT")
METADATA_PATH = Path("data/raw/meta-info.csv")
OUTPUT_WORDS = Path("data/processed/sft_words.csv")
OUTPUT_METRICS = Path("data/processed/sft_metrics.csv")


def extract_participant_id(path: Path) -> str:
    parts = path.stem.split("__")
    return parts[1] if len(parts) >= 2 else path.stem


def main():
    OUTPUT_WORDS.parent.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(METADATA_PATH)

    all_words = []
    all_metrics = []

    transcript_files = sorted(TRANSCRIPT_DIR.glob("*.txt"))
    print(f"Found {len(transcript_files)} SFT transcripts")

    for path in transcript_files:
        participant_id = extract_participant_id(path)
        raw_text = path.read_text(encoding="utf-8", errors="replace")

        scored_df, oth_turn_count = score_trial(raw_text)
        metrics = compute_metrics(scored_df, oth_turn_count)

        match = metadata[metadata["IDs"].astype(str).str.contains(participant_id)]
        diagnosis = match.iloc[0]["diagnosis"] if len(match) > 0 else "Unknown"

        scored_df.insert(0, "participant_id", participant_id)
        scored_df.insert(1, "diagnosis", diagnosis)
        scored_df.insert(2, "source_file", path.name)
        all_words.append(scored_df)

        metrics["participant_id"] = participant_id
        metrics["diagnosis"] = diagnosis
        all_metrics.append(metrics)

    words_df = pd.concat(all_words, ignore_index=True)
    words_df.to_csv(OUTPUT_WORDS, index=False)
    print(f"\nSaved word-level data: {OUTPUT_WORDS}")
    print(f"Total items: {len(words_df)}")
    print("\nClassification counts:")
    print(words_df["classification"].value_counts())

    metrics_df = pd.DataFrame(all_metrics)
    cols = ["participant_id", "diagnosis"] + [
        c for c in metrics_df.columns if c not in ("participant_id", "diagnosis")
    ]
    metrics_df = metrics_df[cols]
    metrics_df.to_csv(OUTPUT_METRICS, index=False)
    print(f"Saved metrics: {OUTPUT_METRICS}")


if __name__ == "__main__":
    main()
