"""Score all Cookie Theft transcripts and save results."""
import os
from pathlib import Path
import pandas as pd
import json

project_root = Path(__file__).parent.parent
os.chdir(project_root)

from dementia_speech.tasks.picture_description.scoring import (
    load_transcript,
    extract_information_units,
    compute_iu_metrics,
)

TRANSCRIPT_DIR = Path("data/raw/text/CTD")  # adjust to your path
METADATA_PATH  = Path("data/raw/meta-info.csv")
OUTPUT_IU      = Path("data/processed/cookie_theft_iu.csv")
OUTPUT_METRICS = Path("data/processed/cookie_theft_metrics.csv")


def extract_participant_id(filename: str) -> str:
    """Extract 3-digit participant ID from filename — adjust to your naming convention."""
    parts = filename.stem.split("__")
    return parts[1] if len(parts) >= 2 else filename.stem


def main():
    OUTPUT_IU.parent.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(METADATA_PATH)

    all_iu_rows = []
    all_metrics = []

    txt_files = sorted(TRANSCRIPT_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} transcript files")

    for txt_path in txt_files:
        participant_id = extract_participant_id(txt_path)
        print(f"Processing participant {participant_id}...")

        try:
            transcript = load_transcript(txt_path)
            iu_results = extract_information_units(transcript)
            metrics    = compute_iu_metrics(iu_results)

            # Diagnosis lookup
            match = metadata[metadata["IDs"].astype(str).str.contains(participant_id)]
            diagnosis = match.iloc[0]["diagnosis"] if len(match) > 0 else "Unknown"

            # IU-level rows (one row per IU)
            for iu_name, result in iu_results.items():
                all_iu_rows.append({
                    "participant_id": participant_id,
                    "diagnosis":      diagnosis,
                    "iu":             iu_name,
                    "present":        result["present"],
                    "evidence":       ", ".join(result["evidence"]),
                })

            # Metrics row
            metrics["participant_id"] = participant_id
            metrics["diagnosis"]      = diagnosis
            all_metrics.append(metrics)

        except Exception as e:
            print(f"  ERROR on {txt_path.name}: {e}")
            continue

    iu_df = pd.DataFrame(all_iu_rows)
    iu_df.to_csv(OUTPUT_IU, index=False)
    print(f"\nSaved IU data: {OUTPUT_IU} ({len(iu_df)} rows)")

    metrics_df = pd.DataFrame(all_metrics)
    cols = ["participant_id", "diagnosis"] + [
        c for c in metrics_df.columns if c not in ["participant_id", "diagnosis"]
    ]
    metrics_df = metrics_df[cols]
    metrics_df.to_csv(OUTPUT_METRICS, index=False)
    print(f"Saved metrics: {OUTPUT_METRICS} ({len(metrics_df)} rows)")
    print("\nDone! Preview:")
    print(metrics_df.head())


if __name__ == "__main__":
    main()