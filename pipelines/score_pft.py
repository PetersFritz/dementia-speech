"""Score all PFT TextGrids and save results to processed data directory."""
import os
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
os.chdir(project_root)

from dementia_speech.io.textgrid import load_word_tier
from dementia_speech.tasks.phonemic_fluency.scoring import score_trial, compute_metrics
from dementia_speech.tasks.phonemic_fluency.features import compute_temporal_bins

TEXTGRID_DIR = Path("data/raw/textgrids/PFT")
METADATA_PATH = Path("data/raw/meta-info.csv")   # adjust path if needed
OUTPUT_WORDS = Path("data/processed/pft_words.csv")
OUTPUT_METRICS = Path("data/processed/pft_metrics.csv")
TARGET_LETTER = "p"


def extract_participant_id(filename: str) -> str:
    parts = filename.stem.split("__")
    return parts[1] if len(parts) >= 2 else filename.stem


def main():
    OUTPUT_WORDS.parent.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata = pd.read_csv(METADATA_PATH)

    all_words = []
    all_metrics = []

    textgrid_files = sorted(TEXTGRID_DIR.glob("*.TextGrid"))
    print(f"Found {len(textgrid_files)} TextGrid files")

    for tg_path in textgrid_files:
        participant_id = extract_participant_id(tg_path)
        print(f"Processing participant {participant_id}...")

        try:
            df = load_word_tier(tg_path)
            scored_df = score_trial(df, target_letter=TARGET_LETTER)
            metrics = compute_metrics(scored_df)
            bins = compute_temporal_bins(scored_df)

            # Add participant ID and diagnosis
            match = metadata[metadata['IDs'].astype(str).str.contains(participant_id)]
            diagnosis = match.iloc[0]['diagnosis'] if len(match) > 0 else 'Unknown'

            scored_df.insert(0, 'participant_id', participant_id)
            scored_df.insert(1, 'diagnosis', diagnosis)
            all_words.append(scored_df)

            metrics['participant_id'] = participant_id
            metrics['diagnosis'] = diagnosis
            metrics.update(bins)
            all_metrics.append(metrics)

        except Exception as e:
            print(f"  ERROR on {tg_path.name}: {e}")
            continue

    # Save
    words_df = pd.concat(all_words, ignore_index=True)
    words_df.to_csv(OUTPUT_WORDS, index=False)
    print(f"\nSaved word-level data: {OUTPUT_WORDS}")

    metrics_df = pd.DataFrame(all_metrics)
    cols = ['participant_id', 'diagnosis'] + [c for c in metrics_df.columns 
            if c not in ['participant_id', 'diagnosis']]
    metrics_df = metrics_df[cols]
    metrics_df.to_csv(OUTPUT_METRICS, index=False)
    print(f"Saved metrics: {OUTPUT_METRICS}")
    print("\nDone! Preview:")
    print(metrics_df.head())


if __name__ == "__main__":
    main()