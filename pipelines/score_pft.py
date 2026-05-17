"""Score all PFT TextGrids and save results to processed data directory."""
import os
from pathlib import Path
import pandas as pd

# Change to project root if needed
project_root = Path(__file__).parent.parent
os.chdir(project_root)

from dementia_speech.io.textgrid import load_word_tier
from dementia_speech.tasks.phonemic_fluency.scoring import score_trial, compute_metrics

TEXTGRID_DIR = Path("data/raw/textgrids/PFT")
OUTPUT_WORDS = Path("data/processed/pft_words.csv")
OUTPUT_METRICS = Path("data/processed/pft_metrics.csv")
TARGET_LETTER = "p"

def extract_participant_id(filename: str) -> str:
    """Extract participant ID from filename.
    e.g. PROCESS-2_rec__033__PFT.TextGrid -> 033
    """
    parts = filename.stem.split("__")
    return parts[1] if len(parts) >= 2 else filename.stem


def main():
    OUTPUT_WORDS.parent.mkdir(parents=True, exist_ok=True)

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

            # Add participant ID to words dataframe
            scored_df.insert(0, "participant_id", participant_id)
            all_words.append(scored_df)

            # Add participant ID to metrics
            metrics["participant_id"] = participant_id
            all_metrics.append(metrics)

        except Exception as e:
            print(f"  ERROR on {tg_path.name}: {e}")
            continue

    # Save
    words_df = pd.concat(all_words, ignore_index=True)
    words_df.to_csv(OUTPUT_WORDS, index=False)
    print(f"\nSaved word-level data: {OUTPUT_WORDS} ({len(words_df)} rows)")

    metrics_df = pd.DataFrame(all_metrics)
    # Put participant_id first
    cols = ["participant_id"] + [c for c in metrics_df.columns if c != "participant_id"]
    metrics_df = metrics_df[cols]
    metrics_df.to_csv(OUTPUT_METRICS, index=False)
    print(f"Saved metrics: {OUTPUT_METRICS} ({len(metrics_df)} rows)")

    print("\nDone! Preview of metrics:")
    print(metrics_df.head())


if __name__ == "__main__":
    main()

# to run enter the following command in terminal:
# python pipelines/score_pft.py