"""Score all PFT TextGrids with the rebuilt (PFT_new) scoring rules."""
import os
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
os.chdir(project_root)

from dementia_speech.io.textgrid import load_word_tier
from dementia_speech.tasks.PFT_new.scoring import score_trial

TEXTGRID_DIR = Path("data/raw/textgrids/PFT")
METADATA_PATH = Path("data/raw/meta-info.csv")
OUTPUT_WORDS = Path("data/processed/pft_words_new.csv")
TARGET_LETTER = "p"


def extract_participant_id(filename: str) -> str:
    parts = filename.stem.split("__")
    return parts[1] if len(parts) >= 2 else filename.stem


def main():
    OUTPUT_WORDS.parent.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(METADATA_PATH)

    all_words = []

    textgrid_files = sorted(TEXTGRID_DIR.glob("*.TextGrid"))
    print(f"Found {len(textgrid_files)} TextGrid files")

    for tg_path in textgrid_files:
        participant_id = extract_participant_id(tg_path)

        try:
            df = load_word_tier(tg_path)
            if len(df) == 0:
                continue
            scored_df = score_trial(df, target_letter=TARGET_LETTER)

            match = metadata[metadata['IDs'].astype(str).str.contains(participant_id)]
            diagnosis = match.iloc[0]['diagnosis'] if len(match) > 0 else 'Unknown'

            scored_df.insert(0, 'participant_id', participant_id)
            scored_df.insert(1, 'diagnosis', diagnosis)
            scored_df.insert(2, 'source_file', tg_path.name)
            all_words.append(scored_df)

        except Exception as e:
            print(f"  ERROR on {tg_path.name}: {e}")
            continue

    words_df = pd.concat(all_words, ignore_index=True)
    words_df.to_csv(OUTPUT_WORDS, index=False)
    print(f"\nSaved word-level data: {OUTPUT_WORDS}")
    print(f"Total words: {len(words_df)}")
    print("\nClassification counts:")
    print(words_df['classification'].value_counts())


if __name__ == "__main__":
    main()
