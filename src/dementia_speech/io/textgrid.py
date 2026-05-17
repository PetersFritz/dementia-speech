"""Parse Praat TextGrids into tidy word-level dataframes."""
from pathlib import Path
import pandas as pd
from praatio import textgrid


def load_word_tier(
    path: Path,
    word_tier_name: str = "transcript/word",
) -> pd.DataFrame:
    """Load a TextGrid and return one row per non-empty interval on the word tier.

    Returns columns: word, start, end, duration.
    """
    tg = textgrid.openTextgrid(str(path), includeEmptyIntervals=False)
    if word_tier_name not in tg.tierNames:
        raise ValueError(
            f"Tier '{word_tier_name}' not found in {path.name}. "
            f"Available tiers: {tg.tierNames}"
        )
    tier = tg.getTier(word_tier_name)
    rows = [
        {"word": label.strip(), "start": start, "end": end, "duration": end - start}
        for start, end, label in tier.entries
        if label.strip()
    ]
    df = pd.DataFrame(rows)
    df["iwi"] = df["start"].diff()  # inter-word interval to previous word
    return df