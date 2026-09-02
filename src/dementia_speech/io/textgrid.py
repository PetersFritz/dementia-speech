"""Parse Praat TextGrids into tidy word-level dataframes."""
from pathlib import Path
import re
import pandas as pd
from praatio import textgrid

from dementia_speech.io.cleaning import clean_token


def load_word_tier(
    path: Path,
    word_tier_name: str = "transcript/word",
) -> pd.DataFrame:
    """Load a TextGrid and return one row per non-empty interval on the word tier.

    Excludes:
    - Empty or whitespace-only intervals
    - Annotations in square brackets e.g. [laugh], [sigh]

    Filled pauses (er, um, uh, ...) and discourse markers (well, so, right,
    yeah, no, ...) are NOT excluded here -- they're kept as ordinary rows
    and classified downstream (e.g. PFT_new.scoring assigns filled pauses
    the Leslie 'F' code; discourse markers fall through the normal
    classification hierarchy).

    Returns columns: word, start, end, duration, iwi.
    """
    tg = textgrid.openTextgrid(str(path), includeEmptyIntervals=False)
    if word_tier_name not in tg.tierNames:
        raise ValueError(
            f"Tier '{word_tier_name}' not found in {path.name}. "
            f"Available tiers: {tg.tierNames}"
        )
    tier = tg.getTier(word_tier_name)

    rows = []
    for start, end, label in tier.entries:
        # Skip bracketed annotations [laugh], [sigh] etc.
        if re.match(r'^\[.*\]$', label.strip()):
            continue

        cleaned = clean_token(label)

        # Skip empty after cleaning
        if not cleaned:
            continue

        rows.append({
            "word": cleaned,
            "start": start,
            "end": end,
            "duration": end - start,
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["iwi"] = df["start"].diff()
    return df
