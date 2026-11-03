"""Generic lexical reference norms shared across tasks -- not task-specific
scoring logic, just a word lookup, used by both PFT_new (letter fluency) and
SFT_new (category fluency).

Age-of-acquisition (AoA) norms (Kuperman et al., 2012, Behavior Research
Methods). Mean AoA rating is in years.
"""
from pathlib import Path

import pandas as pd

from dementia_speech.paths import DATA_DIR

_AOA_PATH = DATA_DIR / "raw/reference/Kuperman2012.csv"

_AOA_WORD_COLS = {"word", "word_english"}
_AOA_RATING_COLS = {"rating.mean", "aoa_mean"}


def _load_aoa_norms(path: Path = _AOA_PATH) -> dict[str, float]:
    if not path.exists():
        return {}

    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)

    word_col = next((c for c in df.columns if str(c).strip().lower() in _AOA_WORD_COLS), None)
    aoa_col = next((c for c in df.columns if str(c).strip().lower() in _AOA_RATING_COLS), None)
    if word_col is None or aoa_col is None:
        raise ValueError(
            f"Expected a word column ({_AOA_WORD_COLS}) and an AoA rating column "
            f"({_AOA_RATING_COLS}) in {path}, found: {list(df.columns)}"
        )

    norms = df[[word_col, aoa_col]].dropna()
    norms[word_col] = norms[word_col].astype(str).str.strip().str.lower()
    return dict(zip(norms[word_col], norms[aoa_col].astype(float)))


AOA_NORMS = _load_aoa_norms()


def get_aoa(word: str) -> float | None:
    """Look up the mean age-of-acquisition rating (years) for a word.

    Returns None if the AoA norms file isn't present, or the word isn't in
    it -- including any multi-word phrase, since the norms are single-word
    only (e.g. a correctly-named "guinea pig" in SFT has no AoA rating).
    """
    return AOA_NORMS.get(word.lower().strip())
