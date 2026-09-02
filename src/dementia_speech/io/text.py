"""Parse plain-text transcripts into tidy word-level dataframes."""
from pathlib import Path
import pandas as pd

from dementia_speech.io.cleaning import clean_token


def load_words_from_txt(path: Path) -> pd.DataFrame:
    """Load a plain-text transcript and return one row per word.

    Expects the file to contain only the participant's spoken words,
    whitespace-separated (no speaker labels, timestamps, or bracketed
    annotations). Tokens are cleaned the same way as load_word_tier
    (punctuation stripped, internal hyphens kept, lowercased).

    Plain text carries no timing, so unlike load_word_tier this returns
    only a 'word' column -- no start/end/duration/iwi. score_trial only
    reads 'word', so scoring works the same either way.
    """
    tokens = Path(path).read_text().split()
    words = [clean_token(t) for t in tokens]
    words = [w for w in words if w]
    return pd.DataFrame({"word": words})
