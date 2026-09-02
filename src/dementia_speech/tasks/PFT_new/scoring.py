"""Score phonemic fluency trials (rebuild against the Leslie scoring criteria)."""
import pandas as pd
from dementia_speech.tasks.PFT_new.rules import (
    starts_with_target_letter,
    is_real_word,
    is_person_name,
    is_country_name,
    find_country_phrase_indices,
    is_repetition,
    is_similar_form_or_meaning,
    is_filler,
)

# Leslie codes this pipeline can currently assign: C, OTH, N, PN, R, SIM, F.
# Not yet implemented: SC_CORR, LP, FRAG, SC_ERR, SC_META, META, TG.
# 'L' (wrong-letter real word) is deliberately merged into 'OTH' here --
# any word not starting with the target letter (including nonwords and
# discourse markers like "well"/"so"/"right") is scored 'OTH' rather than
# kept as a separate error code. 'N' (nonword) is reserved for words that
# DO start with the target letter but aren't recognised as real words --
# i.e. failed attempts at the target letter, not off-target speech.
# Bracketed sounds, BGN and UNCLEAR ('?') are dropped upstream in
# load_word_tier rather than scored here.
_CLASSIFICATION_COLUMN = 'classification'


def score_trial(
    words_df: pd.DataFrame,
    target_letter: str = "p"
) -> pd.DataFrame:
    """Score a phonemic fluency trial against the Leslie code system.

    Classification hierarchy, checked per word in production order:
    1. Filled pause -> 'F'
    2. Doesn't start with target letter -> 'OTH' (other/non-attempt speech,
       regardless of whether it's a real word, function word, or nonword)
    3. Person name or country name -> 'PN' (proper noun)
    4. Exact repeat of an earlier 'C' word -> 'R'
    5. Morphological variant / synonym of an earlier 'C' word -> 'SIM'
    6. Real word -> 'C' (correct), added to the running correct-words history
    7. Otherwise -> 'N' (nonword)

    R and SIM are checked against only the words already classified 'C'
    earlier in the same trial (not against errors), per the Leslie
    definition of R as "repetition of previous correct word".

    Parameters
    ----------
    words_df : pd.DataFrame
        DataFrame with columns: word, start, end, duration, iwi
        (output from load_word_tier)
    target_letter : str
        Target letter for the trial (default: "p")

    Returns
    -------
    pd.DataFrame
        Input dataframe with added boolean feature columns and a final
        'classification' column holding one Leslie code per word.
    """
    if len(words_df) == 0:
        for col in ['starts_with_target', 'is_real_word', 'is_person_name', 'is_country_name', _CLASSIFICATION_COLUMN]:
            words_df[col] = []
        return words_df

    words_df = words_df.copy()
    words_df['starts_with_target'] = words_df['word'].apply(
        lambda w: starts_with_target_letter(w, target_letter)
    )

    words_df['is_real_word'] = words_df['word'].apply(is_real_word)

    words_df['is_person_name'] = words_df['word'].apply(is_person_name)

    words_df['is_country_name'] = words_df['word'].apply(is_country_name)

    phrase_indices = find_country_phrase_indices(words_df['word'].tolist())
    words_df.loc[list(phrase_indices), 'is_country_name'] = True

    classifications = []
    correct_words: list[str] = []

    for _, row in words_df.iterrows():
        word = row['word']

        if is_filler(word):
            code = 'F'
        elif not row['starts_with_target']:
            code = 'OTH'
        elif row['is_person_name'] or row['is_country_name']:
            code = 'PN'
        elif is_repetition(word, correct_words):
            code = 'R'
        elif is_similar_form_or_meaning(word, correct_words):
            code = 'SIM'
        elif row['is_real_word']:
            code = 'C'
        else:
            code = 'N'

        if code == 'C':
            correct_words.append(word)

        classifications.append(code)

    words_df[_CLASSIFICATION_COLUMN] = classifications

    return words_df


def compute_metrics(scored_df: pd.DataFrame) -> dict:
    """Compute summary counts per Leslie code for a scored trial.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Scored trial (output from score_trial)

    Returns
    -------
    dict
        total_words, per-code counts (C, OTH, N, PN, R, SIM, F), and errors
        (everything except 'C', 'F' and 'OTH' -- filled pauses and other/
        non-attempt speech are neither correct responses nor scoring errors)
    """
    if len(scored_df) == 0:
        return {
            'total_words': 0,
            'C': 0, 'OTH': 0, 'N': 0, 'PN': 0, 'R': 0, 'SIM': 0, 'F': 0,
            'errors': 0,
        }

    counts = scored_df[_CLASSIFICATION_COLUMN].value_counts().to_dict()
    correct = counts.get('C', 0)
    fillers = counts.get('F', 0)
    other = counts.get('OTH', 0)

    return {
        'total_words': len(scored_df),
        'C': correct,
        'OTH': other,
        'N': counts.get('N', 0),
        'PN': counts.get('PN', 0),
        'R': counts.get('R', 0),
        'SIM': counts.get('SIM', 0),
        'F': fillers,
        'errors': len(scored_df) - correct - fillers - other,
    }

