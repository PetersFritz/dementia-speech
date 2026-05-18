"""Score phonemic fluency trials."""
import pandas as pd
from pathlib import Path
from dementia_speech.tasks.phonemic_fluency.rules import classify_word


def score_trial(
    words_df: pd.DataFrame,
    target_letter: str = "p"
) -> pd.DataFrame:
    """Score a phonemic fluency trial.
    
    Adds a 'classification' column to the dataframe with values:
    - 'correct': valid word starting with target letter
    - 'intrusion': doesn't start with target letter  
    - 'repetition': exact duplicate of earlier word
    - 'variant': morphological variant of earlier word
    - 'nonword': not a valid English word (optional, requires wordnet)
    
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
        Input dataframe with added 'classification' column
    """
    if len(words_df) == 0:
        words_df['classification'] = []
        return words_df
    
    classifications = []
    previous_words = []
    
    for idx, row in words_df.iterrows():
        word = row['word']
        classification = classify_word(word, previous_words, target_letter)
        classifications.append(classification)
        previous_words.append(word)
    
    words_df['classification'] = classifications
    return words_df


def compute_metrics(scored_df: pd.DataFrame) -> dict:
    """Compute summary metrics from a scored trial.
    
    Parameters
    ----------
    scored_df : pd.DataFrame
        Scored trial (output from score_trial)
        
    Returns
    -------
    dict
        Metrics including:
        - total_words: total number of words produced
        - correct: number of correct words
        - intrusions: number of intrusions
        - repetitions: number of repetitions
        - variants: number of variants
        - nonwords: number of non-words
        - errors: total errors (intrusions + repetitions + variants + nonwords)
    """
    if len(scored_df) == 0:
        return {
            'total_words': 0,
            'correct': 0,
            'intrusions': 0,
            'repetitions': 0,
            'variants': 0,
            'nonwords': 0,
            'errors': 0
        }
    
    counts = scored_df['classification'].value_counts().to_dict()
    
    correct = counts.get('correct', 0)
    intrusions = counts.get('intrusion', 0)
    repetitions = counts.get('repetition', 0)
    variants = counts.get('variant', 0)
    nonwords = counts.get('nonword', 0)
    
    return {
        'total_words': len(scored_df),
        'correct': correct,
        'intrusions': intrusions,
        'repetitions': repetitions,
        'variants': variants,
        'nonwords': nonwords,
        'errors': intrusions + repetitions + variants + nonwords
    }