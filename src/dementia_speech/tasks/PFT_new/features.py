"""Aggregate temporal features for phonemic fluency trials (Leslie code system)."""
import pandas as pd
import numpy as np


def compute_temporal_bins(
    scored_df: pd.DataFrame,
    trial_duration: float = 60.0,
    bin_size: float = 15.0,
) -> dict:
    """Compute correct ('C') word counts per time bin.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Scored trial (output from PFT_new.scoring.score_trial) with columns
        `start` and `classification`.
    trial_duration : float
        Total trial duration in seconds (default: 60)
    bin_size : float
        Bin size in seconds (default: 15)

    Returns
    -------
    dict
        Keys: bin_1, bin_2, bin_3, bin_4 (count of 'C'-classified words per bin)
    """
    correct = scored_df[scored_df['classification'] == 'C'].copy()

    bins = np.arange(0, trial_duration + bin_size, bin_size)
    bin_labels = [f"bin_{i+1}" for i in range(len(bins) - 1)]

    correct['bin'] = pd.cut(
        correct['start'],
        bins=bins,
        labels=bin_labels,
        right=True,
        include_lowest=True,
    )

    counts = correct['bin'].value_counts().reindex(bin_labels, fill_value=0)
    return counts.to_dict()


def compute_latency_features(scored_df: pd.DataFrame) -> dict:
    """Response-latency features from word-level timing -- candidate
    additional measures (not yet checked against C the way the rest of
    this task's measures were; see docs/pft_new_scoring.md "Response
    latency"). Literature: inter-word-interval duration has been linked to
    cognitive-impairment risk (Ayers et al., 2022, Alzheimer's & Dementia:
    DADM) and response timing more broadly to verbal/executive control in
    fluency tasks.

    time_to_first_correct -- onset of the first correct ('C') word, in
    seconds: a retrieval-initiation measure (how long before any real
    attempt lands).
    mean_iwi_correct -- mean inter-word interval (`iwi`: gap since the
    immediately preceding word, any classification -- see
    io/textgrid.py::load_word_tier) before each correct word: a
    word-finding-effort measure. The trial's very first word has no prior
    word to measure from (`iwi` is NaN there) and is dropped rather than
    treated as a zero-length pause.
    """
    correct = scored_df[scored_df['classification'] == 'C'].sort_values('start')
    if len(correct) == 0:
        return {"time_to_first_correct": None, "mean_iwi_correct": None}

    iwi = correct['iwi'].dropna()
    return {
        "time_to_first_correct": float(correct['start'].iloc[0]),
        "mean_iwi_correct": float(iwi.mean()) if len(iwi) else None,
    }
