"""Aggregate features for phonemic fluency trials."""
import pandas as pd
import numpy as np


def compute_temporal_bins(
    scored_df: pd.DataFrame,
    trial_duration: float = 60.0,
    bin_size: float = 15.0,
) -> dict:
    """Compute correct word counts per time bin.
    
    Parameters
    ----------
    scored_df : pd.DataFrame
        Scored trial with columns: word, start, classification
    trial_duration : float
        Total trial duration in seconds (default: 60)
    bin_size : float
        Bin size in seconds (default: 15)
        
    Returns
    -------
    dict
        Keys: bin_1, bin_2, bin_3, bin_4 (correct word count per bin)
    """
    correct = scored_df[scored_df['classification'] == 'correct'].copy()
    
    bins = np.arange(0, trial_duration + bin_size, bin_size)
    bin_labels = [f"bin_{i+1}" for i in range(len(bins) - 1)]
    
    correct['bin'] = pd.cut(
        correct['start'],
        bins=bins,
        labels=bin_labels,
        right=True,
        include_lowest=True
    )
    
    counts = correct['bin'].value_counts().reindex(bin_labels, fill_value=0)
    return counts.to_dict()