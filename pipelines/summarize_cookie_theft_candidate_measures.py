"""Rank BOTH the new candidate measures and the existing Cookie Theft
measures (cookie_theft_metrics.csv -- iu_score, mattr, avg_x,
path_distance_per_100_words, etc.) by how well they separate diagnosis
groups, as a first-pass triage before deciding what the final "Additional
measures" list should be. Joining both files here (rather than ranking the
candidates alone, as the first version of this script did) is what makes
the comparison against the three current HEADLINE_MEASURES
(interpretation.py) actually apples-to-apples, instead of relying on
reasoning about overlap/correlation.

This is descriptive screening on the full, unsplit sample, not a validated
cutoff/classifier and NOT the same TRAIN/TEST discipline
pipelines/check_cookie_theft_measure_separation.py used to pick the current
three headline measures (same caution as the ROC exercise dropped from the
main scoring -- see interpretation.py) -- a measure separating groups here
is a reason to look closer, not a verdict.

Writes a ranked summary CSV; prints the same table to stdout (participant-
level rows never leave the data dir -- only these group-level aggregates do).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from dementia_speech.paths import DATA_DIR

CANDIDATE_PATH = DATA_DIR / "processed/cookie_theft_candidate_measures.csv"
EXISTING_PATH = DATA_DIR / "processed/cookie_theft_metrics.csv"
OUTPUT_PATH = DATA_DIR / "processed/cookie_theft_candidate_measures_summary.csv"

NON_MEASURE_COLS = {
    "participant_id", "diagnosis", "content_units_by_region",
    "has_pause_annotations", "has_sentence_punctuation",
    "has_other_speaker_in_transcript", "wav_missing", "audio_load_error",
    "correct_information_units",  # alias of total_content_units -- see report
    "content_accuracy_ciu_per_100_words",  # alias of iu_mentions_per_100_words
    "total_ius", "total_mentions", "mattr_window",
    "iu_sequence_unique", "iu_sequence_full", "iu_sequence_readable",
}


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return np.nan
    pooled_sd = np.sqrt(((n_a - 1) * a.std() ** 2 + (n_b - 1) * b.std() ** 2) / (n_a + n_b - 2))
    if pooled_sd == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled_sd


def auc_separation(a: pd.Series, b: pd.Series) -> float:
    """Mann-Whitney U rescaled to AUC (0.5 = no separation, 1.0 or 0.0 =
    perfect separation in one direction), direction-agnostic via distance
    from 0.5 -- a measure can be diagnostic whichever way its raw scale
    points."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    auc = u / (len(a) * len(b))
    return max(auc, 1 - auc)


def main():
    candidates = pd.read_csv(CANDIDATE_PATH)
    existing = pd.read_csv(EXISTING_PATH).drop(columns=["diagnosis"])
    df = candidates.merge(existing, on="participant_id", how="left", suffixes=("", "_existing"))

    measure_cols = [
        c for c in df.columns
        if c not in NON_MEASURE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]

    hc = df[df["diagnosis"] == "HC"]
    dem = df[df["diagnosis"] == "Dementia"]
    mci = df[df["diagnosis"] == "MCI"]

    existing_cols = set(existing.columns) - {"participant_id"}
    headline_keys = {"iu_score", "total_words", "path_distance_per_100_words"}

    rows = []
    for col in measure_cols:
        rows.append({
            "measure": col,
            "source": "headline" if col in headline_keys
                      else "existing" if col in existing_cols
                      else "new_candidate",
            "n_non_null": df[col].notna().sum(),
            "hc_mean": hc[col].mean(),
            "mci_mean": mci[col].mean(),
            "dementia_mean": dem[col].mean(),
            "cohens_d_hc_vs_dementia": cohens_d(hc[col], dem[col]),
            "auc_hc_vs_dementia": auc_separation(hc[col], dem[col]),
            "auc_hc_vs_mci": auc_separation(hc[col], mci[col]),
        })

    summary = pd.DataFrame(rows).sort_values("auc_hc_vs_dementia", ascending=False)
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")
    print()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
