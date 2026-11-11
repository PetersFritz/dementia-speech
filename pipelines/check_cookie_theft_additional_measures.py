"""Decide the full "Additional measures" section from scratch (existing
exploratory measures AND new candidates on equal footing), using the same
TRAIN/TEST discipline score_cookie_theft's sibling analyses already use
(meta-info.csv's "Split" column -- see
pipelines/scratch/export_cookie_theft_measures.py's lookup_split) rather than
a single full-sample AUC, which can't tell a stable effect from a fluke or a
reversed-direction one.

For each measure: Cohen's d (HC vs Dementia, HC vs MCI) computed SEPARATELY
in TRAIN and TEST. A measure only counts as a real candidate if the sign is
the same in both splits for HC vs Dementia (same bar
check_cookie_theft_measure_separation.py's docstring describes for the
original headline/exploratory decisions) -- that's what catches e.g.
cycles_per_100_words, which already turned out to run backwards.

participant-level rows never leave the data dir -- only these group-level
aggregates do.
"""
import numpy as np
import pandas as pd

from dementia_speech.paths import DATA_DIR

CANDIDATE_PATH = DATA_DIR / "processed/cookie_theft_candidate_measures.csv"
EXISTING_PATH = DATA_DIR / "processed/cookie_theft_metrics.csv"
METADATA_PATH = DATA_DIR / "raw/meta-info.csv"
OUTPUT_PATH = DATA_DIR / "processed/cookie_theft_additional_measures_check.csv"

NON_MEASURE_COLS = {
    "participant_id", "diagnosis", "split", "content_units_by_region",
    "has_pause_annotations", "has_sentence_punctuation",
    "has_other_speaker_in_transcript", "wav_missing", "audio_load_error",
    "correct_information_units", "content_accuracy_ciu_per_100_words",
    "total_ius", "total_mentions", "mattr_window",
    "iu_sequence_unique", "iu_sequence_full", "iu_sequence_readable",
    # user-excluded families
    "num_utterances", "mean_length_utterance",
    "pause_count", "total_pause_seconds",
}

HEADLINE_KEYS = {"iu_score", "total_words", "path_distance_per_100_words"}


def lookup_split(metadata: pd.DataFrame, participant_id) -> str | None:
    key = f"{int(participant_id):03d}"
    match = metadata[metadata["IDs"].astype(str).str.contains(key)]
    return match.iloc[0]["Split"] if len(match) else None


def cohens_d(a: pd.Series, b: pd.Series) -> float | None:
    a, b = a.dropna(), b.dropna()
    n_a, n_b = len(a), len(b)
    if n_a < 5 or n_b < 5:
        return None
    pooled_sd = np.sqrt(((n_a - 1) * a.std() ** 2 + (n_b - 1) * b.std() ** 2) / (n_a + n_b - 2))
    if not pooled_sd:
        return None
    return (a.mean() - b.mean()) / pooled_sd  # positive = HC higher


def main():
    candidates = pd.read_csv(CANDIDATE_PATH)
    existing = pd.read_csv(EXISTING_PATH).drop(columns=["diagnosis"])
    df = candidates.merge(existing, on="participant_id", how="left", suffixes=("", "_dup"))
    df = df[[c for c in df.columns if not c.endswith("_dup")]]

    metadata = pd.read_csv(METADATA_PATH)
    df["split"] = df["participant_id"].apply(lambda pid: lookup_split(metadata, pid))

    measure_cols = [
        c for c in df.columns
        if c not in NON_MEASURE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]

    rows = []
    for col in measure_cols:
        d_by_split = {}
        d_mci_by_split = {}
        for split in ("TRAIN", "TEST"):
            sub = df[df["split"] == split]
            hc, dem, mci = sub[sub.diagnosis == "HC"][col], sub[sub.diagnosis == "Dementia"][col], sub[sub.diagnosis == "MCI"][col]
            d_by_split[split] = cohens_d(hc, dem)
            d_mci_by_split[split] = cohens_d(hc, mci)

        d_train, d_test = d_by_split["TRAIN"], d_by_split["TEST"]
        stable = (
            d_train is not None and d_test is not None
            and np.sign(d_train) == np.sign(d_test) and abs(d_train) > 0.1 and abs(d_test) > 0.1
        )
        rows.append({
            "measure": col,
            "role": "headline" if col in HEADLINE_KEYS else ("existing_exploratory" if col in set(existing.columns) else "new_candidate"),
            "d_hc_dementia_TRAIN": round(d_train, 3) if d_train is not None else None,
            "d_hc_dementia_TEST": round(d_test, 3) if d_test is not None else None,
            "d_hc_mci_TRAIN": round(d_mci_by_split["TRAIN"], 3) if d_mci_by_split["TRAIN"] is not None else None,
            "d_hc_mci_TEST": round(d_mci_by_split["TEST"], 3) if d_mci_by_split["TEST"] is not None else None,
            "stable_direction_hc_vs_dementia": stable,
        })

    summary = pd.DataFrame(rows).sort_values(
        ["stable_direction_hc_vs_dementia", "role"], ascending=[False, True]
    )
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"\nSplit sizes:\n{df.groupby(['split','diagnosis']).size()}\n")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
