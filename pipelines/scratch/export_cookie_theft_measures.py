"""Export a compact spreadsheet of Cookie Theft measures: one row per
participant, with just the train/test split and diagnosis as identifying
columns, then every computed measure -- no other demographics, for a quick
measures-only view (e.g. to hand to a stats package).

Columns: participant_id, split, diagnosis, then every other column from
cookie_theft_metrics.csv except the three sequence columns
(iu_sequence_unique/iu_sequence_full/iu_sequence_readable) -- those are
JSON/string sequences, not scalar measures, and don't display usefully in
a spreadsheet cell (e.g. '["Boy", "Cookie", "Jar"]"). They're still in
cookie_theft_metrics.csv itself for anything that needs them.
"""
import os
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent.parent
os.chdir(project_root)

from dementia_speech.paths import DATA_DIR

METRICS_PATH  = DATA_DIR / "processed" / "cookie_theft_metrics.csv"
METADATA_PATH = DATA_DIR / "raw" / "meta-info.csv"
OUTPUT_PATH   = DATA_DIR / "processed" / "cookie_theft_measures_export.csv"

# Sequence/string columns, not scalar measures -- dropped from this export.
_NON_MEASURE_COLUMNS = ["iu_sequence_unique", "iu_sequence_full", "iu_sequence_readable"]


def lookup_split(metadata: pd.DataFrame, participant_id) -> str | None:
    """Same convention as app/shared.py's SPLIT lookup: zero-pad to
    meta-info.csv's 3-digit id format and match by substring (its IDs
    column holds a longer string, e.g. "PROCESS-2_rec__002", that contains
    the padded id).
    """
    key = f"{int(participant_id):03d}"
    match = metadata[metadata["IDs"].astype(str).str.contains(key)]
    return match.iloc[0]["Split"] if len(match) else None


def main():
    metrics = pd.read_csv(METRICS_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    metrics["split"] = metrics["participant_id"].apply(lambda pid: lookup_split(metadata, pid))
    missing_split = metrics["split"].isna().sum()
    if missing_split:
        print(f"WARNING: {missing_split} participant(s) have no Split match in {METADATA_PATH.name}")

    measure_columns = [
        c for c in metrics.columns
        if c not in ("participant_id", "diagnosis", "split")
        and c not in _NON_MEASURE_COLUMNS
    ]
    ordered_columns = ["participant_id", "split", "diagnosis"] + measure_columns
    export = metrics[ordered_columns]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {len(export)} rows x {len(ordered_columns)} columns to {OUTPUT_PATH}")
    print(f"\nSplit counts:\n{export['split'].value_counts(dropna=False)}")
    print(f"\nDiagnosis counts:\n{export['diagnosis'].value_counts(dropna=False)}")


if __name__ == "__main__":
    main()
