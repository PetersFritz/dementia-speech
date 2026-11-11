"""Add assessment (recording) dates to Process_demographics.csv for new-CognoSpeak participants.

Joins `CognoSpeak_metadata__2026_06_07.csv` (column `dir_name`, matched against
`Process_demographics.csv`'s `best_match_id`) onto Process_demographics.csv, pulling in that
file's `asses_date` column as a new `assessment_date` column (ISO `YYYY-MM-DD`). Only rows whose
`best_match_id` has a match in the metadata file get a date -- old-CognoSpeak participants have
no `dir_name` to match against and are expected to stay blank.

Writes Process_demographics.csv.bak (one-time backup of the pre-join file, never overwritten if
it already exists) before overwriting Process_demographics.csv in place. Pure CSV join, no
transcripts or GPU involved -- safe to run on a login node.
"""
import os
import shutil
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent.parent
os.chdir(project_root)

from dementia_speech.paths import DATA_DIR

DEMOGRAPHICS_PATH = DATA_DIR / "raw/Process_demographics.csv"
COGNOSPEAK_METADATA_PATH = DATA_DIR / "raw/CognoSpeak_metadata__2026_06_07.csv"
BACKUP_PATH = DEMOGRAPHICS_PATH.with_suffix(".csv.bak")

MATCH_COLUMN = "best_match_id"   # in Process_demographics.csv
META_KEY_COLUMN = "dir_name"     # in the CognoSpeak metadata csv
META_DATE_COLUMN = "assess_date"  # in the CognoSpeak metadata csv
NEW_COLUMN = "assessment_date"


def build_date_map(meta: pd.DataFrame) -> dict[str, str]:
    """{dir_name: ISO date string}, skipping dir_names with inconsistent dates."""
    dates = pd.to_datetime(meta[META_DATE_COLUMN], errors="coerce")
    bad_dates = meta[META_DATE_COLUMN].notna() & dates.isna()
    if bad_dates.any():
        print(f"  WARNING: {bad_dates.sum()} rows have an unparseable {META_DATE_COLUMN!r}, skipped:")
        for value in meta.loc[bad_dates, META_DATE_COLUMN].unique():
            print(f"    {value!r}")

    parsed = meta.assign(_date=dates.dt.strftime("%Y-%m-%d"))
    parsed = parsed.dropna(subset=["_date", META_KEY_COLUMN])

    date_map = {}
    for dir_name, group in parsed.groupby(META_KEY_COLUMN):
        unique_dates = group["_date"].unique()
        if len(unique_dates) > 1:
            print(f"  WARNING: {dir_name!r} has {len(unique_dates)} different dates {list(unique_dates)}, skipped")
            continue
        date_map[dir_name] = unique_dates[0]
    return date_map


def main():
    if MATCH_COLUMN not in pd.read_csv(DEMOGRAPHICS_PATH, nrows=0).columns:
        available = pd.read_csv(DEMOGRAPHICS_PATH, nrows=0).columns.tolist()
        raise SystemExit(
            f"{MATCH_COLUMN!r} not found in {DEMOGRAPHICS_PATH}. Available columns: {available}"
        )

    demographics = pd.read_csv(DEMOGRAPHICS_PATH)
    metadata = pd.read_csv(COGNOSPEAK_METADATA_PATH)

    for col, path in [(META_KEY_COLUMN, COGNOSPEAK_METADATA_PATH), (META_DATE_COLUMN, COGNOSPEAK_METADATA_PATH)]:
        if col not in metadata.columns:
            raise SystemExit(f"{col!r} not found in {path}. Available columns: {metadata.columns.tolist()}")

    date_map = build_date_map(metadata)
    demographics[NEW_COLUMN] = demographics[MATCH_COLUMN].map(date_map)

    matched = demographics[NEW_COLUMN].notna().sum()
    unmatched = len(demographics) - matched
    print(f"Matched {matched}/{len(demographics)} participants to an assessment date.")
    if unmatched and "system" in demographics.columns:
        print("Unmatched, by system (old-CognoSpeak participants are expected here):")
        print(demographics.loc[demographics[NEW_COLUMN].isna(), "system"].value_counts().to_string())

    if not BACKUP_PATH.exists():
        shutil.copy2(DEMOGRAPHICS_PATH, BACKUP_PATH)
        print(f"Backed up original to {BACKUP_PATH}")
    else:
        print(f"Backup already exists at {BACKUP_PATH}, not overwriting it")

    demographics.to_csv(DEMOGRAPHICS_PATH, index=False)
    print(f"Saved: {DEMOGRAPHICS_PATH}")


if __name__ == "__main__":
    main()
