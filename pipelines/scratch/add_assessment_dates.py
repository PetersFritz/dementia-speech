"""Add assessment (recording) dates to Process_demographics.csv for new-CognoSpeak participants.

Joins `CognoSpeak_metadata__2026_06_07.csv` (column `dir_name`, matched against
`Process_demographics.csv`'s `best_match_id`) onto Process_demographics.csv, pulling in that
file's `asses_date` column as a new `assessment_date` column (ISO `YYYY-MM-DD`). Only rows whose
`best_match_id` has a match in the metadata file get a date this way -- old-CognoSpeak
participants have no `dir_name` to match against, and are expected to need a date entered by hand
directly in Process_demographics.csv instead.

Additive: an `assessment_date` that's already in Process_demographics.csv (e.g. a hand-entered one
for an old-CognoSpeak participant) is kept, not overwritten, by a re-run of this script -- only
blank rows get filled from the metadata join. A row where the two actually disagree is reported,
not silently resolved either way.

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
    mapped = demographics[MATCH_COLUMN].map(date_map)

    # Additive, not a blind overwrite: an existing assessment_date (e.g. one
    # entered by hand for an old-CognoSpeak participant with no dir_name
    # match) is kept rather than replaced, so re-running this script can't
    # silently wipe a manual entry -- confirmed happening in practice,
    # 2026-11-15. A row whose freshly-matched date actually disagrees with
    # what's already there is flagged instead of silently resolved, since
    # that's a real discrepancy worth a human look rather than a rerun
    # loading dates for the first time.
    if NEW_COLUMN in demographics.columns:
        existing = demographics[NEW_COLUMN]
        conflicts = existing.notna() & mapped.notna() & (existing.astype(str) != mapped.astype(str))
        if conflicts.any():
            print(f"  WARNING: {conflicts.sum()} rows already have an assessment_date that "
                  f"disagrees with the freshly-matched one -- keeping the EXISTING value for these "
                  f"{MATCH_COLUMN} values (check by hand if this is unexpected):")
            for match_id in demographics.loc[conflicts, MATCH_COLUMN].unique():
                print(f"    {match_id!r}")
        demographics[NEW_COLUMN] = existing.where(existing.notna(), mapped)
    else:
        demographics[NEW_COLUMN] = mapped

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
