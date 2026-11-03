"""Compute the letter-vs-category fluency discrepancy score.

Method from Wright, L. M., De Marco, M., & Venneri, A. (2022). Verbal
fluency discrepancies as a marker of the prehippocampal stages of
Alzheimer's disease. Neuropsychology, 37(7), 790-800.

discrepancy = z_letter - z_category

Positive = category (semantic) fluency fell further below the reference
mean than letter fluency did -- the paper's own sign convention, kept
as-is for comparability with the literature.

Deliberately NOT using the paper's own published norms: their raw scores
are summed across three 1-minute trials per task (letter: F+P+L; category:
Animals+Fruits+Cities/Cars), while this project has only one trial per
task (letter: P only; category: Animals only). Plugging a single-trial
score into norms calibrated on a three-trial sum produces meaningless
z-scores (checked: our own HC group averages ~17 correct on the single P
trial, against the paper's three-letter-sum HC mean of 39.81). Instead,
z-scores are calibrated against this project's own HC group -- the
"equivalent normative dataset" the paper's own practical notes call for --
restricted to the TRAIN split only, so no participant (HC included) is
ever scored against a reference distribution that includes them. The
existing PFT/Cookie Theft dashboard bands don't yet make this TRAIN/TEST
distinction; fixing those is a separate, later task.

Experimental measure: single trial per task (vs. the paper's three) and a
different, smaller reference population -- flag accordingly in the UI
when this is wired into the dashboard.

Reads only already-computed per-participant counts (pft_metrics_new.csv,
sft_metrics.csv) and participant metadata (meta-info.csv's Split column)
-- no transcript text.
"""
import os
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
os.chdir(project_root)

from dementia_speech.paths import DATA_DIR

PFT_METRICS_PATH = DATA_DIR / "processed/pft_metrics_new.csv"
SFT_METRICS_PATH = DATA_DIR / "processed/sft_metrics.csv"
METADATA_PATH = DATA_DIR / "raw/meta-info.csv"
OUTPUT_PATH = DATA_DIR / "processed/fluency_discrepancy.csv"


def _lookup_split(participant_id: str, metadata: pd.DataFrame) -> str | None:
    key = f"{int(participant_id):03d}"
    match = metadata[metadata["IDs"].astype(str).str.contains(key)]
    return match.iloc[0]["Split"] if len(match) else None


def main():
    pft = pd.read_csv(PFT_METRICS_PATH, dtype={"participant_id": str})
    sft = pd.read_csv(SFT_METRICS_PATH, dtype={"participant_id": str})
    metadata = pd.read_csv(METADATA_PATH)

    df = pd.merge(
        pft[["participant_id", "diagnosis", "C"]].rename(columns={"C": "letter_C"}),
        sft[["participant_id", "C"]].rename(columns={"C": "category_C"}),
        on="participant_id", how="outer",
    )
    df["split"] = df["participant_id"].apply(lambda pid: _lookup_split(pid, metadata))

    hc_train = df[(df["diagnosis"] == "HC") & (df["split"] == "TRAIN")]
    letter_mean, letter_sd = hc_train["letter_C"].mean(), hc_train["letter_C"].std()
    category_mean, category_sd = hc_train["category_C"].mean(), hc_train["category_C"].std()

    df["z_letter"] = (df["letter_C"] - letter_mean) / letter_sd
    df["z_category"] = (df["category_C"] - category_mean) / category_sd
    df["discrepancy_score"] = df["z_letter"] - df["z_category"]

    df = df[[
        "participant_id", "diagnosis", "split",
        "letter_C", "category_C", "z_letter", "z_category", "discrepancy_score",
    ]]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(
        f"HC (TRAIN, n={len(hc_train)}) reference -- "
        f"letter: mean={letter_mean:.2f} sd={letter_sd:.2f}; "
        f"category: mean={category_mean:.2f} sd={category_sd:.2f}"
    )
    print(f"\nSaved: {OUTPUT_PATH} ({len(df)} participants)")
    print("\nDiscrepancy score by diagnosis:")
    print(df.groupby("diagnosis")["discrepancy_score"].agg(["mean", "std", "count"]).round(2))


if __name__ == "__main__":
    main()
