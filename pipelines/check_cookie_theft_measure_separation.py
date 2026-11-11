"""Rank Cookie Theft's candidate display measures by how well they separate
HC/MCI/Dementia, to decide which three go at the top of the (currently crowded,
13-measure) tab -- see the dashboard-restructuring discussion in
docs/to_do/TODO_dashboard.md Tier 1 item 9 and the "Confounders to show" /
measure-gating decisions.

Same method pipelines/check_sft_error_rates.py already used to decide SFT/PFT's
headline measures: Cohen's d and whether the group means move monotonically
HC -> MCI -> Dementia (or the reverse). Reports d against both Dementia (the
easiest comparison) and MCI (the clinically harder, earlier-detection case) --
a measure that only separates HC from frank Dementia is less useful here than
one that already separates HC from MCI.

Covers every numeric column in cookie_theft_metrics.csv, not just the measures
interpretation.py currently lists as headline/exploratory -- the point of this
run is re-deciding the top three from scratch, so measures interpretation.py
hasn't opined on yet (e.g. mentioned_ius, total_mentions, the raw unnormalized
graph counts like cycles/total_path_distance/self_cycles/unique_nodes) are
included too. A few will be near-useless (total_ius is a constant 20 for
everyone -- flagged as "no variance", not silently dropped, so you can see it
was considered and ruled out, not just missing) and mentioned_ius/total_ius =
iu_score exactly, so their d and monotonic columns will be identical -- also
left in rather than pre-filtered, since that's exactly the kind of redundancy
you're trying to see directly rather than take my word for.

This only ranks measures -- it doesn't change interpretation.py or the
dashboard itself. Pure aggregate stats (cookie_theft_metrics.csv), no
transcript text, no participant-level output.

Run from the repo root: python pipelines/check_cookie_theft_measure_separation.py
"""
import numpy as np
import pandas as pd

from dementia_speech.paths import DATA_DIR

METRICS_PATH = DATA_DIR / "processed/cookie_theft_metrics.csv"
GROUPS = ["HC", "MCI", "Dementia"]

# Not a measure -- an identifier. Everything else numeric in the file is fair
# game (see module docstring for why that now includes more than just
# interpretation.py's current headline/exploratory lists).
EXCLUDE_COLUMNS = {"participant_id"}


def cohens_d(a: pd.Series, b: pd.Series) -> float | None:
    """d = (mean(b) - mean(a)) / pooled_sd. None for a constant column (e.g.
    total_ius, always 20) -- pooled_sd is 0, so the ratio is undefined rather
    than a spurious inf/nan that would sort to the top of the ranking."""
    n_a, n_b = len(a), len(b)
    pooled_sd = np.sqrt(
        ((n_a - 1) * a.std() ** 2 + (n_b - 1) * b.std() ** 2) / (n_a + n_b - 2)
    )
    if not pooled_sd:
        return None
    return (b.mean() - a.mean()) / pooled_sd


def is_monotonic(group_means: pd.Series) -> bool:
    values = group_means.loc[GROUPS].values
    return bool((values[0] <= values[1] <= values[2]) or (values[0] >= values[1] >= values[2]))


def main():
    df = pd.read_csv(METRICS_PATH)
    df = df[df["diagnosis"].isin(GROUPS)].copy()

    measures = [
        c for c in df.select_dtypes(include="number").columns
        if c not in EXCLUDE_COLUMNS
    ]

    print(f"n per group: {df.groupby('diagnosis').size().to_dict()}")
    print(f"{len(measures)} numeric columns considered: {measures}\n")

    rows = []
    for measure in measures:
        sub = df.dropna(subset=[measure])
        if sub.empty or sub.groupby("diagnosis").size().min() < 2:
            rows.append({"measure": measure, "note": "too few non-missing values per group"})
            continue

        group_means = sub.groupby("diagnosis")[measure].mean()
        monotonic = is_monotonic(group_means)
        d_dementia = cohens_d(sub.loc[sub["diagnosis"] == "HC", measure],
                               sub.loc[sub["diagnosis"] == "Dementia", measure])
        d_mci = cohens_d(sub.loc[sub["diagnosis"] == "HC", measure],
                          sub.loc[sub["diagnosis"] == "MCI", measure])
        # MCI vs Dementia: not HC-anchored like the two above -- this is about
        # progression/staging (does the measure keep moving once someone's
        # already past HC-normal), not detection against a healthy baseline.
        d_mci_dementia = cohens_d(sub.loc[sub["diagnosis"] == "MCI", measure],
                                   sub.loc[sub["diagnosis"] == "Dementia", measure])

        rows.append({
            "measure": measure,
            "HC_mean": round(group_means["HC"], 3),
            "MCI_mean": round(group_means["MCI"], 3),
            "Dementia_mean": round(group_means["Dementia"], 3),
            "monotonic": monotonic,
            "d (HC vs MCI)": round(d_mci, 2) if d_mci is not None else "no variance",
            "d (HC vs Dementia)": round(d_dementia, 2) if d_dementia is not None else "no variance",
            "d (MCI vs Dementia)": round(d_mci_dementia, 2) if d_mci_dementia is not None else "no variance",
        })

    result = pd.DataFrame(rows)
    if "d (HC vs Dementia)" in result.columns:
        result["_rank_key"] = pd.to_numeric(result["d (HC vs Dementia)"], errors="coerce").abs().fillna(-1)
        result = result.sort_values("_rank_key", ascending=False).drop(columns="_rank_key")

    print(result.to_string(index=False))
    print(
        "\nSorted by |d (HC vs Dementia)|. Same bar the project already applies "
        "elsewhere: a measure not monotonic across HC->MCI->Dementia is suspect "
        "even with a large d (see PFT_new/SFT_new interpretation.py docstrings) "
        "-- check that column before picking the top three, not just the |d| rank."
    )


if __name__ == "__main__":
    main()
