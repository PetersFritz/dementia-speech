"""Check whether SFT's error/code rates (OTH_rate, R_rate, F_rate) carry
signal independent of C (correct count), the same way PFT_new's rates were
checked -- see docs/pft_new_scoring.md "Error analysis" and
PFT_new/interpretation.py's docstring for the method and rationale this
mirrors.

For each rate: group means (HC/MCI/Dementia), whether it's monotonic across
groups, Cohen's d (HC vs. Dementia), and whether that effect survives after
residualizing the rate against C (regressing rate ~ C, then re-checking the
effect on the residuals) -- since a participant with fewer correct items
necessarily has more "room" in their output for other codes, so the
question is whether that's genuine independent signal or just arithmetic.

Only decides which SFT measures are "headline" (banded) vs. "exploratory"
in SFT_new/interpretation.py -- doesn't touch the dashboard. Pure aggregate
counts (sft_metrics.csv), no transcript text.

Run from the repo root: python pipelines/check_sft_error_rates.py
"""
import numpy as np
import pandas as pd

from dementia_speech.paths import DATA_DIR

METRICS_PATH = DATA_DIR / "processed/sft_metrics.csv"

GROUPS = ["HC", "MCI", "Dementia"]
RATES = ["OTH_rate", "R_rate", "F_rate"]


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """d = (mean(b) - mean(a)) / pooled_sd."""
    n_a, n_b = len(a), len(b)
    pooled_sd = np.sqrt(
        ((n_a - 1) * a.std() ** 2 + (n_b - 1) * b.std() ** 2) / (n_a + n_b - 2)
    )
    return (b.mean() - a.mean()) / pooled_sd


def is_monotonic(group_means: pd.Series) -> bool:
    values = group_means.loc[GROUPS].values
    return bool((values[0] <= values[1] <= values[2]) or (values[0] >= values[1] >= values[2]))


def residualize(df: pd.DataFrame, rate_col: str) -> pd.Series:
    """Residuals of rate_col after a simple linear regression on C."""
    x = df["C"].values
    y = df[rate_col].values
    slope, intercept = np.polyfit(x, y, 1)
    return pd.Series(y - (slope * x + intercept), index=df.index)


def main():
    df = pd.read_csv(METRICS_PATH)
    df = df[df["diagnosis"].isin(GROUPS)].copy()

    print(f"n per group: {df.groupby('diagnosis').size().to_dict()}\n")

    rows = []
    for rate_col in RATES:
        sub = df.dropna(subset=[rate_col, "C"])
        group_means = sub.groupby("diagnosis")[rate_col].mean()
        d_raw = cohens_d(sub.loc[sub["diagnosis"] == "HC", rate_col],
                          sub.loc[sub["diagnosis"] == "Dementia", rate_col])
        monotonic_raw = is_monotonic(group_means)

        sub["_residual"] = residualize(sub, rate_col)
        d_resid = cohens_d(sub.loc[sub["diagnosis"] == "HC", "_residual"],
                            sub.loc[sub["diagnosis"] == "Dementia", "_residual"])
        resid_means = sub.groupby("diagnosis")["_residual"].mean()
        monotonic_resid = is_monotonic(resid_means)

        rows.append({
            "rate": rate_col,
            "HC_mean": round(group_means["HC"], 2),
            "MCI_mean": round(group_means["MCI"], 2),
            "Dementia_mean": round(group_means["Dementia"], 2),
            "monotonic_raw": monotonic_raw,
            "d_raw (HC vs Dementia)": round(d_raw, 2),
            "monotonic_after_residualizing_C": monotonic_resid,
            "d_after_residualizing_C": round(d_resid, 2),
        })

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    print(
        "\nA rate earns 'headline' (banded) treatment only if it's monotonic "
        "and the effect survives residualizing against C -- same bar PFT_new "
        "applied to OTH_rate. Anything that fails either check is exploratory "
        "only (same reasoning as PFT_new's R_rate/F_rate -- see docstring)."
    )


if __name__ == "__main__":
    main()
