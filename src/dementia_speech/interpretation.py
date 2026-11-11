"""Generic clinician-facing interpretation helpers, shared across tasks.

Converts a raw measure into a plain-language Typical/Borderline/Atypical
band relative to a reference (healthy-control) group, following standard
neuropsychological test-interpretation convention (within 1 SD of the
reference mean = typical; 1-2 SD = borderline; beyond 2 SD = atypical --
cf. Lezak, Howieson & Loring, *Neuropsychological Assessment*), applied
one-directionally: scoring unusually well is still "Typical", since the
point is to flag concern, not exceptional performance.

(Cookie Theft briefly tried per-measure ROC-calibrated cutoffs here instead
of the universal ±1/±2 SD convention -- removed 2026-11-14 in favour of
showing a continuous age-adjusted percentile with no categorical judgement
at all, see percentile_age_adjusted() below and
picture_description/interpretation.py's module docstring for why: any fixed
cutoff, calibrated or not, bakes in a sensitivity/specificity tradeoff the
dashboard shouldn't be making unilaterally, and this cohort's modest
individual-level discrimination -- ~15-55% sensitivity depending how loose
the cutoff -- meant most Dementia participants still looked "Typical"
regardless of which specificity floor was chosen.)

Used by both `tasks/picture_description/interpretation.py` and
`tasks/PFT_new/interpretation.py`, which each define their own task-specific
measure lists (which raw columns count as "headline" vs "exploratory", and
why) -- see those modules and docs/cookie_theft_scoring.md /
docs/pft_new_scoring.md for the analysis behind each task's choices. A
measure only belongs on a "headline" list if it (a) moves monotonically
HC -> MCI -> Dementia (or the reverse) in that task's own data and (b)
still does so, or at least still separates HC from Dementia, after
accounting for the more obvious explanatory variables in that task (verbal
output length, correct-word count, etc.) -- several measures in both tasks
looked informative on their own but turned out to be fully explained by a
more basic measure already on the headline list, which would double-count
one underlying effect as if it were several independent findings.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HeadlineMeasure:
    key: str
    label: str
    description: str
    higher_is_worse: bool
    fmt: str = "{:.2f}"


BAND_COLOURS = {
    "Typical": "#2e7d32",
    "Borderline": "#b8860b",
    "Atypical": "#c0392b",
    "Not available": "#9e9e9e",
}


def band(value, ref_mean, ref_sd, higher_is_worse) -> tuple:
    """Return (band_label, z_oriented) for one value against a reference mean/SD.

    z_oriented is signed so that negative always means "in the impaired
    direction", regardless of whether high or low is the impaired direction
    for this particular raw measure.
    """
    if value is None or ref_mean is None or not ref_sd:
        return "Not available", None
    z = (value - ref_mean) / ref_sd
    z_oriented = -z if higher_is_worse else z
    if z_oriented >= -1:
        return "Typical", z_oriented
    if z_oriented >= -2:
        return "Borderline", z_oriented
    return "Atypical", z_oriented


# ── Age-adjusted norming ──────────────────────────────────────────────────
#
# band() above compares a value to one flat HC mean/SD, which treats a
# 45-year-old and an 85-year-old HC as draws from the same distribution.
# Where a measure actually varies with age in the reference group, that
# flattens out real age effects into "noise", making the reference range
# too wide for younger participants and too narrow for older ones. The fix
# is standard neuropsychological regression-based norming: fit
# measure ~ age on the HC (TRAIN-split) group, then judge each participant
# against the value *predicted for their own age*, with the regression's
# residual SD standing in for the old flat SD.

MIN_HC_FOR_AGE_NORM = 10


def fit_age_adjusted_norms(hc_values, hc_ages) -> tuple:
    """Fit `measure ~ age` on one measure's HC reference values.

    Parameters
    ----------
    hc_values, hc_ages : array-likes (e.g. pandas Series), same length,
        aligned by position -- one row per HC (TRAIN-split) participant.

    Returns
    -------
    (intercept, slope, residual_sd) for use with band_age_adjusted().
    Falls back to slope=0 (i.e. a flat mean, same as the original band())
    when there are fewer than MIN_HC_FOR_AGE_NORM HC participants with both
    a value and an age, or when HC ages don't vary at all -- too little
    data to fit a reliable age trend, so age-adjustment would just add
    noise rather than remove it.
    """
    import numpy as np
    import pandas as pd

    values = pd.Series(hc_values).reset_index(drop=True)
    ages = pd.Series(hc_ages).reset_index(drop=True)
    mask = values.notna() & ages.notna()
    values = values[mask].astype(float).to_numpy()
    ages = ages[mask].astype(float).to_numpy()
    n = len(values)

    if n < 2:
        return None, None, None
    if n < MIN_HC_FOR_AGE_NORM or np.std(ages) == 0:
        sd = float(values.std(ddof=1)) if n > 1 else None
        return float(values.mean()), 0.0, sd

    slope, intercept = np.polyfit(ages, values, 1)
    residuals = values - (intercept + slope * ages)
    # ddof=2: one degree of freedom spent on each of intercept and slope.
    residual_sd = float(residuals.std(ddof=2)) if n > 2 else None
    return float(intercept), float(slope), residual_sd


def band_age_adjusted(value, age, intercept, slope, residual_sd, higher_is_worse) -> tuple:
    """Like band(), but the reference mean is the HC-fitted line's
    prediction for this participant's own age (see fit_age_adjusted_norms),
    not one flat HC mean shared by every age.
    """
    if value is None or age is None or intercept is None or slope is None:
        return "Not available", None
    predicted_mean = intercept + slope * age
    return band(value, predicted_mean, residual_sd, higher_is_worse)


def percentile_age_adjusted(value, age, intercept, slope, residual_sd, higher_is_worse) -> float | None:
    """Age-adjusted percentile (0-100) for one value, relative to the
    HC-fitted line's prediction for this participant's own age (see
    fit_age_adjusted_norms) -- the continuous counterpart to
    band_age_adjusted(): no Typical/Borderline/Atypical judgement, just
    where this value falls in the (assumed approximately normal) reference
    distribution, for the clinician to weigh themselves.

    Oriented like band()'s z_oriented: a LOW percentile always means the
    impaired direction, regardless of whether high or low is the impaired
    raw direction for this particular measure -- so percentile is directly
    comparable across measures the same way z_oriented is, without needing
    to remember which direction is "bad" for each one.
    """
    if value is None or age is None or intercept is None or slope is None or not residual_sd:
        return None
    from scipy.stats import norm

    predicted_mean = intercept + slope * age
    z = (value - predicted_mean) / residual_sd
    z_oriented = -z if higher_is_worse else z
    return float(100 * norm.cdf(z_oriented))
