"""Generic clinician-facing interpretation helpers, shared across tasks.

Converts a raw measure into a plain-language Typical/Borderline/Atypical
band relative to a reference (healthy-control) group, following standard
neuropsychological test-interpretation convention (within 1 SD of the
reference mean = typical; 1-2 SD = borderline; beyond 2 SD = atypical --
cf. Lezak, Howieson & Loring, *Neuropsychological Assessment*), applied
one-directionally: scoring unusually well is still "Typical", since the
point is to flag concern, not exceptional performance.

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
