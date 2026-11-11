"""Clinician-facing interpretation of semantic/category fluency (SFT) measures.

See `dementia_speech.interpretation` for the generic Typical/Borderline/
Atypical banding convention, and `PFT_new/interpretation.py` for the same
method applied to phonemic fluency -- this module runs the identical check
(`pipelines/check_sft_error_rates.py`): does each error/code rate still
separate HC from Dementia after residualizing against `C` (correct count)?

Result is the mirror image of PFT: there, `OTH_rate` (off-target speech)
survived and `R_rate` didn't. Here, `R_rate` (repetition) survives --
monotonic across HC -> MCI -> Dementia (2.95 -> 4.67 -> 6.02), large raw
effect (d=0.70), and still monotonic with a solid residual effect (d=0.48)
after controlling for `C` -- while `OTH_rate`'s raw effect (d=0.34) turned
out to be fully explained by `C` (residual d=-0.07, not even monotonic).
`F_rate` shows no effect either way (not monotonic, d=0.11 raw).

`N_rate` (nonword) was added 2026-11-17, splitting what used to be one
catch-all `OTH` code into `N` (fragment/false start, not a real English
word -- "alpac", "de-de", "ka") and a narrower `OTH` (genuine real-word
intrusion or narration aside -- "car", "I can't think"), the same
distinction PFT_new draws (see SFT_new/scoring.py's module docstring for
the classification detail, including a contraction-handling fix needed
along the way: `is_real_word()`'s dictionaries don't cover contractions/
possessives like "can't" or "cetti's" at all, which would have
misclassified ~45% of the new `N` code before the fix). Checked the same
way: raw d=0.29, monotonic, but residual d=0.14 and non-monotonic -- doesn't
clear the bar, same verdict as PFT's own `N_rate` (moderate raw effect,
floor-effect-prone, exploratory only).

This isn't an inconsistency with PFT -- it's a genuine task difference.
Increased repetition (perseveration) with greater severity is a
well-documented marker of executive/retrieval dysfunction in verbal
fluency tasks, so `R_rate` earning headline treatment here, rather than
`OTH_rate`, is a sensible finding rather than a surprising one. See
docs/sft_scoring.md for the full write-up.

As with PFT, `total_items` (all items produced, correct or not) is kept as
an unbanded context measure, not checked for group separation, for the
same reason PFT's `total_words` isn't banded: alongside `C`, it helps
distinguish low output overall from plenty of output that mostly missed
the mark.

Semantic clustering/switching (Troyer et al. 1997, category scheme from
Zemla et al. 2020 SNAFU -- see SFT_new/clustering.py) is now implemented
and checked the same way as everything else here: `n_clusters` has a large
raw effect (d=-1.01, monotonic) but collapses once residualized against `C`
(d=-0.16, correlation with C=0.80 -- mostly restating output, exactly
PFT_new's own phonemic-clustering finding). `n_switches` and
`largest_cluster_size` fail outright (non-monotonic after residualizing,
sign reversed). `mean_cluster_size` keeps a correctly-directioned but very
small residual effect (d=-0.11, comparable to F_rate's "no real effect"
d=0.11 above). None clear this project's bar for an independent signal --
kept computed and saved in sft_metrics.csv (available for research use and
transcript display) but not shown as a dashboard measure, same treatment
PFT_new gives its own clustering/switching. See docs/sft_scoring.md.

`mean_aoa_correct` has now been checked the same way too (it previously
wasn't): raw d=0.30 (correctly directioned), but residual d=-0.03 (reversed)
after controlling for `C` -- same collapse as PFT's own AoA measure. Kept
visible anyway, per user request, as a research-reference measure rather
than an independent clinical flag (same framing PFT_new uses for its own
AoA card).
"""

from dementia_speech.interpretation import HeadlineMeasure

HEADLINE_MEASURES: list[HeadlineMeasure] = [
    HeadlineMeasure(
        key="C",
        label="Animals produced correctly",
        description="Distinct real animal names produced in one minute.",
        higher_is_worse=False,
        fmt="{:.0f}",
    ),
]

# Shown alongside the headline card as this task's two "key measures" (user
# request, 2026-11-17: just `C` + `total_items` on top, everything else --
# including `R_rate`, despite being validated -- in "Additional measures").
CONTEXT_MEASURES: list[HeadlineMeasure] = [
    HeadlineMeasure(
        key="total_items",
        label="Total items spoken",
        description="All scored items, including repetitions, intrusions, and filled pauses.",
        higher_is_worse=False,
        fmt="{:.0f}",
    ),
]

# The one error/code rate that carries signal independent of `C` -- see
# module docstring. Genuinely validated, but shown in "Additional measures"
# rather than alongside the two key measures above, per user request
# (2026-11-17): the key-measures row is just `C` + `total_items`.
ERROR_HEADLINE_MEASURES: list[HeadlineMeasure] = [
    HeadlineMeasure(
        key="R_rate",
        label="Repetition rate",
        description="Share of items that repeat an animal already named, per 100 items.",
        higher_is_worse=True,
        fmt="{:.1f}",
    ),
]

# Checked the same way as the headline measure, but failed one or both
# criteria -- see module docstring and docs/sft_scoring.md. A HeadlineMeasure
# list (not a plain dict, as of 2026-11-17) so these get the same card +
# percentile + embedded-graph treatment as every other measure, shown in
# one "Additional measures" section -- matches Cookie Theft's layout.
ERROR_EXPLORATORY_MEASURES: list[HeadlineMeasure] = [
    HeadlineMeasure(
        key="OTH_rate",
        label="Intrusion rate",
        description=(
            "Share of items that aren't a known animal, but are a real "
            "word or narration aside, per 100 items. Raw effect fully "
            "explained by correct-item count once controlled for -- "
            "unlike PFT, intrusions here don't carry independent signal."
        ),
        higher_is_worse=True,
        fmt="{:.1f}",
    ),
    HeadlineMeasure(
        key="N_rate",
        label="Nonword rate",
        description=(
            "Share of items that aren't a known animal and aren't a real "
            "word either (fragment/false start), per 100 items."
        ),
        higher_is_worse=True,
        fmt="{:.1f}",
    ),
    HeadlineMeasure(
        key="F_rate",
        label="Filled pause rate",
        description="Share of items that are filled pauses (um, er, ...), per 100 items.",
        higher_is_worse=True,
        fmt="{:.1f}",
    ),
]

# Mean age-of-acquisition (Kuperman et al. 2012) of correctly-named animals.
# Now checked the same way as everything else (see module docstring):
# collapses once residualized against C, same as PFT's own AoA measure.
# Kept visible anyway, per user request, as a research-reference measure.
EXPLORATORY_MEASURES: list[HeadlineMeasure] = [
    HeadlineMeasure(
        key="error_rate",
        label="Error rate",
        description=(
            "Share of items that were neither a correct new animal nor a "
            "filled pause (repetitions, intrusions, and nonwords "
            "combined), per 100 items. Not yet checked for signal "
            "independent of correct-item count."
        ),
        higher_is_worse=True,
        fmt="{:.1f}",
    ),
    HeadlineMeasure(
        key="mean_aoa_correct",
        label="Mean age-of-acquisition of correct animals (yrs)",
        description="Average age (in years) at which the correctly-named animals are typically learned as a word.",
        higher_is_worse=False,
        fmt="{:.2f}",
    ),
]
