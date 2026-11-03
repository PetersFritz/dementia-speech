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
after controlling for `C` -- while `OTH_rate`'s raw effect (d=0.38) turned
out to be fully explained by `C` (residual d=-0.04, not even monotonic).
`F_rate` shows no effect either way (not monotonic, d=0.11 raw).

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

Semantic clustering/switching isn't implemented yet (deferred -- see the
Tier 1 dashboard TODO), so unlike PFT_new there's no exploratory section
for it here.
"""

from dementia_speech.interpretation import HeadlineMeasure

HEADLINE_MEASURES: list[HeadlineMeasure] = [
    HeadlineMeasure(
        key="C",
        label="Animals produced correctly",
        description="Distinct real animal names produced in one minute -- the standard category-fluency outcome measure.",
        higher_is_worse=False,
        fmt="{:.0f}",
    ),
]

# Shown alongside the headline card but not banded -- see module docstring.
CONTEXT_MEASURES: list[HeadlineMeasure] = [
    HeadlineMeasure(
        key="total_items",
        label="Total items spoken",
        description="All scored items, including repetitions, intrusions, and filled pauses. Helps interpret the score above: low output overall, or plenty of output that mostly didn't land on new animals?",
        higher_is_worse=False,
        fmt="{:.0f}",
    ),
]

# The one error/code rate that carries signal independent of `C` -- see
# module docstring. Banded and shown in the dedicated error section.
ERROR_HEADLINE_MEASURES: list[HeadlineMeasure] = [
    HeadlineMeasure(
        key="R_rate",
        label="Repetition rate",
        description="Share of items that repeat an animal already named, per 100 items. Still distinguishes groups after controlling for correct-item count -- a perseveration-like pattern, not just a restatement of reduced output.",
        higher_is_worse=True,
        fmt="{:.1f}",
    ),
]

# Checked the same way as the headline measure, but failed one or both
# criteria -- see module docstring and docs/sft_scoring.md.
# OTH_rate: real raw effect, but fully explained by C (residual d=-0.04,
#   not even monotonic) -- unlike PFT, intrusions here don't carry
#   independent signal once output level is accounted for.
# F_rate: no effect either way (not monotonic, d=0.11 raw).
ERROR_EXPLORATORY_MEASURES = {
    "OTH_rate": "Intrusion rate",
    "F_rate": "Filled pause rate",
}

# Mean age-of-acquisition (Kuperman et al. 2012) of correctly-named animals.
# Not run through the same monotonic/residualize-against-C check the error
# rates above got -- added on request, not yet validated for SFT. PFT's own
# AoA measure looked informative alone but turned out to be fully explained
# by `C` there; shown here as exploratory by default for the same reason,
# pending its own check. A list of HeadlineMeasure (not a plain dict) so it
# can go through the same card/z-score rendering as the other measures,
# even though it's always shown with banded=False (no clinical direction
# has been established, so `higher_is_worse` here is a placeholder, unused
# since band() is never called for it).
EXPLORATORY_MEASURES: list[HeadlineMeasure] = [
    HeadlineMeasure(
        key="mean_aoa_correct",
        label="Mean age-of-acquisition of correct animals (yrs)",
        description="Average age (in years) at which the correctly-named animals are typically learned as a word -- not yet checked for independent signal in this task.",
        higher_is_worse=False,
        fmt="{:.2f}",
    ),
]
