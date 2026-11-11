"""Clinician-facing interpretation of phonemic fluency (PFT) measures.

See `dementia_speech.interpretation` for the generic Typical/Borderline/
Atypical banding convention and for the general rule this project uses to
decide whether a measure belongs on a "headline" list.

Checking every PFT measure for (a) a monotonic HC -> MCI -> Dementia trend
and (b) whether it still separates groups after residualizing against the
number of correctly-produced words (`C`) turned up a striking result: `C`
is by far the dominant, independent signal in this task (large effect,
monotonic, matches decades of letter-fluency literature, e.g. Tombaugh et
al. 1999 COWAT/FAS norms) — and almost everything else that looked
informative on its own (clustering, switching, mean age-of-acquisition)
turned out to be fully explained by `C`: once you control for how many
words someone got right, those measures no longer differ between groups.
Presenting them as separate headline flags would double-count one
underlying effect (reduced word retrieval) as if it were several
independent findings, so they're kept as descriptive/exploratory instead.

The one exception is `OTH_rate` (off-target speech — words that don't even
attempt the target letter): it still separates groups after controlling
for `C` (residual effect d=-0.30, monotonic), meaning it's telling you
something `C` alone doesn't — not just "produced fewer correct words" but
"filled that gap with off-topic speech rather than genuine attempts,
non-words, or filled pauses." See docs/pft_new_scoring.md, "Error
analysis", for the full measure-by-measure check.

`total_words` (all words spoken, correct or not) does not differ reliably
between groups in this task — unlike Cookie Theft, participants tend to
keep talking for the full trial regardless of ability — so it isn't banded
here. It's kept as an unbanded context measure: alongside `C`, it tells you
whether a low correct-word count reflects low overall output/initiation or
plenty of output that mostly missed the target.
"""

from dementia_speech.interpretation import HeadlineMeasure

HEADLINE_MEASURES: list[HeadlineMeasure] = [
    HeadlineMeasure(
        key="C",
        label="Words produced correctly",
        description="Real target-letter words produced in one minute.",
        higher_is_worse=False,
        fmt="{:.0f}",
    ),
]

# Shown alongside the headline card as this task's two "key measures" (user
# request, 2026-11-17: just `C` + `total_words` on top, everything else --
# including `OTH_rate`, despite being validated -- in "Additional measures").
CONTEXT_MEASURES: list[HeadlineMeasure] = [
    HeadlineMeasure(
        key="total_words",
        label="Total words spoken",
        description="All words spoken, including errors and off-target speech.",
        higher_is_worse=False,
        fmt="{:.0f}",
    ),
]

# The one error/off-target measure that carries signal independent of `C`
# (see module docstring) -- genuinely validated, but shown in "Additional
# measures" rather than alongside the two key measures above, per user
# request (2026-11-17): the key-measures row is just `C` + `total_words`.
ERROR_HEADLINE_MEASURES: list[HeadlineMeasure] = [
    HeadlineMeasure(
        key="OTH_rate",
        label="Off-target speech rate",
        description="Share of spoken words that don't even attempt the target letter, per 100 words.",
        higher_is_worse=True,
        fmt="{:.1f}",
    ),
]

# Remaining error/code rates: real but weaker or confounded (see
# docs/pft_new_scoring.md "Error analysis") — a HeadlineMeasure list (not a
# plain dict, as of 2026-11-17) so these get the same card + percentile +
# embedded-graph treatment as every other measure, shown together with
# CONTEXT_MEASURES and EXPLORATORY_MEASURES in one "Additional measures"
# section rather than a separate stacked plot -- matches Cookie Theft's
# layout (one flat "Additional measures" accordion for everything that
# isn't a validated key measure).
ERROR_EXPLORATORY_MEASURES: list[HeadlineMeasure] = [
    HeadlineMeasure(
        key="N_rate",
        label="Nonword rate",
        description=(
            "Share of spoken words that attempt the target letter but "
            "aren't real words, per 100 words."
        ),
        higher_is_worse=True,
        fmt="{:.1f}",
    ),
    HeadlineMeasure(
        key="PN_rate",
        label="Proper noun rate",
        description=(
            "Share of spoken words that are person or place names, per "
            "100 words. Most participants use none at all (floor effect)."
        ),
        higher_is_worse=True,
        fmt="{:.1f}",
    ),
    HeadlineMeasure(
        key="R_rate",
        label="Repetition rate",
        description=(
            "Share of spoken words that repeat an earlier correct word, "
            "per 100 words. Opportunity to repeat scales with how many "
            "correct words came before it, so this is structurally lower "
            "for anyone with fewer correct words to begin with."
        ),
        higher_is_worse=True,
        fmt="{:.1f}",
    ),
    HeadlineMeasure(
        key="SIM_rate",
        label="Similar-word rate",
        description=(
            "Share of spoken words that are a morphological variant or "
            "synonym of an earlier correct word, per 100 words. Same "
            "structural confound as repetition rate."
        ),
        higher_is_worse=True,
        fmt="{:.1f}",
    ),
    HeadlineMeasure(
        key="F_rate",
        label="Filled pause rate",
        description="Share of spoken words that are filled pauses (um, er, ...), per 100 words.",
        higher_is_worse=True,
        fmt="{:.1f}",
    ),
]

# AoA: a real, monotonic effect on its own, but fully explained by `C` (see
# module docstring) -- kept for research use, not as an independent
# clinical flag, per user request (2026-11-17). Clustering/switching are
# computed and saved in pft_metrics_new.csv as before, but left out of this
# list -- and so off the dashboard's exploratory panel -- for the same
# reason; see the Tier 1 dashboard TODO for revisiting a semantic-
# clustering-consistent presentation later.
#
# `time_to_first_correct` (seconds to the first real target-letter word --
# see PFT_new/features.py::compute_latency_features) is a genuinely
# different case, checked the same way (docs/pft_new_scoring.md, "Response
# latency"): large, correctly-directioned raw effect (d=0.65, HC 2.0s vs.
# Dementia 5.3s) that only partly collapses after residualizing against `C`
# (residual d=0.28, still correctly directioned -- stronger than OTH_rate's
# qualifying residual effect). It fails this project's full "headline" bar
# because the 3-group ordering breaks at MCI (MCI's residual latency isn't
# detectably different from HC's; only Dementia's is elevated) -- a
# severity/staging pattern, not a screening one, the same shape as Cookie
# Theft's `avg_x`. Kept here as exploratory, not promoted to headline,
# pending a decision on whether staging-only measures should get their own
# treatment.
EXPLORATORY_MEASURES: list[HeadlineMeasure] = [
    HeadlineMeasure(
        key="error_rate",
        label="Error rate",
        description=(
            "Share of spoken words that were neither correct nor a filled "
            "pause (nonwords, proper nouns, repetitions, similar-form "
            "errors, and off-target speech combined), per 100 words. Not "
            "yet checked for signal independent of correct-word count."
        ),
        higher_is_worse=True,
        fmt="{:.1f}",
    ),
    HeadlineMeasure(
        key="time_to_first_correct",
        label="Time to first correct word",
        description=(
            "Seconds from trial start to the first real target-letter word "
            "produced. Elevated in Dementia but not detectably in MCI once "
            "output is accounted for -- a severity signal, not yet a "
            "screening one."
        ),
        higher_is_worse=True,
        fmt="{:.1f}s",
    ),
    HeadlineMeasure(
        key="mean_aoa_correct",
        label="Mean age-of-acquisition of correct words (yrs)",
        description="Average age (in years) at which the correctly-produced words are typically learned.",
        higher_is_worse=False,
        fmt="{:.2f}",
    ),
]
