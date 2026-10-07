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
        description="Real target-letter words produced in one minute — the standard letter-fluency outcome measure, and the dominant marker in this dataset.",
        higher_is_worse=False,
        fmt="{:.0f}",
    ),
]

# Shown alongside the headline card but not banded — see module docstring.
CONTEXT_MEASURES: list[HeadlineMeasure] = [
    HeadlineMeasure(
        key="total_words",
        label="Total words spoken",
        description="All words spoken, including errors and off-target speech. Doesn't reliably differ by group in this task, but helps interpret the score above: low output overall, or plenty of output that mostly missed the target?",
        higher_is_worse=False,
        fmt="{:.0f}",
    ),
]

# The one error/off-target measure that carries signal independent of `C`
# (see module docstring) — banded and shown in the dedicated error section.
ERROR_HEADLINE_MEASURES: list[HeadlineMeasure] = [
    HeadlineMeasure(
        key="OTH_rate",
        label="Off-target speech rate",
        description="Share of spoken words that don't even attempt the target letter, per 100 words. Still distinguishes groups after controlling for correct-word count — a genuinely separate signal, not just a restatement of reduced output.",
        higher_is_worse=True,
        fmt="{:.1f}",
    ),
]

# Remaining error/code rates: real but weaker or confounded (see
# docs/pft_new_scoring.md) — shown as raw numbers only, not banded.
# PN_rate: 90% of participants use zero proper nouns (floor effect).
# R_rate/SIM_rate: opportunity to repeat/resemble an earlier word scales
#   with how many correct words came before it, so these are structurally
#   biased toward lower values for lower-performing participants.
# F_rate (filled pauses): flat, doesn't separate groups here.
ERROR_EXPLORATORY_MEASURES = {
    "N_rate": "Nonword rate",
    "PN_rate": "Proper noun rate",
    "R_rate": "Repetition rate",
    "SIM_rate": "Similar-word rate",
    "F_rate": "Filled pause rate",
}

# Clustering/switching/AoA: real, monotonic effects on their own, but fully
# explained by `C` (see module docstring) — kept for the qualitative
# transcript view and research use, not as independent clinical flags.
EXPLORATORY_MEASURES = {
    "n_clusters": "Clusters produced",
    "n_switches": "Switches between clusters",
    "largest_cluster_size": "Largest cluster",
    "mean_aoa_correct": "Mean age-of-acquisition of correct words (yrs)",
}
