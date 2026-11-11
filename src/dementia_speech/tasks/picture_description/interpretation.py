"""Clinician-facing interpretation of Cookie Theft measures.

Converts raw metrics into an age-adjusted percentile (0-100, see
`dementia_speech.interpretation.percentile_age_adjusted`) relative to the
healthy-control (HC) reference group, for the subset of measures found to
have a literature-consistent, monotonic HC > MCI > Dementia (or reverse)
effect in this project's own data — see docs/cookie_theft_scoring.md,
"What's actually trustworthy in this cohort".

Measures that don't separate groups reliably in this data (several of the
raw spatio-semantic graph counts turn out to be dominated by how much the
participant talked, not by cognitive content — see the same doc's
"Replication note") are deliberately left out. Presenting those as if they
carried the same weight would overstate what the numbers actually support
here, even though the raw numbers remain available for research use.

No Typical/Borderline/Atypical band, by design (changed 2026-11-14): this
module briefly used per-measure cutoffs calibrated via ROC analysis instead
of the generic +/-1/+/-2 SD convention, but any fixed cutoff -- calibrated
or not -- bakes in a sensitivity/specificity tradeoff the dashboard
shouldn't make unilaterally, and this cohort's modest individual-level
discrimination (HC-vs-MCI/Dementia AUCs mostly 0.5-0.7) meant most Dementia
participants still looked "Typical" no matter how loose the cutoff got
(~15-55% sensitivity depending on the specificity floor chosen). A
continuous percentile makes no such judgement call -- it's purely
descriptive, and the clinician applies whatever threshold matters to them.
See `dementia_speech.interpretation.percentile_age_adjusted`.

Age-adjustment itself (fitting `measure ~ age` on the TRAIN-split HC group,
via `dementia_speech.interpretation.fit_age_adjusted_norms`) and the
HeadlineMeasure record type are generic and shared with the PFT task — see
`dementia_speech.interpretation` for the general rule on what qualifies a
measure for a "headline" list.

HEADLINE_MEASURES cut from 5 to 3 (2026-11-09) to declutter a tab that had
grown to 13 measures total -- see pipelines/check_cookie_theft_measure_separation.py
for the Cohen's d / monotonicity sweep this was decided from (HC vs MCI and
HC vs Dementia both checked, not just Dementia -- MCI is the clinically
harder, earlier case). `iu_score` and `total_words` were the two strongest
standalone effects in the whole sweep (d=-0.83/-0.52 HC vs Dementia); `mattr`
and `indefinite_term_rate` (d=-0.34/0.22) were the weakest of the original 5
and are demoted to the exploratory/"Additional measures" section below, not
dropped outright. `path_distance_per_100_words` (d=0.46) stayed despite
being weaker than several raw, unnormalized alternatives (e.g.
`self_cycles_quadrants`, d=-0.65) -- it's deliberately confound-corrected
(normalized against total_words, unlike those raw counts, which are
suspected of mostly restating "talked less" rather than carrying
independent signal -- unchecked as of this cut) and already replicates a
specific literature finding (Ambadi et al. 2021 report `total_path_distance`,
its unnormalized form, as one of six measures significant in their own
1,349-participant cohort).
"""

from dementia_speech.interpretation import HeadlineMeasure

HEADLINE_MEASURES: list[HeadlineMeasure] = [
    HeadlineMeasure(
        key="iu_score",
        label="Content coverage",
        description="Share of the picture's key content (people, objects, actions) mentioned at all.",
        higher_is_worse=False,
        fmt="{:.0%}",
    ),
    HeadlineMeasure(
        key="total_words",
        label="Verbal output",
        description="Total words spoken describing the picture.",
        higher_is_worse=False,
        fmt="{:.0f}",
    ),
    HeadlineMeasure(
        key="path_distance_per_100_words",
        label="Visual scanning effort",
        description="How much the description jumps around the picture, per 100 words spoken.",
        higher_is_worse=True,
        fmt="{:.0f}",
    ),
]

# Remaining measures the pipeline computes, shown in a collapsed
# "Additional measures" section with a value, an age-adjusted percentile,
# and a group-comparison graph each (see app/modules/cookie_theft.py) -- no
# band, same as HEADLINE_MEASURES above now. These are still a separate,
# secondary section rather than promoted to HEADLINE_MEASURES: each only
# separates HC, MCI, and Dementia from EACH OTHER on SOME pairwise
# contrasts, not all three the way the 3 headline measures do (see each
# description for which) -- see the per-pair Cohen's d evidence in
# pipelines/scratch/analyse_cookie_theft_measures.py and
# data/cookie_theft_age_adjusted_scores.csv for the specifics quoted below.
#
# Widened 2026-11-12 from a 3-measure cut to every clinically-distinct,
# literature-grounded, non-duplicate measure the pipeline computes, then
# narrowed again the same day to drop two categories: measures with no real
# TRAIN/TEST signal either way, and measures that run opposite to their
# expected/literature direction -- "clinically informative" doesn't extend
# to a measure that's actively misleading or just noise. What's left is
# correctly-directed on at least one real pairwise contrast in both splits
# (even where it's null on a different contrast, which is captioned, not
# hidden):
#   - `mattr`: correct direction (not reversed) on every contrast in both
#     splits, though the effect is near-zero for HC-vs-MCI in the larger
#     TRAIN sample -- kept as a genuine, if uneven, signal, not a reversal.
#   - `indefinite_term_rate`: weak but correctly-signed means in both
#     splits; the known floor effect (71% score zero) explains why it's
#     weak, not why it would be wrong.
#   - `total_path_per_node` / `cross_ratio_quadrants`: correctly separate
#     HC from MCI/Dementia in both splits; null (not reversed) on MCI vs
#     Dementia specifically.
#   - `avg_x`: correctly separates MCI from Dementia and HC from Dementia in
#     both splits; reversed only on HC vs MCI, kept anyway (by request) as
#     a severity/staging signal rather than a screening one.
#
# Dropped for running opposite to their expected/literature direction in
# BOTH splits (not just noise -- a reproducible reversal): `semantic_relevance`
# and `iu_mentions_per_100_words`/`iu_mention_density` (MCI/dementia scored
# higher, not lower), `cycles_per_100_words` (HC repeated content more, not
# less -- confirms docs/cookie_theft_scoring.md's "Replication note"),
# `lexical_density` (dementia scored higher density, opposite of Boschi et
# al.'s "emptier speech" prediction), `stdev_x` (opposite of Ambadi et al.'s
# direction here).
#
# Dropped for having no reliable signal either way across TRAIN/TEST (sign
# flips between splits, or null in both): `pronoun_ratio`,
# `total_path_per_unique_nodes` (also outperformed by total_path_per_node,
# kept above, which normalizes the same distance differently).
#
# Excluded as a TRUE duplicate of another entry regardless of performance:
# `iu_mention_density` (same ratio as `iu_mentions_per_100_words` at a
# different scale -- moot now, both are dropped above anyway),
# `lexical_diversity` (the length-confounded version of what `mattr`
# already answers), `spatial_left_bias` (a pure sign-flipped re-expression
# of `avg_x`), `total_path_distance` (raw, unnormalized -- correlates
# 0.6-0.87 with `total_words` in this cohort, so it mostly restates verbal
# output length, already a headline measure).
#
# Extended 2026-11-17 with 6 further measures from a 27-item candidate list
# (pipelines/compute_cookie_theft_candidate_measures.py computes the raw
# values; pipelines/check_cookie_theft_additional_measures.py is the
# TRAIN/TEST direction-stability screen the final 6 were chosen from --
# same bar as the original 5 above, not just a full-sample AUC/Cohen's d,
# which can't tell a stable effect from a reversed or fluke one).
# Candidates dropped at that screening stage, with reasons:
#   - Utterance-count/MLU and transcriber pause-annotation measures: user
#     call on reliability -- sentence and pause annotation isn't a
#     standardised transcription step here (pause annotations present in
#     only ~58% of CTD transcripts; "not annotated" isn't "no pauses").
#   - total_content_units/correct_information_units: literal duplicate of
#     iu_score's numerator, already a headline measure.
#   - content_units_left/right, noun_count, verb_count,
#     immediate_word_repetitions, total_audio_duration_seconds: raw,
#     unnormalized counts/durations superseded by a ratio/rate version below
#     that avoids the same "mostly restates how much the participant talked"
#     confound already documented for the raw graph counts above.
#   - region_coverage: ceiling effect (~97-100% for nearly everyone), no
#     room to separate groups regardless of direction-stability.
#   - filled_pause_count/rate, words_per_minute, syllables_per_minute,
#     subordinate_clause_rate, mean_dependency_depth: direction flips sign
#     between TRAIN and TEST -- no reliable signal either way.
#   - grammatical_accuracy_proxy: direction was stable, but it's a spaCy
#     clause-form heuristic (subject + finite verb present), not a
#     validated grammar/error checker -- left out on construct-validity
#     grounds, not a statistical one.
#   - total_syllables: tracks total_words too closely to earn its own slot
#     (user call); syllables_per_content_unit (a ratio, not a raw count)
#     is kept below.
#
# Trimmed by two the same day (user call, not a new statistical finding):
#   - `avg_x` ("Spatial position") and `left_right_cu_ratio` ("Left/right
#     content balance") measure the same underlying left-shift construct two
#     ways -- order-weighted mean x-position vs. simple presence counts.
#     Keeping both double-counts one effect as if it were two independent
#     findings (the exact thing this list's selection criteria already
#     guard against elsewhere -- see the HEADLINE_MEASURES docstring's
#     general rule). `avg_x` is the better-validated of the pair (replicates
#     Ambadi et al.'s published result, separates slightly better here:
#     AUC 0.596 vs 0.565) but its raw value ("513") only means something
#     with the picture's pixel layout in mind; `left_right_cu_ratio`'s value
#     ("0.56") reads directly as "56% right-sided" with no such context
#     needed. Kept `left_right_cu_ratio`, dropped `avg_x` -- interpretability
#     over validation strength, a deliberate trade, not an oversight.
#   - `indefinite_term_rate` ("Vague word use"): weakest separation of the
#     11 (AUC 0.520) -- see the "known floor effect" note above for why it
#     was weak to begin with.
#
# Final list below is every measure in this section, sorted by HC-vs-
# Dementia separation strength (AUC, full sample) -- see
# cookie_theft_candidate_measures_summary.csv.
EXPLORATORY_MEASURES: list[HeadlineMeasure] = [
    HeadlineMeasure(
        key="speech_duration_seconds",
        label="Response duration",
        description=(
            "Total time spent speaking, silences trimmed, estimated from the "
            "recording via an energy threshold -- not forced alignment or "
            "diarization. A small minority of recordings include another "
            "speaker's voice, which inflates this value for that participant."
        ),
        higher_is_worse=False,
        fmt="{:.0f}s",
    ),
    HeadlineMeasure(
        key="repetition_rate_per_100_words",
        label="Repetition rate",
        description=(
            "Immediate word-for-word repetitions (\"the the\"), per 100 words. "
            "Only catches that specific pattern, not other self-corrections, so "
            "a low value here doesn't rule those out."
        ),
        higher_is_worse=True,
        fmt="{:.2f}",
    ),
    HeadlineMeasure(
        key="cross_ratio_quadrants",
        label="Attention switching",
        description=(
            "How often attention switched between quadrants of the picture, versus "
            "staying within one. Separates typical from impaired performance; doesn't "
            "distinguish MCI from dementia severity."
        ),
        higher_is_worse=True,
        fmt="{:.2f}",
    ),
    HeadlineMeasure(
        key="total_path_per_node",
        label="Scanning distance per mention",
        description=(
            "How far the description jumps around the picture, per mention of picture "
            "content (including repeats). Separates typical from impaired performance; "
            "doesn't distinguish MCI from dementia severity."
        ),
        higher_is_worse=True,
        fmt="{:.0f}",
    ),
    HeadlineMeasure(
        key="mattr",
        label="Vocabulary variety",
        description="Variety of words used, adjusted for how much the participant talked.",
        higher_is_worse=False,
        fmt="{:.2f}",
    ),
    HeadlineMeasure(
        key="noun_verb_ratio",
        label="Noun/verb balance",
        description=(
            "Ratio of nouns to lexical verbs (auxiliaries like \"is\"/\"has\" excluded "
            "from the verb count). A lower ratio in this cohort goes with the impaired "
            "direction."
        ),
        higher_is_worse=False,
        fmt="{:.2f}",
    ),
    HeadlineMeasure(
        key="syllables_per_content_unit",
        label="Speech required per concept",
        description=(
            "Total syllables spoken divided by content units mentioned. In this cohort "
            "a LOWER value goes with impaired performance -- it tracks reduced overall "
            "verbal output rather than cleaner efficiency, so read it alongside Verbal "
            "output rather than as a standalone efficiency score."
        ),
        higher_is_worse=False,
        fmt="{:.1f}",
    ),
    HeadlineMeasure(
        key="left_right_cu_ratio",
        label="Left/right content balance",
        description=(
            "Ratio of mentioned content units on the picture's left half to its right "
            "half. A higher ratio -- relatively more left-sided content -- is the "
            "spatial-neglect pattern reported by Ambadi et al. (2021)."
        ),
        higher_is_worse=True,
        fmt="{:.2f}",
    ),
    HeadlineMeasure(
        key="content_units_per_minute",
        label="Meaningful concepts per minute",
        description=(
            "Content units mentioned per minute of speech. In this cohort this is "
            "pulled UP by shorter, more compressed responses rather than cleaner "
            "information packing -- a higher value goes with the impaired direction "
            "here, the opposite of what the label alone might suggest. The duration "
            "denominator is a rough automatic estimate, not validated timing."
        ),
        higher_is_worse=True,
        fmt="{:.1f}",
    ),
]
