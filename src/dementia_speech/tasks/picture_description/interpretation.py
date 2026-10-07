"""Clinician-facing interpretation of Cookie Theft measures.

Converts raw metrics into plain-language, banded ("Typical" / "Borderline" /
"Atypical") flags relative to the healthy-control (HC) reference group, for
the subset of measures found to have a literature-consistent, monotonic
HC > MCI > Dementia (or reverse) effect in this project's own data — see
docs/cookie_theft_scoring.md, "What's actually trustworthy in this cohort".

Measures that don't separate groups reliably in this data (several of the
raw spatio-semantic graph counts turn out to be dominated by how much the
participant talked, not by cognitive content — see the same doc's
"Replication note") are deliberately left out of banding. Presenting those
with a confident Typical/Atypical label would overstate what the numbers
actually support here, even though the raw numbers remain available for
research use.

Banding itself (the Typical/Borderline/Atypical convention and the
HeadlineMeasure record type) is generic and shared with the PFT task — see
`dementia_speech.interpretation` for that and for the general rule on what
qualifies a measure for a "headline" list.
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
    HeadlineMeasure(
        key="mattr",
        label="Vocabulary variety",
        description="How varied the vocabulary is, controlling for how much was said.",
        higher_is_worse=False,
        fmt="{:.2f}",
    ),
    HeadlineMeasure(
        key="indefinite_term_rate",
        label="Vague word use",
        description="Rate of vague placeholder words (\"thing\", \"stuff\", ...) per 100 words — often standing in for a word the speaker couldn't retrieve.",
        higher_is_worse=True,
        fmt="{:.2f}",
    ),
]

# Remaining measures the pipeline computes but which don't band reliably in
# this cohort (non-monotonic across HC/MCI/Dementia, and/or a literature
# direction that doesn't replicate here) — shown as raw numbers in an
# "exploratory" section, not as clinical flags.
EXPLORATORY_MEASURES = {
    "iu_mentions_per_100_words": "IU mentions / 100 words",
    "lexical_density": "Lexical density",
    "pronoun_ratio": "Pronoun ratio",
    "cycles_per_100_words": "Repeated mentions / 100 words",
    "total_path_per_unique_nodes": "Path distance / unique node",
    "cross_ratio_quadrants": "Cross-quadrant ratio",
    "spatial_left_bias": "Spatial left bias",
}
