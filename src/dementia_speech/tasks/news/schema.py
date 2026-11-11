"""Pydantic schema for recent-news (STN) answer extraction output.

The model's job stops at identifying WHICH (if any) of the candidate events
shown in the prompt the participant's answer describes, plus a coarse
quality read on the answer itself -- it never judges recency or computes a
score itself; see scoring.py for why that's deliberately kept in code.

A single answer can describe more than one distinct event ("there was that
storm, and something about the Prime Minister resigning too") -- claims is a
list for exactly this reason, one entry per distinct event the participant
describes, rather than a single matched_candidate_number that would silently
drop every event after the first. scoring.py takes the best-supported claim
for the final category/score, but every claim is kept in the output for
audit/richness.
"""
from typing import Literal

from pydantic import BaseModel, Field

# "extracted" is the only status the model's output maps to; "needs_review" is
# reserved for extract.py's own fallback when a completion can't be parsed/
# validated after retrying.
Status = Literal["extracted", "needs_review"]

# - "specific": names at least one concrete enough event/person/topic that it
#   could in principle be checked against a real news event (whether or not
#   any of the candidates shown actually match it).
# - "vague": too generic to check against anything ("something about the
#   government", "the usual bad news").
# - "none": no real answer -- "I don't know", a refusal, or clearly
#   off-topic.
AnswerQuality = Literal["specific", "vague", "none"]


class NewsClaim(BaseModel):
    """One distinct event the participant's answer describes."""
    matched_candidate_number: int | None = Field(
        default=None,
        description=(
            "1-indexed position in the candidate event list shown in the "
            "prompt that this specific claim describes. Null if this claim "
            "doesn't match any of those exact candidates -- do not guess or "
            "invent a number."
        ),
    )
    evidence_quote: str = Field(
        default="",
        description=(
            "Verbatim substring of the transcript describing this specific claim, except that a "
            "personal name/place from the participant's OWN life (not the news event itself) is "
            "replaced with [NAME]/[PLACE] -- see prompt.py's NEWS_REDACTION_INSTRUCTIONS and "
            "dementia_speech.tasks._shared.redaction."
        ),
    )
    # Captures "fluent but empty" recall -- a participant can correctly
    # identify an event (matched_candidate_number set) while giving no
    # substance beyond naming it, which merely naming the topic can't
    # distinguish on its own. See scoring.py's detail metrics and
    # docs/news_scoring_rules.md.
    details: list[str] = Field(
        default_factory=list,
        description=(
            "Specific, concrete pieces of information the participant gave about "
            "THIS event, beyond just identifying/naming it -- a name, place, "
            "number, cause, or consequence they mentioned. Each entry must be a "
            "verbatim substring of the transcript, copied character-for-character "
            "(same rule as evidence_quote). Merely recognizing or naming the "
            "event ('there was a storm') is NOT itself a detail -- leave this "
            "an empty list if the participant adds nothing beyond that, even "
            "though matched_candidate_number may still be set."
        ),
    )


class NewsAnswerExtraction(BaseModel):
    """Extracted recent-news answer status for one participant's transcript."""
    status: Status
    answer_quality: AnswerQuality = "none"
    # One entry per distinct event described, when answer_quality is
    # "specific" -- empty when answer_quality is "vague" or "none". A claim
    # whose evidence_quote isn't grounded in the transcript, or whose
    # matched_candidate_number is out of range, is dropped by
    # verify.sanitize_claims() rather than failing the whole record.
    claims: list[NewsClaim] = Field(default_factory=list)
    summary: str = Field(default="", description="One-sentence paraphrase of the whole answer.")
