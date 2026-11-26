"""Pydantic schemas for the event-based recent-news (STN) pipeline (v2).

Unlike schema.py (which asks the model, in ONE call, to match the whole answer
against a keyword-retrieved candidate list), this pipeline works per event, in
two model passes:

  Pass 1 -- identify_events: read the transcript ALONE (no headlines shown) and
            list every distinct news event the participant mentions.
  Pass 2 -- assess_event, once per event: show that one event plus a short
            candidate list retrieved FOR THAT EVENT, and ask which articles
            (if any) report it, and how much concrete detail the participant
            gave about it.

The model never judges recency or computes a score -- whether a matched
article falls inside the 7-day window is plain date arithmetic in
events_scoring.py. See docs/news_scoring_rules.md.
"""
from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["extracted", "needs_review"]

# - "specific": names at least one concrete event the participant mentions
#   (whether or not it turns out to match a real article).
# - "vague": too generic to identify any event ("something about politics").
# - "none": no real answer -- "I don't know", a refusal, or off-topic.
AnswerQuality = Literal["specific", "vague", "none"]

# How much concrete information the participant gave about ONE event, beyond
# merely naming or recognising it:
# - "none": only names/recognises the topic -- fluent filler around a bare
#   topic counts as none.
# - "some": one or two specific facts (a name, place, number, cause, outcome).
# - "rich": three or more specific facts, or a coherent account of what
#   happened and why/with what consequence.
DetailLevel = Literal["none", "some", "rich"]


class MentionedEvent(BaseModel):
    """Pass 1 output: one distinct news event the participant mentions."""
    description: str = Field(
        description=(
            "A short neutral description (max ~15 words) of the event, in the "
            "model's own words -- used only to search for matching articles, "
            "never shown as the participant's own words."
        ),
    )
    evidence_quote: str = Field(
        description=(
            "Verbatim substring of the transcript where the participant "
            "describes THIS event (except that personal names/places from the "
            "participant's OWN life are replaced with [NAME]/[PLACE])."
        ),
    )


class EventList(BaseModel):
    """Pass 1 output for one participant's transcript."""
    status: Status
    answer_quality: AnswerQuality = "none"
    events: list[MentionedEvent] = Field(default_factory=list)


class EventAssessment(BaseModel):
    """Pass 2 output for ONE event."""
    status: Status
    matched_candidate_numbers: list[int] = Field(
        default_factory=list,
        description=(
            "1-indexed positions, in the candidate list shown, of EVERY article "
            "that reports this same event, ORDERED BEST MATCH FIRST (the first "
            "is the article that most directly reports the event the "
            "participant described). Empty if none do -- never guess or "
            "pick the closest-sounding article just to have an answer."
        ),
    )
    detail_level: DetailLevel = "none"
    details: list[str] = Field(
        default_factory=list,
        description=(
            "Specific, concrete facts the participant gave about THIS event "
            "beyond naming it (a name, place, number, cause, consequence). Each "
            "must be a verbatim substring of the transcript. Empty if the "
            "participant only names or recognises the topic."
        ),
    )
