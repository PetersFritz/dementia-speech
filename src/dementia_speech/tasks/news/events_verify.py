"""Grounding checks for the event-based recent-news pipeline (v2).

Same philosophy as verify.py: the model is never trusted to reproduce
anything it could hallucinate. Every quote and detail must actually appear in
the transcript (redaction-aware -- see dementia_speech.tasks._shared.
redaction), and every matched candidate number must be a real position in the
candidate list actually shown. Anything that fails is dropped individually,
never failing whatever contains it.
"""
from dementia_speech.tasks._shared.redaction import is_grounded_allowing_redaction as is_grounded
from dementia_speech.tasks.news.events_schema import (
    DetailLevel, EventAssessment, EventList, MentionedEvent,
)


def sanitize_events(result: EventList, transcript: str) -> EventList:
    """Drop any event whose evidence_quote isn't grounded in the transcript,
    and de-duplicate events that share the same quote."""
    if result.status != "extracted":
        return result
    seen_quotes: set[str] = set()
    kept: list[MentionedEvent] = []
    for event in result.events:
        quote = event.evidence_quote.strip()
        if not quote or quote in seen_quotes or not is_grounded(quote, transcript):
            continue
        seen_quotes.add(quote)
        kept.append(event)
    if kept == result.events:
        return result
    return result.model_copy(update={"events": kept})


def _level_from_detail_count(n: int) -> DetailLevel:
    if n == 0:
        return "none"
    return "some" if n <= 2 else "rich"


def sanitize_assessment(result: EventAssessment, transcript: str, n_candidates: int) -> EventAssessment:
    """Keep only real, de-duplicated candidate numbers (order preserved) and grounded details,
    then make detail_level consistent with the details that survived: with
    no grounded details the level can't be above "none", and with at least
    one it can't be "none" -- so the level the dashboard shows is always
    backed by quotable evidence."""
    if result.status != "extracted":
        return result

    # Keep the model's order (best match first) -- drop invalid numbers and
    # repeats, but never re-sort.
    numbers = list(dict.fromkeys(
        n for n in result.matched_candidate_numbers if 1 <= n <= n_candidates
    ))
    details = [d for d in result.details if is_grounded(d, transcript)]

    level = result.detail_level
    if not details:
        level = "none"
    elif level == "none":
        level = _level_from_detail_count(len(details))

    return result.model_copy(update={
        "matched_candidate_numbers": numbers,
        "details": details,
        "detail_level": level,
    })
