"""Grounding checks for recent-news (STN) extraction output.

An answer can have several claims (see schema.py), and each claim can have
several detail entries -- every level is checked independently, and
anything that fails is dropped rather than failing whatever contains it: a
bad detail only drops that one detail (the claim it belongs to, and its
match, stand on their own); a bad claim only drops that one claim (other
claims from the same answer stand on their own). A claim's
evidence_quote/details entries must actually appear in the transcript (same
convention as who_concerned/finance), and its matched_candidate_number, if
set, must be a real position in the candidate list that was actually shown
to the model -- the model is never asked to reproduce a date or headline
itself (see schema.py), precisely so this check can catch a
hallucinated/out-of-range number rather than trusting whatever came back.

is_grounded is a thin alias for the shared redaction-aware check -- see
dementia_speech.tasks._shared.redaction for why evidence_quote/details can
contain [NAME]/[PLACE] placeholders (the participant's own personal
references, per prompt.py's NEWS_REDACTION_INSTRUCTIONS) and still count
as grounded.
"""
from dementia_speech.tasks._shared.redaction import is_grounded_allowing_redaction as is_grounded
from dementia_speech.tasks.news.schema import NewsAnswerExtraction, NewsClaim


def is_valid_candidate_number(matched_candidate_number: int | None, n_candidates: int) -> bool:
    """True if matched_candidate_number is unset, or a real 1-indexed position
    in a candidate list of n_candidates entries."""
    if matched_candidate_number is None:
        return True
    return 1 <= matched_candidate_number <= n_candidates


def _sanitize_claim(claim: NewsClaim, transcript: str, n_candidates: int) -> NewsClaim | None:
    """None if the claim itself doesn't survive (ungrounded evidence_quote or
    an out-of-range match); otherwise the claim with any ungrounded detail
    entries individually dropped."""
    if not is_grounded(claim.evidence_quote, transcript):
        return None
    if not is_valid_candidate_number(claim.matched_candidate_number, n_candidates):
        return None
    grounded_details = [d for d in claim.details if is_grounded(d, transcript)]
    if len(grounded_details) == len(claim.details):
        return claim
    return claim.model_copy(update={"details": grounded_details})


def sanitize_claims(result: NewsAnswerExtraction, transcript: str, n_candidates: int) -> NewsAnswerExtraction:
    """Drop any claim that isn't grounded in the transcript or points outside
    the candidate list actually shown, and any detail entry within a
    surviving claim that isn't grounded -- keeps everything else intact
    rather than demoting the whole record. If every claim turns out
    invalid, the answer is left with an empty claims list, which
    scoring.categorize() treats the same as a specific answer matching
    nothing (-> 'unverifiable')."""
    if result.status != "extracted":
        return result
    sanitized = [_sanitize_claim(c, transcript, n_candidates) for c in result.claims]
    valid_claims = [c for c in sanitized if c is not None]
    if valid_claims == result.claims:
        return result
    return result.model_copy(update={"claims": valid_claims})
