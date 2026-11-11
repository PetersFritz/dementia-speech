"""Quote-grounding check: confirm each finding's evidence_quote actually appears in the transcript.

is_grounded is now a thin alias for the shared redaction-aware check -- see
dementia_speech.tasks._shared.redaction for why evidence_quote can contain
[NAME]/[PLACE] placeholders and still count as grounded.
"""
from dementia_speech.tasks._shared.redaction import is_grounded_allowing_redaction as is_grounded
from dementia_speech.tasks.memory_complaints.schema import ComplaintResult


def drop_ungrounded(result: ComplaintResult, transcript: str) -> ComplaintResult:
    """Return a copy of result with any finding whose evidence_quote isn't grounded removed.

    A 'complaint' result left with zero findings after dropping ungrounded ones is
    demoted to 'needs_review' rather than silently becoming an unflagged 'complaint'
    with an empty finding list -- that combination should always get a human look.
    """
    kept = [f for f in result.findings if is_grounded(f.evidence_quote, transcript)]
    if result.status == "complaint" and not kept:
        return ComplaintResult(status="needs_review", findings=[])
    return ComplaintResult(status=result.status, findings=kept)
