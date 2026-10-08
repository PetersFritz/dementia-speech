"""Quote-grounding check: confirm each finding's evidence_quote actually appears in the transcript."""
import re

from dementia_speech.tasks.memory_complaints.schema import ComplaintResult


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def is_grounded(evidence_quote: str, transcript: str) -> bool:
    """True if evidence_quote is a substring of transcript (whitespace/case-insensitive)."""
    return _normalize(evidence_quote) in _normalize(transcript)


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
