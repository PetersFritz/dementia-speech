"""Quote-grounding check: confirm the evidence_quote actually appears in the transcript."""
import re

from dementia_speech.tasks.finance.schema import FinanceStatus


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def is_grounded(evidence_quote: str, transcript: str) -> bool:
    """True if evidence_quote is a substring of transcript (whitespace/case-insensitive)."""
    return bool(evidence_quote) and _normalize(evidence_quote) in _normalize(transcript)


def drop_if_ungrounded(result: FinanceStatus, transcript: str) -> FinanceStatus:
    """Demote an extracted result to 'needs_review' if its evidence_quote isn't grounded.

    manager/changed_recently/change_reason are all read off the same quoted span, so an
    ungrounded quote means none of them can be trusted -- the whole record is flagged for
    a human look rather than keeping fields with no checkable backing.
    """
    if result.status == "extracted" and not is_grounded(result.evidence_quote, transcript):
        return FinanceStatus(status="needs_review")
    return result
