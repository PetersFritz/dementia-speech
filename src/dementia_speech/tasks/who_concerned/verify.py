"""Quote-grounding check: confirm the evidence_quote actually appears in the transcript.

is_grounded is now a thin alias for the shared redaction-aware check -- see
dementia_speech.tasks._shared.redaction for why evidence_quote can contain
[NAME]/[PLACE] placeholders and still count as grounded.
"""
from dementia_speech.tasks._shared.redaction import is_grounded_allowing_redaction as is_grounded
from dementia_speech.tasks.who_concerned.schema import WhoConcernedStatus


def drop_if_ungrounded(result: WhoConcernedStatus, transcript: str) -> WhoConcernedStatus:
    """Demote an extracted result to 'needs_review' if its evidence_quote isn't grounded.

    who_concerned is read off the quoted span, so an ungrounded quote means it can't be
    trusted -- the whole record is flagged for a human look rather than keeping a field
    with no checkable backing.
    """
    if result.status == "extracted" and not is_grounded(result.evidence_quote, transcript):
        return WhoConcernedStatus(status="needs_review")
    return result
