"""Pydantic schema for who-is-concerned-about-memory extraction output."""
from typing import Literal

from pydantic import BaseModel, Field

Concerned = Literal["self", "other", "both", "unclear"]

# "extracted" is the only status the model's output maps to; "needs_review" is
# reserved for extract.py's own fallback when a completion can't be parsed/validated
# after retrying, or when verify.py can't ground the evidence_quote.
Status = Literal["extracted", "needs_review"]


class WhoConcernedStatus(BaseModel):
    """Extracted who-is-concerned-about-memory status for one participant's transcript."""
    status: Status
    who_concerned: Concerned = "unclear"
    summary: str = Field(default="", description="One-sentence paraphrase of the answer.")
    evidence_quote: str = Field(
        default="",
        description=(
            "Verbatim substring of the transcript supporting the extraction, except that any "
            "person's name or specific place is replaced with [NAME]/[PLACE] -- see "
            "dementia_speech.tasks._shared.redaction."
        ),
    )
