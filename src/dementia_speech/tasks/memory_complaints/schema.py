"""Pydantic schema for memory-complaint extraction output."""
from typing import Literal

from pydantic import BaseModel, Field

Domain = Literal[
    "short_term_memory",
    "long_term_memory",
    "word_finding",
    "orientation",
    "attention_concentration",
    "executive_function",
    "other",
]

# "complaint" / "no_complaint" are the only values the model is asked to
# produce (see prompt.py). "needs_review" is reserved for extract.py's own
# fallback when a completion can't be parsed/validated after retrying --
# never something the model should choose itself.
Status = Literal["complaint", "no_complaint", "needs_review"]


class Finding(BaseModel):
    """One memory-related complaint extracted from a transcript."""
    domain: Domain
    summary: str = Field(description="One-sentence paraphrase of the complaint.")
    evidence_quote: str = Field(
        description="Verbatim substring of the transcript supporting this finding."
    )


class ComplaintResult(BaseModel):
    """Full extraction result for one participant's transcript."""
    status: Status
    findings: list[Finding] = Field(default_factory=list)
