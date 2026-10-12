"""Pydantic schema for finance-management extraction output."""
from typing import Literal

from pydantic import BaseModel, Field

Manager = Literal["self", "other", "shared", "unclear"]
Changed = Literal["yes", "no", "unclear"]

ChangeReason = Literal[
    "cognitive_decline",
    "physical_health",
    "convenience_or_preference",
    "other",
]

# "extracted" is the only status the model's output maps to; "needs_review" is
# reserved for extract.py's own fallback when a completion can't be parsed/validated
# after retrying, or when verify.py can't ground the evidence_quote.
Status = Literal["extracted", "needs_review"]


class FinanceStatus(BaseModel):
    """Extracted finance-management status for one participant's transcript."""
    status: Status
    manager: Manager = "unclear"
    changed_recently: Changed = "unclear"
    change_reason: ChangeReason | None = Field(
        default=None, description="Why management changed -- set only when changed_recently is 'yes'."
    )
    summary: str = Field(default="", description="One-sentence paraphrase of the answer.")
    evidence_quote: str = Field(
        default="", description="Verbatim substring of the transcript supporting the extraction."
    )
