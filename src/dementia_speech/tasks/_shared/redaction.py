"""Shared PID-redaction convention for every Qwen-based extraction task
(who_concerned, finance, memory_complaints, news -- any future one too).

Person names and specific place names must never reach a dashboard-facing
CSV (user request, 2026-11-11). Rather than a separate NER/gazetteer
post-processing pass -- which would also have to invent its own notion of
"verbatim grounding" once the text had already been changed -- every task's
own prompt instructs the model to substitute NAME_PLACEHOLDER/
PLACE_PLACEHOLDER directly into evidence_quote/details/summary as it
writes them (REDACTION_INSTRUCTIONS below, embedded in each task's own
SYSTEM_PROMPT), and is_grounded_allowing_redaction treats those exact
placeholders as wildcards when checking a quote is still a genuine
substring of the transcript, rather than breaking the existing verbatim-
quote grounding check that every task already relies on.

This is a best-effort instruction to the model, not a guaranteed filter --
same caveat as every other LLM-extracted field in this project. It will
miss a name/place the model fails to recognise as such, same as any LLM-
based redaction would. See docs/pid_redaction.md, and spot-check a sample
of dashboard-facing output after (re-)running any of these pipelines.
"""
import re

NAME_PLACEHOLDER = "[NAME]"
PLACE_PLACEHOLDER = "[PLACE]"

# Embedded verbatim into each task's own SYSTEM_PROMPT (f-string) so the
# instruction's wording can't drift between tasks.
REDACTION_INSTRUCTIONS = f"""Before writing any evidence_quote, detail, or summary: replace every \
person's name (the participant's own, a family member's, a friend's, a doctor's -- anyone named) \
with the literal text {NAME_PLACEHOLDER}, and every specific place name (a town, city, street, \
named building, or other named location) with the literal text {PLACE_PLACEHOLDER}. Do this even \
though evidence_quote/details must otherwise be copied character-for-character from the \
transcript -- the placeholder substitution is the one allowed exception to "verbatim". A \
relationship word on its own ("my daughter", "the doctor") is not a name and must be left as-is; \
only the actual name or place itself is replaced."""

_NORMALIZE_RE = re.compile(r"\s+")
# Up to a 3-word span (e.g. "Royal Hallamshire Hospital", "Sarah Jane Smith")
# stands in for the real name/place text a placeholder replaced -- bounded
# so a placeholder can't match an unboundedly large stretch of transcript.
_PLACEHOLDER_SPAN = r"\w+(?:[\s-]\w+){0,2}"


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", text.strip().lower())


def _redacted_pattern(quote: str) -> re.Pattern:
    pattern = re.escape(quote)
    pattern = pattern.replace(re.escape(NAME_PLACEHOLDER), _PLACEHOLDER_SPAN)
    pattern = pattern.replace(re.escape(PLACE_PLACEHOLDER), _PLACEHOLDER_SPAN)
    return re.compile(pattern, re.IGNORECASE)


def is_grounded_allowing_redaction(quote: str, transcript: str) -> bool:
    """True if quote is grounded in transcript -- same plain normalized
    substring check every task used before this feature when quote has no
    placeholders, or, when it does, treating each NAME_PLACEHOLDER/
    PLACE_PLACEHOLDER as a wildcard for the real name/place text that was
    there before the model redacted it."""
    if not quote:
        return False
    if NAME_PLACEHOLDER not in quote and PLACE_PLACEHOLDER not in quote:
        return _normalize(quote) in _normalize(transcript)
    return bool(_redacted_pattern(quote).search(transcript))
