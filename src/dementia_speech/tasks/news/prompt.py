"""Question text and few-shot prompt for recent-news (STN) answer extraction.

PROMPT_VERSION is logged alongside every scored row (pipelines/score_news.py)
per docs/news_scoring_rules.md's audit requirement -- bump it whenever the
wording below changes, so old and new rows in the audit log stay
distinguishable.

Unlike who_concerned/finance's prompt.py, each call here needs its own
candidate-event list spliced into the user turn (retrieval.py narrows the
cache down to a short list per participant) -- build_messages() takes that
list already formatted by _format_candidates, not a fixed system prompt
alone.
"""
import json

import pandas as pd

from dementia_speech.tasks._shared.redaction import NAME_PLACEHOLDER, PLACE_PLACEHOLDER

PROMPT_VERSION = "v4"  # v4: redact the participant's own personal names/places (not news content)

# Deliberately NOT the shared REDACTION_INSTRUCTIONS text: `details` is
# designed to capture things like a storm's name or the place a news event
# happened as evidence of real substance (see the `details` instructions
# below) -- blanket-redacting every name/place would destroy that measure.
# Only the participant's own personal references need redacting here, never
# the public news content itself.
NEWS_REDACTION_INSTRUCTIONS = f"""If, while describing an event, the participant also mentions a \
person or place from their OWN life rather than the news itself -- a family member, friend, \
carer, doctor, their own home town, or similar -- replace that name with {NAME_PLACEHOLDER} and \
that place with {PLACE_PLACEHOLDER} in evidence_quote and details (the same allowed exception to \
"verbatim" as elsewhere). Do NOT redact names or places that are themselves part of the news \
event being described -- a storm's name, a politician, a country, the city something happened \
in, etc. Those are public information, not the participant's personal details, and redacting \
them would make the answer impossible to check against the candidate list above."""

# GPCOG-style "recent news" question. Exact per-system wording varies (see
# who_concerned/prompt.py's own note on this) -- this is a reasonable
# generic phrasing; check it against data/raw/matching_prompts.csv's STN row
# before relying on this for anything beyond a first pass.
QUESTION_TEXT = "WHAT HAS BEEN IN THE NEWS RECENTLY?"

SYSTEM_PROMPT = f"""You are assisting a clinical dementia-screening study. Participants were asked:

    "{QUESTION_TEXT}"

You will be shown their transcribed answer and a numbered list of real news events \
that were reported in the weeks before their interview. The answer may describe ONE \
event or SEVERAL distinct events -- find every one they describe, not just the first.

answer_quality (use exactly one of these strings, for the answer as a whole):
- "specific": names at least one concrete enough event, person, or topic that it could \
in principle be checked against a real news event -- whether or not any of the listed \
candidates actually turns out to match it.
- "vague": too generic to check against anything ("something about the government", \
"the usual bad news", "I don't really follow it but there's always something").
- "none": no real answer at all -- "I don't know", a refusal, or clearly off-topic.

claims: when answer_quality is "specific", list ONE entry per distinct event the answer \
describes (often just one, but list every one you find -- e.g. an answer mentioning both \
a storm and a tax announcement needs two entries, not one). Leave this an empty list when \
answer_quality is "vague" or "none". For each claim:
- matched_candidate_number: the exact number of the candidate from the list above that \
this specific claim describes, or null if it doesn't match any of them. Do NOT guess, \
invent a number, or pick the closest-sounding one just to have an answer -- a real but \
uncached event should get null, not a wrong guess.
- evidence_quote: the verbatim substring of the transcript describing this specific claim \
(not the whole answer if there are multiple claims -- just the part about this one event).
- details: a list of the SPECIFIC, CONCRETE facts the participant added about this \
particular event -- a name, place, number, cause, or consequence. This is important: \
someone can correctly recognise that an event happened (matched_candidate_number set) \
while giving no real substance about it at all, just filler around the bare topic \
("yes there was definitely something about a storm, it was a big deal, everyone was \
talking about it" -- fluent, but zero concrete content). Merely naming or recognising \
the topic is NOT a detail by itself -- only extra facts beyond that count. If the \
participant adds nothing beyond recognising the topic, details MUST be an empty list, \
even though matched_candidate_number is still set. Each entry must be a verbatim \
substring of the transcript, same rule as evidence_quote.

{NEWS_REDACTION_INSTRUCTIONS}

Respond with ONLY a single JSON object, no prose before or after it, matching this schema:
{{
  "answer_quality": "specific" | "vague" | "none",
  "claims": [
    {{"matched_candidate_number": <integer from the numbered list, or null>,
      "evidence_quote": "<verbatim substring copied exactly from the transcript>",
      "details": ["<verbatim substring>", ...]}},
    ...
  ],
  "summary": "<one sentence paraphrase of the whole answer>"
}}

Every evidence_quote and every details entry must be copied character-for-character from
the transcript -- do not paraphrase inside them."""


def _format_candidates(candidates: pd.DataFrame) -> str:
    if len(candidates) == 0:
        return "(no candidate events available for this date range)"
    lines = []
    for i, row in enumerate(candidates.itertuples(), start=1):
        lines.append(f"{i}. [{row.date}] {row.headline} — {row.summary}")
    return "\n".join(lines)


def _format_user_turn(transcript: str, candidates: pd.DataFrame) -> str:
    return (
        f'Participant\'s answer:\n"{transcript}"\n\n'
        f"Candidate recent news events (most recent first):\n{_format_candidates(candidates)}"
    )


# Each example embeds its own small candidate list, matching the real input
# format exactly (see module docstring) -- covers every answer_quality
# outcome, the multi-claim case, the "fluent but empty" case (examples 1
# and 5), and -- importantly, example 6 -- the personal-vs-public
# redaction distinction: "Storm Bertha" (public, part of the news) stays,
# "my neighbour Janet" and "Chesterfield" (the participant's own life) get
# redacted, in the very same evidence_quote.
_FEW_SHOT_EXAMPLES = [
    (
        "There was that big storm, I think they called it Bertha, that flooded a load of "
        "houses down south, something like three hundred homes I heard. And something "
        "about the Prime Minister too, an announcement about taxes.",
        pd.DataFrame([
            {"date": "2026-10-28", "headline": "Storm Bertha floods hundreds of homes across southern England",
             "summary": "Emergency services carried out dozens of rescues as rivers overtopped flood defences."},
            {"date": "2026-10-25", "headline": "Chancellor announces income tax threshold freeze",
             "summary": "The move was confirmed in a statement to Parliament on Thursday."},
            {"date": "2026-10-20", "headline": "Local council approves new bypass route",
             "summary": "Construction is expected to begin next spring."},
        ]),
        {
            "answer_quality": "specific",
            "claims": [
                {"matched_candidate_number": 1,
                 "evidence_quote": "that big storm, I think they called it Bertha, that flooded a load of houses down south, something like three hundred homes I heard",
                 "details": ["they called it Bertha", "something like three hundred homes I heard"]},
                {"matched_candidate_number": 2,
                 "evidence_quote": "something about the Prime Minister too, an announcement about taxes",
                 "details": []},
            ],
            "summary": "Participant describes a recent storm (with specifics) and a tax announcement (bare, no elaboration).",
        },
    ),
    (
        "Oh, just the usual, you know, politics and people arguing about something or other. "
        "I don't really pay close attention to be honest.",
        pd.DataFrame([
            {"date": "2026-10-28", "headline": "Storm Bertha floods hundreds of homes across southern England",
             "summary": "Emergency services carried out dozens of rescues as rivers overtopped flood defences."},
        ]),
        {
            "answer_quality": "vague",
            "claims": [],
            "summary": "Participant gives only a generic reference to politics, with no specific event.",
        },
    ),
    (
        "I'm not sure, I can't think of anything right now.",
        pd.DataFrame([
            {"date": "2026-10-28", "headline": "Storm Bertha floods hundreds of homes across southern England",
             "summary": "Emergency services carried out dozens of rescues as rivers overtopped flood defences."},
        ]),
        {
            "answer_quality": "none",
            "claims": [],
            "summary": "Participant says they cannot recall any recent news.",
        },
    ),
    (
        "There was an earthquake somewhere, I think it was in Turkey, quite a bad one. And "
        "that storm too, the one that caused all the flooding.",
        pd.DataFrame([
            {"date": "2026-10-28", "headline": "Storm Bertha floods hundreds of homes across southern England",
             "summary": "Emergency services carried out dozens of rescues as rivers overtopped flood defences."},
            {"date": "2026-10-25", "headline": "Chancellor announces income tax threshold freeze",
             "summary": "The move was confirmed in a statement to Parliament on Thursday."},
        ]),
        {
            "answer_quality": "specific",
            "claims": [
                {"matched_candidate_number": None,
                 "evidence_quote": "an earthquake somewhere, I think it was in Turkey, quite a bad one",
                 "details": []},
                {"matched_candidate_number": 1,
                 "evidence_quote": "that storm too, the one that caused all the flooding",
                 "details": []},
            ],
            "summary": "Participant describes an earthquake in Turkey (no matching candidate) and the recent storm.",
        },
    ),
    (
        "Oh yes, there was definitely something about a storm, I remember hearing about it "
        "on the television, it was quite a big deal, everyone was talking about it at the "
        "time.",
        pd.DataFrame([
            {"date": "2026-10-28", "headline": "Storm Bertha floods hundreds of homes across southern England",
             "summary": "Emergency services carried out dozens of rescues as rivers overtopped flood defences."},
        ]),
        {
            "answer_quality": "specific",
            "claims": [
                {"matched_candidate_number": 1,
                 "evidence_quote": "there was definitely something about a storm, I remember hearing about it on the television, it was quite a big deal, everyone was talking about it at the time",
                 "details": []},
            ],
            "summary": "Participant correctly recognises the recent storm but gives no concrete detail about it, just fluent filler around the bare topic.",
        },
    ),
    (
        "There was that storm, Storm Bertha, that flooded a load of homes. My neighbour Janet "
        "down in Chesterfield had water right up to her back door, she told me about it on "
        "the phone.",
        pd.DataFrame([
            {"date": "2026-10-28", "headline": "Storm Bertha floods hundreds of homes across southern England",
             "summary": "Emergency services carried out dozens of rescues as rivers overtopped flood defences."},
        ]),
        {
            "answer_quality": "specific",
            "claims": [
                {"matched_candidate_number": 1,
                 "evidence_quote": "that storm, Storm Bertha, that flooded a load of homes. My neighbour "
                     "[NAME] down in [PLACE] had water right up to her back door, she told me about it "
                     "on the phone",
                 "details": ["Storm Bertha", "had water right up to her back door"]},
            ],
            "summary": "Participant describes the recent storm, illustrated via a neighbour's own flooded home.",
        },
    ),
]


def build_messages(
    transcript: str, candidates: pd.DataFrame, retry_feedback: str | None = None
) -> list[dict]:
    """Build the chat-template message list for one extraction call.

    retry_feedback, when set, is appended as an extra user turn asking the
    model to correct a previous malformed completion -- see extract.py's
    retry loop.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_transcript, example_candidates, example_output in _FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": _format_user_turn(example_transcript, example_candidates)})
        messages.append({"role": "assistant", "content": json.dumps(example_output)})
    messages.append({"role": "user", "content": _format_user_turn(transcript, candidates)})
    if retry_feedback:
        messages.append({"role": "user", "content": retry_feedback})
    return messages
