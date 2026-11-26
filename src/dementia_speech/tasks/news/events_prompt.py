"""Prompts for the event-based recent-news (STN) pipeline (v2) -- two passes.

Both prompts put the model in the role of a clinician: someone assessing
people for possible cognitive impairment, who wants to know whether the
participant correctly identified real news events. (Pass 1 only IDENTIFIES
what was mentioned; pass 2 checks it against real articles and rates the
detail given.)

PROMPT_VERSION is logged with every scored row (pipelines/
score_news_events.py) -- bump it whenever any wording below changes.

Deliberately reuses dementia_speech.tasks.news.prompt.QUESTION_TEXT so the
question wording stays in one place.
"""
import json

import pandas as pd

from dementia_speech.tasks._shared.redaction import NAME_PLACEHOLDER, PLACE_PLACEHOLDER
from dementia_speech.tasks.news.prompt import QUESTION_TEXT

PROMPT_VERSION = "events-v2"  # v2: matched articles ordered best match first

_CLINICIAN_FRAMING = f"""You are an experienced clinician (a neuropsychologist) assessing people \
for possible cognitive impairment. As part of the assessment, each participant was asked:

    "{QUESTION_TEXT}"

What we want to know is whether the participant correctly identified real news events \
from the past week, and how much they actually knew about them. Be fair and literal: \
only credit what the participant actually said, and never fill gaps with your own \
assumptions about what they meant."""

# Only the participant's OWN personal references get redacted -- never the
# public news content (a storm's name, a politician, a country), since that is
# exactly the evidence of what they correctly identified.
_REDACTION = f"""If the participant mentions a person or place from their OWN life rather than \
the news itself -- a family member, friend, carer, doctor, their own home town, or similar -- \
replace that name with {NAME_PLACEHOLDER} and that place with {PLACE_PLACEHOLDER} wherever you \
quote it (the one allowed exception to "verbatim"). Do NOT redact names or places that are part \
of the news event itself -- a storm's name, a politician, a country, the city where something \
happened. Those are public information."""


# ---------------------------------------------------------------------------
# Pass 1 -- identify the events mentioned (no headlines shown)
# ---------------------------------------------------------------------------

IDENTIFY_SYSTEM_PROMPT = f"""{_CLINICIAN_FRAMING}

Your task now is only to IDENTIFY which news events the participant mentions. You are not \
shown any news articles and you must not judge whether what they said is true or recent -- \
that is checked separately.

answer_quality (use exactly one of these strings, for the answer as a whole):
- "specific": the participant names at least one concrete event, person, or topic that could \
in principle be checked against a real news story.
- "vague": too generic to identify any event ("something about the government", "the usual \
bad news", "I don't really follow it").
- "none": no real answer -- "I don't know", a refusal, or clearly off-topic.

events: when answer_quality is "specific", list ONE entry for EVERY distinct news event the \
participant mentions -- often one, but an answer that mentions a storm and a tax announcement \
needs two entries. Do not merge different events, and do not list the same event twice. Leave \
this an empty list when answer_quality is "vague" or "none". For each event:
- description: a short neutral description of the event (max ~15 words), in your own words, \
as specifically as the participant allows -- e.g. "storm causing flooding in southern England". \
It is used to search for matching news articles.
- evidence_quote: the verbatim substring of the transcript where the participant talks about \
THIS event (just that part, not the whole answer).

{_REDACTION}

Respond with ONLY a single JSON object, no prose before or after it:
{{
  "answer_quality": "specific" | "vague" | "none",
  "events": [
    {{"description": "<short neutral description>",
      "evidence_quote": "<verbatim substring copied exactly from the transcript>"}},
    ...
  ]
}}

Every evidence_quote must be copied character-for-character from the transcript -- do not \
paraphrase inside it."""

_IDENTIFY_EXAMPLES = [
    (
        "There was that big storm, I think they called it Bertha, that flooded a load of "
        "houses down south. And something about the Prime Minister too, an announcement "
        "about taxes.",
        {
            "answer_quality": "specific",
            "events": [
                {"description": "storm (Bertha) flooding homes in the south",
                 "evidence_quote": "that big storm, I think they called it Bertha, that flooded a load of houses down south"},
                {"description": "Prime Minister's announcement about taxes",
                 "evidence_quote": "something about the Prime Minister too, an announcement about taxes"},
            ],
        },
    ),
    (
        "Oh, just the usual, you know, politics and people arguing about something or other. "
        "I don't really pay close attention to be honest.",
        {"answer_quality": "vague", "events": []},
    ),
    (
        "I'm not sure, I can't think of anything right now.",
        {"answer_quality": "none", "events": []},
    ),
    (
        "There was that storm, Storm Bertha. My neighbour Janet down in Chesterfield had water "
        "right up to her back door, she told me about it on the phone.",
        {
            "answer_quality": "specific",
            "events": [
                {"description": "Storm Bertha causing flooding",
                 "evidence_quote": "that storm, Storm Bertha. My neighbour [NAME] down in [PLACE] had water right up to her back door"},
            ],
        },
    ),
]


def build_identify_messages(transcript: str, retry_feedback: str | None = None) -> list[dict]:
    """Chat messages for pass 1 (identify the events mentioned)."""
    messages = [{"role": "system", "content": IDENTIFY_SYSTEM_PROMPT}]
    for example_transcript, example_output in _IDENTIFY_EXAMPLES:
        messages.append({"role": "user", "content": f'Participant\'s answer:\n"{example_transcript}"'})
        messages.append({"role": "assistant", "content": json.dumps(example_output)})
    messages.append({"role": "user", "content": f'Participant\'s answer:\n"{transcript}"'})
    if retry_feedback:
        messages.append({"role": "user", "content": retry_feedback})
    return messages


# ---------------------------------------------------------------------------
# Pass 2 -- check ONE event against candidate articles, and rate the detail
# ---------------------------------------------------------------------------

ASSESS_SYSTEM_PROMPT = f"""{_CLINICIAN_FRAMING}

You are now shown the participant's full answer, ONE event they mentioned, and a numbered \
list of real news articles from the week before the interview. Decide two things about \
that one event.

1. matched_candidate_numbers: the numbers of EVERY article in the list that reports this \
same real-world event (several outlets often cover the same story -- include all of them), \
ORDERED BEST MATCH FIRST: put the article that most directly reports the event the participant \
described first, then any other articles covering the same event. The first number is shown to \
clinicians as "what the participant's answer matches", so choose it carefully. \
Be strict: an article about a different event on a similar topic is NOT a match (a different \
storm, a different tax announcement). If no article reports it, return an empty list -- the \
participant may be describing something older than the past week, which does not match. Do NOT \
guess, and do NOT pick the closest-sounding article just to have an answer -- a real event \
that simply isn't in the list should get an empty list.

2. detail_level and details: how much concrete information the participant gave about THIS \
event, beyond merely naming or recognising it. As a clinician you want to tell apart someone \
who genuinely knows about the event from someone producing fluent filler around a bare topic \
("yes there was something about a storm, it was a big deal, everyone was talking about it" \
-- confident, but no actual content).
- details: a list of the specific, concrete facts the participant gave about this event -- \
a name, place, number, cause, or consequence. Naming or recognising the topic is NOT a detail \
by itself. Each entry must be a verbatim substring of the participant's answer.
- detail_level (use exactly one of these strings):
  - "none": only names or recognises the topic; details is an empty list.
  - "some": one or two specific facts.
  - "rich": three or more specific facts, or a coherent account of what happened and why or \
with what consequence.
Rate the detail the participant ACTUALLY gave, independent of whether the event matched any \
article.

{_REDACTION}

Respond with ONLY a single JSON object, no prose before or after it:
{{
  "matched_candidate_numbers": [<integers from the numbered list, possibly empty>],
  "detail_level": "none" | "some" | "rich",
  "details": ["<verbatim substring>", ...]
}}"""


def _format_candidates(candidates: pd.DataFrame) -> str:
    if len(candidates) == 0:
        return "(no candidate articles available for this date range)"
    return "\n".join(
        f"{i}. [{row.date}] {row.headline} — {row.summary}"
        for i, row in enumerate(candidates.itertuples(), start=1)
    )


def _format_assess_turn(transcript: str, description: str, evidence_quote: str, candidates: pd.DataFrame) -> str:
    return (
        f'Participant\'s full answer:\n"{transcript}"\n\n'
        f"The event to assess: {description}\n"
        f'(Where they talk about it: "{evidence_quote}")\n\n'
        f"Candidate news articles (most recent first):\n{_format_candidates(candidates)}"
    )


_ASSESS_CANDIDATES = pd.DataFrame([
    {"date": "2026-10-28", "headline": "Storm Bertha floods hundreds of homes across southern England",
     "summary": "Emergency services carried out dozens of rescues as rivers overtopped flood defences."},
    {"date": "2026-10-28", "headline": "Hundreds evacuated as Storm Bertha hits the south coast",
     "summary": "Rail services were suspended and flood warnings remain in place."},
    {"date": "2026-10-25", "headline": "Chancellor announces income tax threshold freeze",
     "summary": "The move was confirmed in a statement to Parliament on Thursday."},
    {"date": "2026-10-20", "headline": "Local council approves new bypass route",
     "summary": "Construction is expected to begin next spring."},
])

# Covers: a match with concrete detail and TWO covering articles (example 1);
# a match with zero detail -- the "fluent but empty" case (example 2); a real
# event that matches nothing in the list (example 3); a different-event-same-
# topic non-match (example 3 also tests strictness); and the personal-vs-
# public redaction split (example 4).
_ASSESS_EXAMPLES = [
    (
        "There was that big storm, I think they called it Bertha, that flooded a load of houses "
        "down south, something like three hundred homes I heard.",
        "storm (Bertha) flooding homes in the south",
        "that big storm, I think they called it Bertha, that flooded a load of houses down south, something like three hundred homes I heard",
        {"matched_candidate_numbers": [1, 2],
         "detail_level": "rich",
         "details": ["they called it Bertha", "flooded a load of houses down south", "something like three hundred homes I heard"]},
    ),
    (
        "Oh yes, there was definitely something about a storm, I remember hearing about it on the "
        "television, it was quite a big deal, everyone was talking about it at the time.",
        "a storm that was widely discussed",
        "there was definitely something about a storm, I remember hearing about it on the television, it was quite a big deal, everyone was talking about it at the time",
        {"matched_candidate_numbers": [1, 2], "detail_level": "none", "details": []},
    ),
    (
        "There was an earthquake somewhere, I think it was in Turkey, quite a bad one.",
        "earthquake in Turkey",
        "an earthquake somewhere, I think it was in Turkey, quite a bad one",
        {"matched_candidate_numbers": [], "detail_level": "some", "details": ["I think it was in Turkey"]},
    ),
    (
        "There was that storm, Storm Bertha. My neighbour Janet down in Chesterfield had water "
        "right up to her back door, she told me about it on the phone.",
        "Storm Bertha causing flooding",
        "that storm, Storm Bertha. My neighbour [NAME] down in [PLACE] had water right up to her back door",
        {"matched_candidate_numbers": [1, 2],
         "detail_level": "some",
         "details": ["Storm Bertha", "had water right up to her back door"]},
    ),
]


def build_assess_messages(
    transcript: str,
    description: str,
    evidence_quote: str,
    candidates: pd.DataFrame,
    retry_feedback: str | None = None,
) -> list[dict]:
    """Chat messages for pass 2 (assess ONE event against its candidate list)."""
    messages = [{"role": "system", "content": ASSESS_SYSTEM_PROMPT}]
    for ex_transcript, ex_description, ex_quote, ex_output in _ASSESS_EXAMPLES:
        messages.append({
            "role": "user",
            "content": _format_assess_turn(ex_transcript, ex_description, ex_quote, _ASSESS_CANDIDATES),
        })
        messages.append({"role": "assistant", "content": json.dumps(ex_output)})
    messages.append({
        "role": "user",
        "content": _format_assess_turn(transcript, description, evidence_quote, candidates),
    })
    if retry_feedback:
        messages.append({"role": "user", "content": retry_feedback})
    return messages
