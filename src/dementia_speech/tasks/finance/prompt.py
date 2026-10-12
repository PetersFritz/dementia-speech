"""Question text and few-shot prompt for finance-management extraction."""

# IVA q9 (q8 on some older systems) / "finance" prompt code, per data/raw/PromptsMapping.csv.
# Wording varies slightly by system; this is the fullest (cchat) version.
QUESTION_TEXT = "WHO MANAGES YOUR FINANCES? IS IT YOU OR SOMEBODY ELSE? HAS THIS CHANGED RECENTLY AND IF SO WHY?"

CHANGE_REASON_DESCRIPTIONS = {
    "cognitive_decline": "Change attributed to memory, thinking, or confusion problems.",
    "physical_health": "Change attributed to a physical illness, disability, or mobility problem.",
    "convenience_or_preference": "Change attributed to practicality or preference, with no health reason "
        "given -- e.g. a partner has always done it, retirement, simplicity.",
    "other": "A genuine reason given that doesn't fit any category above.",
}

_REASON_LIST = "\n".join(f"- {name}: {desc}" for name, desc in CHANGE_REASON_DESCRIPTIONS.items())

SYSTEM_PROMPT = f"""You are assisting a clinical dementia-screening study. Participants were asked:

    "{QUESTION_TEXT}"

Read their transcribed answer and extract three things: who currently manages their finances,
whether that has changed recently, and -- only if it changed -- why.

manager (use exactly one of these strings):
- "self": the participant manages their own finances
- "other": someone else manages them (partner, family, carer, professional)
- "shared": the participant and someone else manage together
- "unclear": the answer doesn't make this clear

changed_recently (use exactly one of these strings):
- "yes": they say management has changed
- "no": they say it hasn't changed
- "unclear": not stated or ambiguous

change_reason -- ONLY set when changed_recently is "yes", otherwise null (use exactly one of these strings):
{_REASON_LIST}

Respond with ONLY a single JSON object, no prose before or after it, matching this schema:
{{
  "manager": "self" | "other" | "shared" | "unclear",
  "changed_recently": "yes" | "no" | "unclear",
  "change_reason": "<one of the reason names above>" | null,
  "summary": "<one sentence paraphrase of the answer>",
  "evidence_quote": "<verbatim substring copied exactly from the transcript>"
}}

change_reason must be null unless changed_recently is "yes". Every evidence_quote must be copied
character-for-character from the transcript -- do not paraphrase inside the quote."""

_FEW_SHOT_EXAMPLES = [
    (
        "I've always done my own finances and that hasn't changed, I pay the bills online every month.",
        {
            "manager": "self",
            "changed_recently": "no",
            "change_reason": None,
            "summary": "Participant still manages their own finances; no change reported.",
            "evidence_quote": "I've always done my own finances and that hasn't changed",
        },
    ),
    (
        "My husband took it all over about a year ago because I kept forgetting to pay things and "
        "getting confused by the statements.",
        {
            "manager": "other",
            "changed_recently": "yes",
            "change_reason": "cognitive_decline",
            "summary": "Participant's husband took over finances about a year ago due to forgetfulness and confusion.",
            "evidence_quote": "My husband took it all over about a year ago because I kept forgetting to "
                "pay things and getting confused by the statements",
        },
    ),
]


def build_messages(transcript: str, retry_feedback: str | None = None) -> list[dict]:
    """Build the chat-template message list for one extraction call.

    retry_feedback, when set, is appended as an extra user turn asking the model to
    correct a previous malformed completion -- see extract.py's retry loop.
    """
    import json

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_transcript, example_output in _FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example_transcript})
        messages.append({"role": "assistant", "content": json.dumps(example_output)})
    messages.append({"role": "user", "content": transcript})
    if retry_feedback:
        messages.append({"role": "user", "content": retry_feedback})
    return messages
