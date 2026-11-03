"""Question text and few-shot prompt for who-is-concerned-about-memory extraction."""

# IVA q3 / "who-concerned" ("crn") prompt code, per data/PromptsMapping.csv. Wording
# varies slightly by system -- iva2/iva3/ivamem/ivastroke just ask who's most worried
# ("WHO IS MOST WORRIED ABOUT YOUR MEMORY YOU OR SOMEBODY ELSE"); cchat also asks for
# an example of a memory lapse, which this extraction deliberately ignores (only
# who's concerned matters here, not the example given). This is the fullest (cchat)
# version, same convention as finance/prompt.py. (iva1 bundles this with a separate
# short-term-weekend question into one "crnstw" code -- irrelevant here since no
# participant in this dataset uses iva1.)
QUESTION_TEXT = (
    "WHO IS MOST WORRIED ABOUT YOUR MEMORY? IS IT YOU, OR SOMEBODY ELSE? "
    "AND CAN YOU GIVE AN EXAMPLE OF THE LAST TIME YOUR MEMORY LET YOU DOWN."
)

SYSTEM_PROMPT = f"""You are assisting a clinical dementia-screening study. Participants were asked:

    "{QUESTION_TEXT}"

Read their transcribed answer and extract who is most worried about the participant's memory.
Ignore any example of a memory lapse they give -- only extract who is concerned.

who_concerned (use exactly one of these strings):
- "self": the participant says they themselves are the one worried
- "other": the participant says someone else (partner, family, friend, doctor) is the one worried
- "both": the participant says both themselves and someone else are worried
- "unclear": the answer doesn't make this clear

Respond with ONLY a single JSON object, no prose before or after it, matching this schema:
{{
  "who_concerned": "self" | "other" | "both" | "unclear",
  "summary": "<one sentence paraphrase of the answer>",
  "evidence_quote": "<verbatim substring copied exactly from the transcript>"
}}

Every evidence_quote must be copied character-for-character from the transcript -- do not
paraphrase inside the quote."""

_FEW_SHOT_EXAMPLES = [
    (
        "I think it's probably me that worries about it more than anyone, I notice when I forget things. "
        "Like last week I couldn't remember where I'd put my keys for about an hour.",
        {
            "who_concerned": "self",
            "summary": "Participant says they themselves are the one who worries about their memory.",
            "evidence_quote": "I think it's probably me that worries about it more than anyone",
        },
    ),
    (
        "My daughter keeps telling me I repeat myself, I don't really notice it myself but she's worried. "
        "She says I asked her the same question three times at dinner last week.",
        {
            "who_concerned": "other",
            "summary": "Participant's daughter is worried about their memory; participant doesn't notice it themselves.",
            "evidence_quote": "My daughter keeps telling me I repeat myself",
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
