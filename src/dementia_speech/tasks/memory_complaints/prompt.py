"""Question text, domain taxonomy, and the few-shot prompt for memory-complaint extraction."""
from dementia_speech.tasks._shared.redaction import REDACTION_INSTRUCTIONS

# IVA q2 / "memory-problems" prompt code, per data/raw/PromptsMapping.csv.
QUESTION_TEXT = "TELL ME WHAT PROBLEMS HAVE YOU HAD WITH YOUR MEMORY"

DOMAIN_DESCRIPTIONS = {
    "short_term_memory": "Forgetting recent conversations, misplacing items, repeating questions, forgetting appointments.",
    "long_term_memory": "Difficulty recalling distant/autobiographical events or names from the past.",
    "word_finding": "Trouble finding the right word, naming objects, tip-of-the-tongue states.",
    "orientation": "Confusion about time, date, or place.",
    "attention_concentration": "Difficulty concentrating or following conversations -- only use this domain "
        "when the participant themselves frames it as an attention/concentration problem, not memory.",
    "executive_function": "Difficulty planning, organising, or multitasking.",
    "other": "A genuine memory-related complaint that doesn't fit any domain above.",
}

_DOMAIN_LIST = "\n".join(f"- {name}: {desc}" for name, desc in DOMAIN_DESCRIPTIONS.items())

SYSTEM_PROMPT = f"""You are assisting a clinical dementia-screening study. Participants were asked:

    "{QUESTION_TEXT}"

Read their transcribed answer and decide whether they are reporting one or more genuine memory
complaints. A complaint must be about memory specifically -- if the participant attributes their
difficulty to something else (hearing, attention, tiredness, general health) rather than memory,
or explicitly denies having memory problems, that is NOT a memory complaint.

Domains (use exactly one of these strings per finding):
{_DOMAIN_LIST}

{REDACTION_INSTRUCTIONS}

Respond with ONLY a single JSON object, no prose before or after it, matching this schema:
{{
  "status": "complaint" | "no_complaint",
  "findings": [
    {{"domain": "<one of the domain names above>", "summary": "<one sentence paraphrase>",
      "evidence_quote": "<verbatim substring copied exactly from the transcript>"}}
  ]
}}

If status is "no_complaint", findings must be an empty list. Every evidence_quote must be copied
character-for-character from the transcript, except where instructed above to substitute a
redaction placeholder -- do not paraphrase inside the quote."""

_FEW_SHOT_EXAMPLES = [
    (
        "I keep forgetting where I've put my keys and I've started writing myself notes "
        "because I forget what my daughter tells me on the phone sometimes.",
        {
            "status": "complaint",
            "findings": [
                {
                    "domain": "short_term_memory",
                    "summary": "Participant forgets where they put everyday items and forgets recent phone conversations.",
                    "evidence_quote": "I keep forgetting where I've put my keys",
                },
                {
                    "domain": "short_term_memory",
                    "summary": "Has started writing notes to compensate for forgetting recent conversations.",
                    "evidence_quote": "I forget what my daughter tells me on the phone sometimes",
                },
            ],
        },
    ),
    (
        "My daughter Emma says I asked her the same thing three times last week when she visited "
        "from Leeds, I don't even remember the conversation at all.",
        {
            "status": "complaint",
            "findings": [
                {
                    "domain": "short_term_memory",
                    "summary": "Participant repeated the same question several times and has no memory of the conversation.",
                    "evidence_quote": "My daughter [NAME] says I asked her the same thing three times last "
                        "week when she visited from [PLACE], I don't even remember the conversation at all",
                },
            ],
        },
    ),
    (
        "Not really memory problems as such, I just don't hear as well as I used to so I miss "
        "things people say and then it looks like I've forgotten, but it's my hearing not my memory.",
        {
            "status": "no_complaint",
            "findings": [],
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
