"""Run the two-pass event-based recent-news extraction with a local Qwen model.

Pass 1 (identify_events): transcript -> the distinct events mentioned.
Pass 2 (assess_event): one event + its own candidate articles -> which
articles report it, and how much detail the participant gave.

Loading the model requires torch/transformers and a GPU -- only import this
module from code that's actually running on a GPU node (e.g. pipelines/
score_news_events.py), not from the dashboard/scoring side of the project.
"""
import json
from typing import Callable

import pandas as pd
from pydantic import BaseModel, ValidationError

from dementia_speech.tasks._shared.model import load_model
from dementia_speech.tasks.news.events_prompt import build_assess_messages, build_identify_messages
from dementia_speech.tasks.news.events_schema import EventAssessment, EventList

MAX_RETRIES = 3
IDENTIFY_MAX_NEW_TOKENS = 500   # several events, each with a quote
ASSESS_MAX_NEW_TOKENS = 250     # one event: a few numbers + a few details


def _generate(messages: list[dict], max_new_tokens: int) -> str:
    tokenizer, model = load_model()
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    completion = output[0][encoded["input_ids"].shape[-1]:]
    return tokenizer.decode(completion, skip_special_tokens=True)


def _parse_json(raw: str, schema: type[BaseModel]) -> BaseModel:
    text = raw.strip()
    if text.startswith("```"):
        # Strip a markdown code fence if the model wrapped the JSON in one.
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    data = json.loads(text)
    data["status"] = "extracted"
    return schema.model_validate(data)


def _run_with_retries(
    build: Callable[[str | None], list[dict]],
    schema: type[BaseModel],
    max_new_tokens: int,
    fallback: BaseModel,
) -> tuple[BaseModel, str]:
    """Generate, parse and validate, retrying on a malformed completion.

    Returns (result, raw_completion_of_the_attempt_that_was_used) -- the raw
    text is logged by the pipeline for the audit requirement, even on the
    final fallback. Falls back to `fallback` (status='needs_review') after
    MAX_RETRIES failed attempts rather than raising, so a batch run never
    aborts on one bad completion.
    """
    feedback = None
    raw = ""
    for _ in range(MAX_RETRIES):
        raw = _generate(build(feedback), max_new_tokens)
        try:
            return _parse_json(raw, schema), raw
        except (json.JSONDecodeError, ValidationError) as e:
            feedback = (
                f"Your last response could not be parsed ({e}). "
                "Return ONLY the JSON object, with no other text."
            )
    return fallback, raw


def identify_events(transcript: str) -> tuple[EventList, str]:
    """Pass 1: every distinct news event the participant mentions."""
    return _run_with_retries(
        lambda feedback: build_identify_messages(transcript, retry_feedback=feedback),
        EventList,
        IDENTIFY_MAX_NEW_TOKENS,
        EventList(status="needs_review"),
    )


def assess_event(
    transcript: str, description: str, evidence_quote: str, candidates: pd.DataFrame
) -> tuple[EventAssessment, str]:
    """Pass 2: which candidate articles report this ONE event, and the detail given."""
    return _run_with_retries(
        lambda feedback: build_assess_messages(
            transcript, description, evidence_quote, candidates, retry_feedback=feedback
        ),
        EventAssessment,
        ASSESS_MAX_NEW_TOKENS,
        EventAssessment(status="needs_review"),
    )
