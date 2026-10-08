"""Run memory-complaint extraction over a transcript with a local Qwen model.

Loading the model requires torch/transformers and a GPU -- only import this module
from code that's actually running on a GPU node (e.g. pipelines/score_memory_complaints.py),
not from the dashboard/scoring side of the project.
"""
import json
from functools import lru_cache

from pydantic import ValidationError

from dementia_speech.tasks.memory_complaints.prompt import build_messages
from dementia_speech.tasks.memory_complaints.schema import ComplaintResult

MODEL_PATH = "/mnt/parscratch/users/acp23fp/dementia-speech/models/Qwen2.5-14B-Instruct"
MAX_RETRIES = 3
MAX_NEW_TOKENS = 512


@lru_cache(maxsize=1)
def _load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto"
    )
    return tokenizer, model


def _generate(transcript: str, retry_feedback: str | None) -> str:
    tokenizer, model = _load_model()
    messages = build_messages(transcript, retry_feedback=retry_feedback)
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    output = model.generate(
        **encoded,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    completion = output[0][encoded["input_ids"].shape[-1]:]
    return tokenizer.decode(completion, skip_special_tokens=True)


def _parse(raw: str) -> ComplaintResult:
    text = raw.strip()
    if text.startswith("```"):
        # Strip a markdown code fence if the model wrapped the JSON in one.
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    data = json.loads(text)
    return ComplaintResult.model_validate(data)


def extract(transcript: str) -> ComplaintResult:
    """Classify one transcript, retrying on a malformed completion.

    Falls back to status='needs_review' with no findings after MAX_RETRIES failed
    attempts, rather than raising -- so a batch run over hundreds of participants
    never aborts on one bad completion (see pipelines/score_memory_complaints.py).
    """
    feedback = None
    for _ in range(MAX_RETRIES):
        raw = _generate(transcript, retry_feedback=feedback)
        try:
            return _parse(raw)
        except (json.JSONDecodeError, ValidationError) as e:
            feedback = (
                f"Your last response could not be parsed ({e}). "
                "Return ONLY the JSON object, with no other text."
            )
    return ComplaintResult(status="needs_review", findings=[])
