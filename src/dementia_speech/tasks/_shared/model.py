"""Shared Qwen model loader for LLM-based extraction tasks.

Loading requires torch/transformers and a GPU -- only import this module from code
that's actually running on a GPU node, not from the dashboard/scoring side of the
project. @lru_cache means every task module that calls load_model() in the same
process shares the one loaded copy, rather than each loading its own 14B-parameter
model into GPU memory.
"""
from functools import lru_cache

MODEL_PATH = "/mnt/parscratch/users/acp23fp/dementia-speech/models/Qwen2.5-14B-Instruct"


@lru_cache(maxsize=1)
def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto"
    )
    return tokenizer, model
