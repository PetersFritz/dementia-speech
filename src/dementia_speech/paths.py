"""Single source of truth for where the project's data directory lives.

The data directory (raw transcripts, processed outputs, reference data)
is deliberately kept OUTSIDE this repo and outside Claude Code's working
directory, for data governance reasons -- patient transcript data should
not sit somewhere an AI coding assistant's file tools can read it. Every
script that needs it imports DATA_DIR from here instead of hardcoding a
relative "data/..." path, so the actual location is configured once, per
machine, via an environment variable rather than baked into the repo.

Set DEMENTIA_SPEECH_DATA_DIR in your shell profile to wherever the data/
directory actually lives on that machine, e.g.:
    export DEMENTIA_SPEECH_DATA_DIR=/mnt/xdrive/.../Fritz/dementia-speech-data
"""
import os
from pathlib import Path

_ENV_VAR = "DEMENTIA_SPEECH_DATA_DIR"


def _resolve_data_dir() -> Path:
    value = os.environ.get(_ENV_VAR)
    if not value:
        raise RuntimeError(
            f"{_ENV_VAR} is not set. The data/ directory lives outside this "
            "repo on purpose (data governance) -- set this environment "
            "variable to its location, e.g. in your shell profile:\n"
            f"    export {_ENV_VAR}=/path/to/dementia-speech-data"
        )
    path = Path(value).expanduser()
    if not path.is_dir():
        raise RuntimeError(
            f"{_ENV_VAR} is set to '{path}', but that directory doesn't exist."
        )
    return path


DATA_DIR = _resolve_data_dir()
