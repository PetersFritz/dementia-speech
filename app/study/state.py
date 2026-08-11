"""Study-mode state: token-derived case sampling and response persistence.

A rater's 20-case set and case order are derived deterministically from their
token (via a seeded RNG) rather than being pre-shared with the consent Google
Form — the local server never needs to talk to Google. Progress is recovered
by reading the token's session file, so closing and reopening the same
emailed link resumes where the rater left off.

Persistence is one JSON file per token (data/study/sessions/<token>.json),
written via write-temp-then-atomic-rename, rather than a single shared
SQLite database. The deployment target for the study server isn't decided
yet and may end up on a network-mounted filesystem (as this dev checkout
currently is) — CIFS mounts here were confirmed not to support the file
locking SQLite requires, while atomic rename-into-place does not depend on
byte-range locking and works on both local disks and this network mount.
"""
import hashlib
import json
import os
import random
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from shared import ct_metrics_df

REPO_ROOT = Path(__file__).parent.parent.parent
META_PATH = REPO_ROOT / "data" / "raw" / "meta-info.csv"
SESSIONS_DIR = REPO_ROOT / "data" / "study" / "sessions"

N_CASES = 20
DIAGNOSES = ["HC", "MCI", "Dementia"]
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


def is_valid_token(token: str) -> bool:
    return bool(token) and bool(TOKEN_RE.match(token))


def _test_pool_by_diagnosis() -> dict[str, list[int]]:
    """TEST-split participant IDs with Cookie Theft data, grouped by diagnosis."""
    meta = pd.read_csv(META_PATH, dtype={"IDs": str})
    meta["participant_id"] = meta["IDs"].str.split("__").str[-1].astype(int)
    test_meta = meta[meta["Split"] == "TEST"]

    ct_ids = set(ct_metrics_df["participant_id"].astype(int))
    pool = test_meta[test_meta["participant_id"].isin(ct_ids)]

    return {
        dx: sorted(pool.loc[pool["diagnosis"] == dx, "participant_id"].tolist())
        for dx in DIAGNOSES
    }


def _proportional_allocation(pool_sizes: dict[str, int], n: int) -> dict[str, int]:
    """Largest-remainder allocation of n cases across groups, proportional to pool size."""
    total = sum(pool_sizes.values())
    raw = {k: pool_sizes[k] * n / total for k in pool_sizes}
    counts = {k: min(int(raw[k]), pool_sizes[k]) for k in pool_sizes}
    remainder = n - sum(counts.values())

    by_fraction = sorted(
        pool_sizes,
        key=lambda k: (-(raw[k] - int(raw[k])), DIAGNOSES.index(k)),
    )
    for dx in by_fraction:
        if remainder <= 0:
            break
        if counts[dx] < pool_sizes[dx]:
            counts[dx] += 1
            remainder -= 1

    return counts


def derive_case_list(token: str, n: int = N_CASES) -> list[int]:
    """Deterministically derive a rater's stratified, shuffled case list from their token."""
    if not is_valid_token(token):
        raise ValueError(f"invalid token: {token!r}")

    pool = _test_pool_by_diagnosis()
    counts = _proportional_allocation({dx: len(ids) for dx, ids in pool.items()}, n)

    seed = int(hashlib.sha256(token.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    selected: list[int] = []
    for dx in DIAGNOSES:
        selected.extend(rng.sample(pool[dx], counts[dx]))
    rng.shuffle(selected)
    return selected


def _session_path(token: str) -> Path:
    return SESSIONS_DIR / f"{token}.json"


def _load_session(token: str) -> dict | None:
    path = _session_path(token)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_session(token: str, data: dict) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=SESSIONS_DIR, prefix=f".{token}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp_path, _session_path(token))
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def get_or_create_session(token: str) -> list[int]:
    """Return the token's assigned case list, creating the session file on first visit."""
    data = _load_session(token)
    if data is not None:
        return data["assigned_cases"]

    cases = derive_case_list(token)
    _save_session(token, {
        "token": token,
        "assigned_cases": cases,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decisions": [],
        "questionnaire": None,
    })
    return cases


def record_decision(
    token: str, position: int, participant_id: int, diagnosis_guess: str, confidence: int
) -> None:
    if diagnosis_guess not in DIAGNOSES:
        raise ValueError(f"invalid diagnosis_guess: {diagnosis_guess!r}")
    if not (1 <= confidence <= 5):
        raise ValueError(f"invalid confidence: {confidence!r}")

    get_or_create_session(token)
    data = _load_session(token)
    if any(d["position"] == position for d in data["decisions"]):
        return  # duplicate submit for this position — ignore

    data["decisions"].append({
        "position": position,
        "participant_id": participant_id,
        "diagnosis_guess": diagnosis_guess,
        "confidence": confidence,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_session(token, data)


def get_progress(token: str) -> tuple[int, list[int]]:
    """Return (current_index, assigned_cases).

    current_index is the number of recorded decisions for this token — the
    0-based index of the next case to show, or len(assigned_cases) once the
    rater has completed all cases.
    """
    cases = get_or_create_session(token)
    data = _load_session(token)
    return len(data["decisions"]), cases


def record_questionnaire(token: str, answers: dict) -> None:
    get_or_create_session(token)
    data = _load_session(token)
    if data.get("questionnaire") is not None:
        return  # already submitted — ignore

    data["questionnaire"] = {
        "answers": answers,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_session(token, data)


def get_questionnaire(token: str) -> dict | None:
    data = _load_session(token)
    return data.get("questionnaire") if data else None
