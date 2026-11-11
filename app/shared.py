"""Shared data loading and constants for the dashboard."""
import json

import pandas as pd

from dementia_speech.paths import DATA_DIR as _DATA_ROOT

DATA_DIR = _DATA_ROOT / "processed"

# ── Phonemic fluency data (Leslie code scoring, PFT_new) ─────────────────────
pft_words_df   = pd.read_csv(DATA_DIR / "pft_words_new.csv")
pft_metrics_df = pd.read_csv(DATA_DIR / "pft_metrics_new.csv")

# ── Semantic/category fluency data (SFT_new) ─────────────────────────────────
sft_words_df   = pd.read_csv(DATA_DIR / "sft_words.csv")
sft_metrics_df = pd.read_csv(DATA_DIR / "sft_metrics.csv")

# ── Cookie Theft data ─────────────────────────────────────────────────────────
ct_iu_df      = pd.read_csv(DATA_DIR / "cookie_theft_iu.csv")
ct_metrics_df = pd.read_csv(DATA_DIR / "cookie_theft_metrics.csv")
# Manually-transcribed CTD transcripts, for the Transcript tab's highlighted
# text -- same directory (and the same "manual, not Whisper ASR" reasoning)
# as score_cookie_theft.py's TRANSCRIPT_DIR, so what's displayed always
# matches what the metrics were actually computed from. Was raw/text/CTD
# (Whisper ASR output) until 2026-11-10, left stale when scoring switched to
# the manual transcripts -- the mismatch this caused is exactly what a
# participant's displayed transcript disagreeing with their own measures
# looks like.
CTD_TRANSCRIPT_DIR = DATA_DIR.parent / "raw" / "process" / "CTD"


def _index_ctd_transcripts(directory):
    """Map participant_id -> transcript path, using the exact same filename
    convention score_cookie_theft.py's extract_participant_id() uses (split
    the stem on "__", take the 2nd segment), normalized to a bare numeric
    string to match ct_metrics_df's own participant_id convention (CSV
    round-tripping already strips any zero-padding there -- see
    _lookup_demographics's re-padding below).

    This exists because the dashboard used to find a participant's
    transcript with `directory.glob(f"*{pid}*.txt")` -- an unanchored
    substring match that silently returns the *wrong* file whenever one
    participant's id is a substring of another's filename (e.g. id "2"
    matches "002", "012", "021", "102", "200", ... just as happily as the
    actual file for participant 2), which is exactly what made one
    participant's displayed transcript disagree with their own computed
    measures.
    """
    index = {}
    if not directory.exists():
        return index
    for path in directory.glob("*.txt"):
        parts = path.stem.split("__")
        raw_id = parts[1] if len(parts) >= 2 else path.stem
        try:
            pid = str(int(raw_id))
        except ValueError:
            pid = raw_id
        index[pid] = path
    return index


CTD_TRANSCRIPT_PATHS = _index_ctd_transcripts(CTD_TRANSCRIPT_DIR)

# Participant list — union across tasks, sorted
_pft_ids = set(pft_words_df["participant_id"].astype(str))
_sft_ids = set(sft_metrics_df["participant_id"].astype(str))
_ct_ids  = set(ct_metrics_df["participant_id"].astype(str))
PARTICIPANTS = sorted(_pft_ids | _sft_ids | _ct_ids)

# ── Demographics (meta-info.csv) ──────────────────────────────────────────────
# participant_id here is the bare numeric id used throughout the processed
# data (e.g. "1"), which came from a zero-padded token in the original
# filename (e.g. "001"); meta-info.csv still keys on the full raw IDs string
# (e.g. "PROCESS-2_rec__001"), so re-pad and match the same way the scoring
# pipelines do (pipelines/score_pft_new.py::extract_participant_id).
_metadata_df = pd.read_csv(DATA_DIR.parent / "raw" / "meta-info.csv")

def _lookup_demographics(participant_id):
    key = f"{int(participant_id):03d}"
    match = _metadata_df[_metadata_df["IDs"].astype(str).str.contains(key)]
    if len(match) == 0:
        return {"age": None, "gender": None, "diagnosis": None}
    row = match.iloc[0]
    return {"age": row["age"], "gender": row["gender"], "diagnosis": row["diagnosis"]}

DEMOGRAPHICS = {pid: _lookup_demographics(pid) for pid in PARTICIPANTS}

def _lookup_split(participant_id):
    key = f"{int(participant_id):03d}"
    match = _metadata_df[_metadata_df["IDs"].astype(str).str.contains(key)]
    return match.iloc[0]["Split"] if len(match) else None

# TRAIN/TEST split -- used by SFT_new's banding reference (and the fluency
# discrepancy score) so the HC group a participant is scored against never
# includes that participant or a held-out test case. PFT/Cookie Theft don't
# make this distinction yet -- see the Tier 1 dashboard TODO.
SPLIT = {pid: _lookup_split(pid) for pid in PARTICIPANTS}

# ── Memory complaints (Q2 "memory-problems" LLM extraction) ──────────────────
# One row per finding; a participant with no findings (no_complaint/needs_review)
# still has exactly one row with domain/summary/evidence_quote all null.
_mem_complaints_df = pd.read_csv(DATA_DIR / "memory_complaints.csv")
_mem_complaints_df["participant_id"] = _mem_complaints_df["participant_id"].astype(str)

MEMORY_COMPLAINTS_BY_PARTICIPANT = {
    pid: group.to_dict("records")
    for pid, group in _mem_complaints_df.groupby("participant_id")
}

STATUS_COLOURS = {
    "complaint":     "#D85A30",
    "no_complaint":  "#1D9E75",
    "needs_review":  "#BA7517",
}
STATUS_LABELS = {
    "complaint":     "Complaint reported",
    "no_complaint":  "No complaint",
    "needs_review":  "Needs review",
}

# ── Finance (Q9 "finance" LLM extraction) ─────────────────────────────────────
# One row per participant (unlike memory complaints, no multi-finding groupby needed).
_finance_df = pd.read_csv(DATA_DIR / "finance.csv")
_finance_df["participant_id"] = _finance_df["participant_id"].astype(str)

FINANCE_BY_PARTICIPANT = {
    pid: row.to_dict()
    for pid, row in _finance_df.set_index("participant_id").iterrows()
}

FINANCE_STATUS_COLOURS = {
    "extracted":    "#1D9E75",
    "needs_review": "#BA7517",
}
FINANCE_STATUS_LABELS = {
    "extracted":    "Extracted",
    "needs_review": "Needs review",
}
FINANCE_MANAGER_LABELS = {
    "self":    "Participant manages",
    "other":   "Someone else manages",
    "shared":  "Shared management",
    "unclear": "Unclear",
}
FINANCE_MANAGER_COLOURS = {
    "self":    "#1D9E75",  # green  -- participant still manages themself
    "shared":  "#BA7517",  # yellow -- shared / partial
    "unclear": "#BA7517",  # yellow
    "other":   "#D85A30",  # orange -- someone else has taken over
}
FINANCE_CHANGED_LABELS = {
    "yes":     "Changed recently",
    "no":      "No recent change",
    "unclear": "Unclear",
}
FINANCE_CHANGED_COLOURS = {
    "no":      "#1D9E75",  # green  -- no change
    "unclear": "#BA7517",  # yellow
    "yes":     "#D85A30",  # orange -- a change has happened
}

# ── Who is concerned about memory (Q3 "who-concerned" LLM extraction) ────────
# One row per participant, same shape as finance.
_who_concerned_df = pd.read_csv(DATA_DIR / "who_concerned.csv")
_who_concerned_df["participant_id"] = _who_concerned_df["participant_id"].astype(str)

WHO_CONCERNED_BY_PARTICIPANT = {
    pid: row.to_dict()
    for pid, row in _who_concerned_df.set_index("participant_id").iterrows()
}

WHO_CONCERNED_STATUS_COLOURS = {
    "extracted":    "#1D9E75",
    "needs_review": "#BA7517",
}
WHO_CONCERNED_STATUS_LABELS = {
    "extracted":    "Extracted",
    "needs_review": "Needs review",
}
WHO_CONCERNED_LABELS = {
    "self":    "Participant themself",
    "other":   "Someone else",
    "both":    "Both",
    "unclear": "Unclear",
}
# Neutral, not valence-coded (unlike FINANCE_MANAGER_COLOURS/FINANCE_CHANGED_COLOURS) --
# whether it's clinically meaningful for someone *other* than the participant to be the
# one concerned hasn't been checked against this project's own HC/MCI/Dementia groups the
# way the headline measures have (see dementia_speech.interpretation), so this doesn't
# claim a good/bad direction the way a green/orange scheme would.
WHO_CONCERNED_COLOURS = {
    "self":    "#4A7FB5",
    "other":   "#7A6FB0",
    "both":    "#5B9AA9",
    "unclear": "#9e9e9e",
}

# ── Fluency discrepancy (letter z-score minus category z-score) ──────────────
# Experimental: single trial per task (vs. three summed in Wright et al. 2022)
# and calibrated against this project's own TRAIN-split HC group, not the
# paper's published norms -- see pipelines/score_fluency_discrepancy.py.
# Also: on this dataset the score doesn't show the group separation the
# paper reports -- see the fluency discrepancy discussion. Flag as
# experimental wherever it's shown.
_discrepancy_df = pd.read_csv(DATA_DIR / "fluency_discrepancy.csv")
_discrepancy_df["participant_id"] = _discrepancy_df["participant_id"].astype(str)

FLUENCY_DISCREPANCY_BY_PARTICIPANT = {
    pid: row.to_dict()
    for pid, row in _discrepancy_df.set_index("participant_id").iterrows()
}
# Whole-cohort frame (diagnosis + discrepancy_score for every participant),
# for the per-group comparison violin plot alongside the card (app.py) --
# same need as PFT_HEADLINE_Z/SFT_HEADLINE_Z/CT_HEADLINE_Z.
FLUENCY_DISCREPANCY_DF = _discrepancy_df

# ── Colours (Leslie code system: C, OTH, N, PN, R, SIM, F) ────────────────────
CLASSIFICATION_COLOURS = {
    "C":   "#2ecc71",
    "OTH": "#95a5a6",
    "N":   "#e74c3c",
    "PN":  "#3498db",
    "R":   "#e67e22",
    "SIM": "#9b59b6",
    "F":   "#bdc3c7",
}
CLASSIFICATION_LABELS = {
    "C":   "Correct",
    "OTH": "Other / off-target",
    "N":   "Nonword",
    "PN":  "Proper noun",
    "R":   "Repetition",
    "SIM": "Similar form/meaning",
    "F":   "Filled pause",
}
GROUP_COLOURS = {
    "HC":       "#1D9E75",
    "MCI":      "#BA7517",
    "Dementia": "#D85A30",
}

# Cycled per phonemic cluster id in the transcript (underline colour) --
# independent of CLASSIFICATION_COLOURS, only applied to 'C' words, so a
# singleton cluster (colour cycles back eventually on long trials) is fine.
CLUSTER_PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]


def parse_sequence(value):
    """Safely parse a JSON-encoded sequence list from the metrics CSV."""
    if not isinstance(value, str):
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []