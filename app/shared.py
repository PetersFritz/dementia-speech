"""Shared data loading and constants for the dashboard."""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# ── Phonemic fluency data (Leslie code scoring, PFT_new) ─────────────────────
pft_words_df   = pd.read_csv(DATA_DIR / "pft_words_new.csv")
pft_metrics_df = pd.read_csv(DATA_DIR / "pft_metrics_new.csv")

# ── Cookie Theft data ─────────────────────────────────────────────────────────
ct_iu_df      = pd.read_csv(DATA_DIR / "cookie_theft_iu.csv")
ct_metrics_df = pd.read_csv(DATA_DIR / "cookie_theft_metrics.csv")

# Participant list — union across tasks, sorted
_pft_ids = set(pft_words_df["participant_id"].astype(str))
_ct_ids  = set(ct_metrics_df["participant_id"].astype(str))
PARTICIPANTS = sorted(_pft_ids | _ct_ids)

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