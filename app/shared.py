"""Shared data loading and constants for the dashboard."""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# ── Phonemic fluency data ─────────────────────────────────────────────────────
pft_words_df   = pd.read_csv(DATA_DIR / "pft_words.csv")
pft_metrics_df = pd.read_csv(DATA_DIR / "pft_metrics.csv")

# ── Cookie Theft data ─────────────────────────────────────────────────────────
ct_iu_df      = pd.read_csv(DATA_DIR / "cookie_theft_iu.csv")
ct_metrics_df = pd.read_csv(DATA_DIR / "cookie_theft_metrics.csv")

# Participant list — union across tasks, sorted
_pft_ids = set(pft_words_df["participant_id"].astype(str))
_ct_ids  = set(ct_metrics_df["participant_id"].astype(str))
PARTICIPANTS = sorted(_pft_ids | _ct_ids)

# ── Colours ───────────────────────────────────────────────────────────────────
CLASSIFICATION_COLOURS = {
    "correct":    "#2ecc71",
    "intrusion":  "#e74c3c",
    "repetition": "#e67e22",
    "variant":    "#95a5a6",
}
CLASSIFICATION_LABELS = {
    "correct":    "Correct",
    "intrusion":  "Intrusion",
    "repetition": "Repetition",
    "variant":    "Variant",
}
GROUP_COLOURS = {
    "HC":       "#1D9E75",
    "MCI":      "#BA7517",
    "Dementia": "#D85A30",
}


def parse_sequence(value):
    """Safely parse a JSON-encoded sequence list from the metrics CSV."""
    if not isinstance(value, str):
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []