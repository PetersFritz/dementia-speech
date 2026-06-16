"""Information unit extraction for Cookie Theft picture description task."""

import re 
from pathlib import Path

SYNONYMS = {
    "Boy": ["boy", "son", "brother", "male child"],
    "Girl": ["girl", "daughter", "sister", "female child"],
    "Woman": ["woman","mom", "mother", "lady", "parent", "female", "adult", "grownup", "mum"],
    "Kitchen": ["kitchen", "room"],
    "Cookie": ["cookie", "biscuit", "cake", "treat"],
    "Jar": ["jar", "container", "crock", "pot"],
    "Stool": ["stool", "seat", "chair", "ladder"],
    "Sink": ["sink", "basin", "washbasin", "washbowl", "washstand", "tap"],
    "Plate": ["plate"],
    "Dishcloth": ["dishcloth", "dishrag", "rag", "cloth", "napkin", "towel"],
    "Water": ["water", "dishwater", "liquid"],
    "Window": ["window", "frame", "glass"],
    "Cupboard": ["cupboard", "closet", "shelf"],
    "Dishes": ["dish", "dishes", "cup", "cups", "counter"],
    "Curtains": ["curtain", "curtains", "drape", "drapes", "drapery", "blind", "blinds", "screen", "screens"],
    "Exterior": ["exterior", "outside", "garden", "yard", "outdoors", "backyard", "driveway", "path", "tree", "bush"],
    "Steal": ["take", "steal", "taking", "stealing"],
    "Fall": ["fall", "falling", "slip", "slipping"],
    "Wash": ["wash", "dry", "clean", "washing", "drying", "cleaning"],
    "Overflow": ["overflow", "spill", "overflowing", "spilling"]
}

def load_transcript(path: Path) -> str:
    """Load a plain text transcript file, lowercased and stripped."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip().lower()
 
 
def extract_information_units(
    transcript: str,
    synonyms: dict = SYNONYMS,
) -> dict:
    """Extract information units from a transcript using synonym matching.
 
    Finds ALL mentions of each IU (every synonym occurrence), not just the
    first, so that repeat-aware sequence and cycle metrics are possible.
 
    Returns
    -------
    dict
        {IU_name: {
            "present": bool,
            "evidence": [matched terms],
            "positions": [character indices of every match, sorted],
            "count": number of mentions
        }}
    """
    results = {}
    for iu, terms in synonyms.items():
        matched_terms = []
        positions = []
 
        for term in terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            for match in re.finditer(pattern, transcript):
                positions.append(match.start())
                matched_terms.append(term)
 
        positions.sort()
        results[iu] = {
            "present": len(positions) > 0,
            "evidence": sorted(set(matched_terms)),
            "positions": positions,
            "count": len(positions),
        }
    return results
 
 
def get_iu_sequence_unique(iu_results: dict) -> list:
    """Return each mentioned IU once, ordered by first appearance.
 
    Use for: display, IU coverage, first-appearance ordering.
    """
    first_positions = [
        (iu, data["positions"][0])
        for iu, data in iu_results.items()
        if data["present"]
    ]
    first_positions.sort(key=lambda x: x[1])
    return [iu for iu, _ in first_positions]
 
 
def get_iu_sequence_full(iu_results: dict) -> list:
    """Return the full mention sequence with repeats, ordered by position.
 
    Use for: graph construction, cycle and self-cycle metrics, path distance.
 
    Every occurrence of every IU is included and the whole list is sorted
    by character position in the transcript, so the order reflects how the
    participant actually moved between concepts, including revisits.
    """
    mentions = []
    for iu, data in iu_results.items():
        for pos in data["positions"]:
            mentions.append((pos, iu))
    mentions.sort(key=lambda x: x[0])
    return [iu for _, iu in mentions]
 
 
def compute_iu_metrics(iu_results: dict) -> dict:
    """Compute summary coverage metrics from IU extraction results.
 
    Returns
    -------
    dict
        total_ius, mentioned_ius, iu_score, total_mentions
    """
    total = len(iu_results)
    mentioned = sum(1 for v in iu_results.values() if v["present"])
    total_mentions = sum(v["count"] for v in iu_results.values())
    return {
        "total_ius": total,
        "mentioned_ius": mentioned,
        "iu_score": round(mentioned / total, 3) if total > 0 else 0,
        "total_mentions": total_mentions,
    }