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
    """Load a plain text transcript file.

    Parameters
    ----------
    path : Path
        Path to .txt transcript file

    Returns
    -------
    str
        Raw transcript text, lowercased and stripped
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip().lower()


def extract_information_units(
    transcript: str,
    synonyms: dict = SYNONYMS,
) -> dict:
    """Extract information units from a transcript using synonym matching.

    For each IU, checks whether any synonym appears in the transcript.
    Returns a dict with:
    - present: bool — whether the IU was mentioned
    - evidence: list[str] — which synonyms were found

    Parameters
    ----------
    transcript : str
        Lowercased transcript text
    synonyms : dict
        Mapping of IU name to list of synonyms

    Returns
    -------
    dict
        {IU_name: {"present": bool, "evidence": [matched terms]}}
    """
    results = {}
    for iu, terms in synonyms.items():
        matched = []
        for term in terms:
            # Whole-word match to avoid e.g. "dish" matching "dishcloth"
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, transcript):
                matched.append(term)
        results[iu] = {
            "present": len(matched) > 0,
            "evidence": matched,
        }
    return results


def compute_iu_metrics(iu_results: dict) -> dict:
    """Compute summary metrics from IU extraction results.

    Parameters
    ----------
    iu_results : dict
        Output from extract_information_units

    Returns
    -------
    dict
        total_ius: total number of IUs in the scoring scheme
        mentioned_ius: number of IUs mentioned by participant
        iu_score: proportion of IUs mentioned (0-1)
    """
    total = len(iu_results)
    mentioned = sum(1 for v in iu_results.values() if v["present"])
    return {
        "total_ius": total,
        "mentioned_ius": mentioned,
        "iu_score": round(mentioned / total, 3) if total > 0 else 0,
    }