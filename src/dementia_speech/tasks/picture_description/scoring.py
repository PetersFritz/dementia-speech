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

    Tracks position of first match so IUs can be ordered by appearance.

    Parameters
    ----------
    transcript : str
        Lowercased transcript text
    synonyms : dict
        Mapping of IU name to list of synonyms

    Returns
    -------
    dict
        {IU_name: {"present": bool, "evidence": [matched terms], "position": int}}
        position is the character index of the first match in the transcript,
        or None if not present.
    """
    results = {}
    for iu, terms in synonyms.items():
        matched = []
        first_position = None

        for term in terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            match = re.search(pattern, transcript)
            if match:
                matched.append(term)
                # Track earliest position across all synonyms
                if first_position is None or match.start() < first_position:
                    first_position = match.start()

        results[iu] = {
            "present":  len(matched) > 0,
            "evidence": matched,
            "position": first_position,
        }
    return results

def get_iu_sequence(iu_results: dict) -> list[str]:
    """Return IUs in the order they first appear in the transcript.

    Parameters
    ----------
    iu_results : dict
        Output from extract_information_units

    Returns
    -------
    list[str]
        IU names ordered by first appearance position
    """
    mentioned = [
        (iu, data["position"])
        for iu, data in iu_results.items()
        if data["present"]
    ]
    mentioned.sort(key=lambda x: x[1])
    return [iu for iu, _ in mentioned]


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