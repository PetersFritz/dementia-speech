"""Information unit extraction for Cookie Theft picture description task."""

import re
from pathlib import Path

from dementia_speech.tasks.picture_description.registry import SYNONYMS

WORD_PATTERN = re.compile(r"[a-z']+")

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


def tokenize_words(transcript: str) -> list:
    """Split a transcript into lowercase word tokens (letters/apostrophes only).

    Shared by the efficiency and lexical-semantic measures so word counts
    are computed identically wherever they're used.
    """
    return WORD_PATTERN.findall(transcript.lower())


def compute_efficiency_metrics(transcript: str, iu_results: dict) -> dict:
    """Compute IU informativeness relative to verbal output.

    Coverage (`iu_score`, from `compute_iu_metrics`) counts *which* IUs were
    mentioned but says nothing about how much talking it took to mention
    them. Two participants can reach the same coverage while one needed
    twice as many words — a pattern of verbose, low-yield speech associated
    with impaired discourse (Nicholas & Brookshire, 1993, "A system for
    quantifying the informativeness and efficiency of connected speech");
    this is the CIU "efficiency" half of their coverage/efficiency pair
    (their original efficiency measure is CIUs-per-minute, using recording
    duration — unavailable here since these transcripts carry no timing, so
    this uses CIUs-per-word instead).

    Returns
    -------
    dict
        total_words, iu_mentions_per_100_words (total IU mentions per 100
        words spoken — higher means more of the participant's speech landed
        on-target; lower means more circumlocution/repetition/off-topic
        content per IU mentioned).
    """
    total_words = len(tokenize_words(transcript))
    total_mentions = sum(v["count"] for v in iu_results.values())
    return {
        "total_words": total_words,
        "iu_mentions_per_100_words": (
            round(100 * total_mentions / total_words, 2) if total_words > 0 else None
        ),
    }