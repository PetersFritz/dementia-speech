"""Tests for lexical.compute_lexical_diversity -- plain type-token ratio,
added alongside the length-corrected compute_mattr.
"""
from dementia_speech.tasks.picture_description.lexical import (
    compute_lexical_diversity,
    compute_lexical_metrics,
)


def test_basic_ratio():
    # "the cat sat on the mat the cat ran": 9 words, 6 distinct types
    # (the, cat, sat, on, mat, ran).
    transcript = "the cat sat on the mat the cat ran"
    assert compute_lexical_diversity(transcript) == {"lexical_diversity": round(6 / 9, 3)}


def test_all_distinct_words_gives_one():
    assert compute_lexical_diversity("the cat sat on mat") == {"lexical_diversity": 1.0}


def test_empty_transcript_returns_none():
    assert compute_lexical_diversity("") == {"lexical_diversity": None}


def test_included_in_combined_metrics():
    metrics = compute_lexical_metrics("the cat sat on the mat")
    assert "lexical_diversity" in metrics
    assert "mattr" in metrics  # still present alongside it, not replaced
