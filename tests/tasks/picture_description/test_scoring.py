"""Tests for Semantic Relevance (SemR, Stegmann et al. 2022) -- scoring.compute_semantic_relevance.

Focus is specifically on the two ways the numerator could silently double-count:
repeated mentions of the same IU, and two different synonyms for the same IU. The
pure division itself (compute_semantic_relevance) is also tested directly, since it's
what score_cookie_theft.py calls with already-computed counts.
"""
from dementia_speech.tasks.picture_description.scoring import (
    compute_iu_metrics,
    compute_semantic_relevance,
    compute_iu_mention_density,
    extract_information_units,
    tokenize_words,
)


def test_basic_ratio():
    assert compute_semantic_relevance(mentioned_ius=12, total_words=74) == 12 / 74


def test_full_precision_not_rounded():
    # Deliberately not 0.162 (the spec's rounded example) -- SemR is stored at full
    # precision, unlike iu_score's round(..., 3) in compute_iu_metrics.
    result = compute_semantic_relevance(mentioned_ius=12, total_words=74)
    assert result == 0.16216216216216217


def test_no_words_returns_none():
    assert compute_semantic_relevance(mentioned_ius=0, total_words=0) is None


def test_repeated_mentions_of_one_iu_count_once():
    transcript = "the boy the boy the boy is on the stool"
    iu_results = extract_information_units(transcript)
    metrics = compute_iu_metrics(iu_results)

    assert iu_results["Boy"]["count"] == 3  # every occurrence, for the graph/cycle metrics
    assert metrics["mentioned_ius"] == 2  # Boy + Stool, each counted once

    total_words = len(tokenize_words(transcript))
    semr = compute_semantic_relevance(metrics["mentioned_ius"], total_words)
    assert semr == 2 / total_words


def test_synonyms_of_same_iu_collapse_to_one():
    # "woman" and "mother" are both synonyms for the Woman IU.
    transcript = "the woman is the mother of the children"
    iu_results = extract_information_units(transcript)
    metrics = compute_iu_metrics(iu_results)

    assert iu_results["Woman"]["present"] is True
    assert iu_results["Woman"]["count"] == 2  # both synonym occurrences
    assert metrics["mentioned_ius"] == 1  # still one unique picture element

    total_words = len(tokenize_words(transcript))
    semr = compute_semantic_relevance(metrics["mentioned_ius"], total_words)
    assert semr == 1 / total_words


def test_multiple_ius_each_counted_once_regardless_of_repeats():
    transcript = "boy boy boy cookie cookie jar jar jar jar"
    iu_results = extract_information_units(transcript)
    metrics = compute_iu_metrics(iu_results)

    assert metrics["mentioned_ius"] == 3  # Boy, Cookie, Jar
    assert metrics["total_mentions"] == 9  # 3 + 2 + 4 -- the repeat-aware count, unused by SemR

    total_words = len(tokenize_words(transcript))
    semr = compute_semantic_relevance(metrics["mentioned_ius"], total_words)
    assert semr == 3 / total_words


def test_empty_transcript_returns_none():
    transcript = ""
    iu_results = extract_information_units(transcript)
    metrics = compute_iu_metrics(iu_results)
    total_words = len(tokenize_words(transcript))

    assert total_words == 0
    assert compute_semantic_relevance(metrics["mentioned_ius"], total_words) is None


def test_iu_mention_density_counts_repeats_unlike_semantic_relevance():
    # Unlike compute_semantic_relevance, repeats DO count here -- that's the
    # whole point of this being a separate measure.
    transcript = "boy boy boy cookie cookie jar jar jar jar"
    iu_results = extract_information_units(transcript)
    metrics = compute_iu_metrics(iu_results)
    total_words = len(tokenize_words(transcript))

    assert metrics["mentioned_ius"] == 3
    assert metrics["total_mentions"] == 9

    semr = compute_semantic_relevance(metrics["mentioned_ius"], total_words)
    density = compute_iu_mention_density(metrics["total_mentions"], total_words)

    assert semr == 3 / total_words
    assert density == 9 / total_words
    assert density != semr


def test_iu_mention_density_no_words_returns_none():
    assert compute_iu_mention_density(total_mentions=0, total_words=0) is None
