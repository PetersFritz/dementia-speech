"""Tests for the event-based recent-news pipeline's model-free parts:
grounding/sanitising (events_verify) and date logic + answer summary
(events_scoring). The model passes themselves (events_extract) need a GPU and
aren't exercised here."""
from datetime import date

import pandas as pd

from dementia_speech.tasks.news.events_schema import EventAssessment, EventList, MentionedEvent
from dementia_speech.tasks.news.events_scoring import (
    WINDOW_DAYS, classify_event, summarize_answer,
)
from dementia_speech.tasks.news.events_verify import sanitize_assessment, sanitize_events

TRANSCRIPT = (
    "There was that big storm, I think they called it Bertha, that flooded a load of "
    "houses down south. And something about taxes too."
)
ASSESSMENT_DATE = date(2026, 10, 30)

CANDIDATES = pd.DataFrame([
    {"date": "2026-10-28", "source": "A", "headline": "Storm Bertha floods homes", "summary": "s", "url": "u1"},
    {"date": "2026-10-27", "source": "B", "headline": "Bertha hits the coast", "summary": "s", "url": "u2"},
    {"date": "2026-09-20", "source": "C", "headline": "Older flooding story", "summary": "s", "url": "u3"},
])


# ---- sanitize_events -------------------------------------------------------

def _events(*quotes):
    return EventList(
        status="extracted", answer_quality="specific",
        events=[MentionedEvent(description="d", evidence_quote=q) for q in quotes],
    )


def test_ungrounded_event_is_dropped():
    result = sanitize_events(_events("that big storm", "a total invention"), TRANSCRIPT)
    assert [e.evidence_quote for e in result.events] == ["that big storm"]


def test_duplicate_quote_is_dropped():
    result = sanitize_events(_events("that big storm", "that big storm"), TRANSCRIPT)
    assert len(result.events) == 1


def test_needs_review_passes_through():
    r = EventList(status="needs_review")
    assert sanitize_events(r, TRANSCRIPT) is r


# ---- sanitize_assessment ---------------------------------------------------

def _assessment(numbers, level, details):
    return EventAssessment(
        status="extracted", matched_candidate_numbers=numbers,
        detail_level=level, details=details,
    )


def test_out_of_range_and_duplicate_numbers_removed_order_kept():
    # The model's order (best match first) must survive: 2 stays ahead of 1.
    result = sanitize_assessment(_assessment([2, 2, 9, 0, 1], "none", []), TRANSCRIPT, 3)
    assert result.matched_candidate_numbers == [2, 1]


def test_no_grounded_details_forces_level_none():
    result = sanitize_assessment(_assessment([1], "rich", ["made up fact"]), TRANSCRIPT, 3)
    assert result.details == [] and result.detail_level == "none"


def test_details_without_level_get_a_level():
    result = sanitize_assessment(
        _assessment([1], "none", ["they called it Bertha", "flooded a load of houses"]), TRANSCRIPT, 3,
    )
    assert result.detail_level == "some"


# ---- classify_event --------------------------------------------------------

def test_recent_match_lists_all_articles_in_model_order():
    status, articles = classify_event([1, 2], CANDIDATES, ASSESSMENT_DATE)
    assert status == "recent_match"
    assert [a["source"] for a in articles] == ["A", "B"]
    assert all(a["in_window"] for a in articles)


def test_best_match_first_is_not_re_sorted_by_date():
    # Candidate 2 (27 Oct) is older than candidate 1 (28 Oct), but the model
    # ranked it best -- it must stay first.
    _, articles = classify_event([2, 1], CANDIDATES, ASSESSMENT_DATE)
    assert [a["source"] for a in articles] == ["B", "A"]


def test_one_recent_article_is_enough():
    status, articles = classify_event([3, 1], CANDIDATES, ASSESSMENT_DATE)
    assert status == "recent_match"
    assert [a["source"] for a in articles] == ["C", "A"]
    assert [a["in_window"] for a in articles] == [False, True]


def test_only_old_article_is_not_a_recent_match():
    # Candidates are normally drawn from inside the window only, but an
    # article outside it must never count as a recent recall.
    status, articles = classify_event([3], CANDIDATES, ASSESSMENT_DATE)
    assert status == "no_match"
    assert articles[0]["in_window"] is False


def test_no_numbers_is_no_match():
    assert classify_event([], CANDIDATES, ASSESSMENT_DATE) == ("no_match", [])


def test_window_boundaries():
    cands = pd.DataFrame([
        {"date": "2026-10-23", "source": "x", "headline": "h", "summary": "s", "url": "u"},   # 7 days before
        {"date": "2026-10-22", "source": "y", "headline": "h", "summary": "s", "url": "u"},   # 8 days before
        {"date": "2026-10-31", "source": "z", "headline": "h", "summary": "s", "url": "u"},   # after the assessment
    ])
    assert WINDOW_DAYS == 7
    assert classify_event([1], cands, ASSESSMENT_DATE)[0] == "recent_match"
    assert classify_event([2], cands, ASSESSMENT_DATE)[0] == "no_match"
    assert classify_event([3], cands, ASSESSMENT_DATE)[0] == "no_match"


# ---- summarize_answer ------------------------------------------------------

def test_summary_counts_and_best_recent_detail():
    rows = [
        {"match_status": "recent_match", "detail_level": "none"},
        {"match_status": "recent_match", "detail_level": "some"},
        {"match_status": "no_match", "detail_level": "rich"},
    ]
    s = summarize_answer("specific", rows)
    assert s["n_events"] == 3
    assert (s["n_recent_events"], s["n_unmatched_events"]) == (2, 1)
    assert s["recalled_recent"] is True
    # best detail is taken over RECENT matches only (the unmatched "rich" doesn't count)
    assert s["best_recent_detail"] == "some"


def test_summary_with_nothing_recent():
    s = summarize_answer("specific", [{"match_status": "no_match", "detail_level": "some"}])
    assert s["recalled_recent"] is False and s["best_recent_detail"] is None


def test_summary_with_no_events():
    s = summarize_answer("vague", [])
    assert s["n_events"] == 0 and s["recalled_recent"] is False
