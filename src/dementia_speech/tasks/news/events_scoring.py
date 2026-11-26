"""Date logic and answer-level summary for the event-based recent-news
pipeline (v2) -- pure code, no model.

The model only ever points at articles (pass 2); whether a matched article is
"recent" is plain date subtraction here, never a judgement call spent on the
model.

Only the WINDOW_DAYS before the assessment date matter (that is also all the
news cache holds -- see pipelines/build_news_event_cache.py), so there is no
"older event" category.

Per event (match_status):
  - "recent_match":  at least one matched article is within WINDOW_DAYS before
                     the assessment date -> the participant correctly recalled
                     a recent event.
  - "no_match":      no cached article reports it. NOT evidence the participant
                     was wrong: it may be an older event, or a real recent one
                     the cache doesn't cover (see docs/news_scoring_rules.md on
                     cache coverage).

Per answer (summarize_answer): whether anything recent was recalled, how many
events were mentioned/recalled, and the level of detail on the recalled ones.
"""
from datetime import date
from typing import Literal

import pandas as pd

from dementia_speech.tasks.news.events_schema import DetailLevel

# Same 7-day window as scoring.py's WINDOW_DAYS (GPCOG's own recent-news
# item asks about the preceding week).
WINDOW_DAYS = 7

MatchStatus = Literal["recent_match", "no_match"]

_DETAIL_RANK: dict[DetailLevel, int] = {"none": 0, "some": 1, "rich": 2}


def article_in_window(article_date, assessment_date: date) -> bool:
    days_before = (assessment_date - pd.to_datetime(article_date).date()).days
    return 0 <= days_before <= WINDOW_DAYS


def classify_event(
    matched_numbers: list[int], candidates: pd.DataFrame, assessment_date: date
) -> tuple[MatchStatus, list[dict]]:
    """(match_status, matched_articles) for one event.

    matched_articles is one dict per matched candidate (date/source/headline/
    summary/url, plus `in_window`), in the model's order -- BEST MATCH FIRST,
    so matched_articles[0] is the article the dashboard shows as what the
    participant's answer matches, with the rest as "also reported by". numbers
    must already be validated (events_verify).
    A matched article outside the window (shouldn't happen, since candidates
    are only drawn from inside it) is still listed but doesn't make the event
    a recent match.
    """
    articles = []
    for n in matched_numbers:
        row = candidates.iloc[n - 1].to_dict()
        row["in_window"] = article_in_window(row["date"], assessment_date)
        articles.append(row)

    if not articles:
        return "no_match", []
    if any(a["in_window"] for a in articles):
        return "recent_match", articles
    return "no_match", articles


def summarize_answer(answer_quality: str, event_rows: list[dict]) -> dict:
    """Answer-level headline numbers from the per-event rows.

    Each event row needs `match_status` and `detail_level`. best_recent_detail
    is the highest detail level among events that matched a recent article
    (None if there are none) -- "of the things they correctly recalled, how
    much did they actually know".
    """
    recent = [e for e in event_rows if e["match_status"] == "recent_match"]
    unmatched = [e for e in event_rows if e["match_status"] == "no_match"]
    best_recent_detail = (
        max((e["detail_level"] for e in recent), key=lambda lvl: _DETAIL_RANK[lvl])
        if recent else None
    )
    return {
        "answer_quality": answer_quality,
        "n_events": len(event_rows),
        "n_recent_events": len(recent),
        "n_unmatched_events": len(unmatched),
        "recalled_recent": len(recent) > 0,
        "best_recent_detail": best_recent_detail,
    }
