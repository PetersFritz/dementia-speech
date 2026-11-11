"""Cheap, code-only candidate retrieval for the recent-news (STN) question.

The cached event list (data/processed/news_events_cache.csv, built by
pipelines/build_news_event_cache.py) covers every day in a lookback window
before every participant's assessment date -- too many rows (Guardian alone
returns up to 50/day) to hand the LLM wholesale for one participant without
either truncating badly or burning an enormous prompt. This module narrows
that down to a short candidate list *before* the model ever sees it: a plain
keyword-overlap score between the participant's answer and each event's
headline+summary, keeping only the top matches.

This is deliberately simple (set overlap on stopword-filtered lowercase
tokens, no embeddings/extra ML dependency) and deliberately NOT the final
word on whether an answer is correct -- that judgement is still the model's
(extract.py) and the date-window logic is still pure code (scoring.py). This
step only decides which events are worth the model's attention. Its main
failure mode is a real, correct answer phrased with none of the cached
headline's/summary's own key terms, which would wrongly fail to surface the
matching event and read as "unverifiable" rather than correct -- exactly the
kind of error the hand-rating pass in docs/news_scoring_rules.md should
catch before this measure ships.
"""
import re
from datetime import date, timedelta

import pandas as pd

# Wide enough to also catch a real but "outdated" answer (see scoring.py's
# WINDOW_DAYS for the much narrower "recent enough" cutoff) -- an
# interview-day description of something from a month or two back should
# still be matchable to a real cached event, just not one that counts as
# "recent" once scoring.py applies the date check.
CANDIDATE_WINDOW_DAYS = 60
TOP_K_CANDIDATES = 20

# Minimal stopword list -- just enough to stop near-every-event overlap on
# filler words ("the", "a government minister said...") from swamping the
# genuine content-word signal. Not trying to be a complete stopword list.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "there", "about", "into",
    "over", "after", "before", "has", "have", "had", "will", "would", "can",
    "could", "said", "says", "say", "news", "think", "something", "someone",
    "thing", "things", "maybe", "well", "just", "like", "know", "don't",
    "didn't", "really", "yeah", "um", "er", "uh",
}

_WORD_RE = re.compile(r"[a-z']+")


def _tokenize(text: str) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def select_candidates(
    answer_text: str,
    events_df: pd.DataFrame,
    assessment_date: date,
    window_days: int = CANDIDATE_WINDOW_DAYS,
    top_k: int = TOP_K_CANDIDATES,
) -> pd.DataFrame:
    """Top `top_k` cached events (by keyword overlap with `answer_text`) from
    the `window_days` before (and including) `assessment_date`.

    Returns a DataFrame with columns date/source/category/headline/summary/url
    (events_df's own columns), sorted most-recent-first, index reset to
    0..n-1 -- callers (prompt.py) number candidates 1..n off this order, and
    scoring.py reads a candidate back out by that same 1-indexed position.
    Empty if nothing falls in the window (e.g. cache not yet backfilled for
    this participant's dates -- see build_news_event_cache.py).
    """
    dates = pd.to_datetime(events_df["date"], errors="coerce").dt.date
    window_start = assessment_date - timedelta(days=window_days)
    in_window = events_df[(dates >= window_start) & (dates <= assessment_date)]
    if in_window.empty:
        return in_window.reset_index(drop=True)

    answer_tokens = _tokenize(answer_text)
    event_tokens = (in_window["headline"].fillna("") + " " + in_window["summary"].fillna("")).map(_tokenize)
    overlap = event_tokens.map(lambda toks: len(answer_tokens & toks))

    ranked = (
        in_window.assign(_overlap=overlap, _date=dates.loc[in_window.index])
        .sort_values(["_overlap", "_date"], ascending=[False, False])
        .head(top_k)
        .sort_values("_date", ascending=False)
        .drop(columns=["_overlap", "_date"])
        .reset_index(drop=True)
    )
    return ranked
