"""Pure-code category/score logic for recent-news (STN) answers.

Deliberately separate from extract.py: the model (extract.py) only decides
WHAT the participant described and WHICH candidate(s) (if any) match, one
per distinct event they mentioned; whether each match counts as "recent
enough" is a plain date comparison, done here, not left to the model's own
judgement -- per docs/news_scoring_rules.md and the Tier 2 TODO's explicit
instruction that "date logic and final score are computed in code, not by
the model."

Categories, in descending order of credit (see docs/news_scoring_rules.md
for the full rubric and rationale):
  in_window_match -- at least one claim matches a candidate within
                     WINDOW_DAYS before the assessment date.              -> 3
  outdated        -- at least one claim matches a candidate, but none are
                     within WINDOW_DAYS (all older).                      -> 2
  vague            -- too generic to check against anything.              -> 1
  none_or_wrong    -- no real answer at all.                              -> 0
  unverifiable     -- specific answer, but no claim matched a candidate
                     (could be a real event the cache doesn't cover, or a
                     fabricated one -- can't tell which).              -> None (missing)
  no_date          -- no assessment_date for this participant, so no
                     candidate search was even possible.                -> None (missing)

An answer with several claims is scored on its BEST claim -- recalling one
genuinely recent real event is good evidence of intact recall even if the
same answer also includes something vague or unverifiable. All matched
events (not just the best one) are returned alongside the category/score
for the audit log, so that richer signal isn't silently thrown away.

unverifiable and no_date are explicitly NOT 0: per the Tier 2 TODO ("unsure
or inconsistent model output becomes missing, never zero"), an answer this
pipeline can't verify is a gap in the measure, not evidence the participant
got it wrong.

EXPLORATORY ADDITION: detail/substance metrics (compute_detail_metrics
below). A participant can correctly identify a recent real event
(category/score as above) while giving no actual substance about it --
fluent, confident-sounding speech with no real content, a pattern worth
flagging separately from whether the topic itself checks out (especially
relevant for MCI/dementia, where retrieval can be genuinely successful
while elaboration is thin). Still needs the same HC-group-difference check
as every other exploratory measure in this project before being shown as
more than descriptive -- see the Tier 2 TODO's equivalent note for the
life-story question's own "detail per 100 words" idea.
"""
from datetime import date
from typing import Literal

import pandas as pd

from dementia_speech.tasks.news.schema import NewsAnswerExtraction, NewsClaim

# Matches build_news_event_cache.py's own WINDOW_DAYS (GPCOG's recent-news
# item looks back 7 days) -- kept as a separate constant here, not imported,
# since pipeline scripts in this project aren't designed to import each
# other; see docs/news_scoring_rules.md if these ever need to diverge.
WINDOW_DAYS = 7

Category = Literal[
    "in_window_match", "outdated", "vague", "none_or_wrong", "unverifiable", "no_date",
]

SCORE_BY_CATEGORY: dict[Category, int | None] = {
    "in_window_match": 3,
    "outdated": 2,
    "vague": 1,
    "none_or_wrong": 0,
    "unverifiable": None,
    "no_date": None,
}

# Ranks a claim's own category for picking the "best" one among several --
# higher is better, matching SCORE_BY_CATEGORY's ordering but as a plain int
# so ties (e.g. two in-window matches) can be broken by detail count below.
_CLAIM_RANK = {"in_window_match": 2, "outdated": 1}


def _claim_category(event_date, assessment_date: date) -> Category:
    days_before = (assessment_date - event_date).days
    if 0 <= days_before <= WINDOW_DAYS:
        return "in_window_match"
    return "outdated"


def categorize(
    result: NewsAnswerExtraction,
    candidates: pd.DataFrame,
    assessment_date: date,
) -> tuple[Category, int | None, list[dict], NewsClaim | None]:
    """Final (category, score, matched_events, best_claim) for one
    participant's answer.

    matched_events is the matched row from `candidates` (date/source/
    category/headline/summary/url) for every claim that matched a
    candidate, as a list of dicts -- empty unless category is
    'in_window_match' or 'outdated'. best_claim is the single NewsClaim
    that decided the category (ties among equally-ranked claims broken by
    whichever has the most `details`, so detail reporting -- see
    compute_detail_metrics -- reflects the richest claim, not an arbitrary
    one); None if category is 'vague', 'none_or_wrong', or 'unverifiable'.
    """
    if result.status != "extracted":
        return "unverifiable", SCORE_BY_CATEGORY["unverifiable"], [], None

    if result.answer_quality == "none":
        return "none_or_wrong", SCORE_BY_CATEGORY["none_or_wrong"], [], None
    if result.answer_quality == "vague":
        return "vague", SCORE_BY_CATEGORY["vague"], [], None

    # answer_quality == "specific" from here on -- evaluate every claim,
    # keep every match, decide the category + representative claim from the
    # best-ranked (and, among ties, richest) one.
    matched_events = []
    best_category: Category = "unverifiable"
    best_claim: NewsClaim | None = None
    for claim in result.claims:
        n = claim.matched_candidate_number
        if n is None or not (1 <= n <= len(candidates)):
            continue
        matched = candidates.iloc[n - 1]
        event_date = pd.to_datetime(matched["date"]).date()
        claim_category = _claim_category(event_date, assessment_date)
        matched_events.append(matched.to_dict())

        is_better_rank = best_claim is None or _CLAIM_RANK[claim_category] > _CLAIM_RANK[best_category]
        is_tied_but_richer = (
            best_claim is not None
            and _CLAIM_RANK[claim_category] == _CLAIM_RANK[best_category]
            and len(claim.details) > len(best_claim.details)
        )
        if is_better_rank or is_tied_but_richer:
            best_category = claim_category
            best_claim = claim

    return best_category, SCORE_BY_CATEGORY[best_category], matched_events, best_claim


def compute_detail_metrics(result: NewsAnswerExtraction, best_claim: NewsClaim | None, word_count: int) -> dict:
    """Exploratory detail/substance metrics for one answer (see module
    docstring) -- plain counts, no model judgement involved.

    - detail_count: total concrete details across every claim in the
      answer (whether or not that claim matched a real cached event --
      elaboration is informative even when the topic itself can't be
      verified).
    - best_claim_detail_count: details specifically on the claim that
      decided the category/score (None if no claim did, e.g. vague/none/
      unverifiable) -- the direct answer to "they got the event right, but
      how much did they actually say about it?"
    - detail_rate_per_100_words: detail_count normalised by how much the
      participant said overall, so a long, fluent, low-substance answer
      doesn't score well just by rambling -- the exact pattern this metric
      is meant to catch.
    """
    detail_count = sum(len(c.details) for c in result.claims)
    return {
        "detail_count": detail_count,
        "best_claim_detail_count": len(best_claim.details) if best_claim is not None else None,
        "detail_rate_per_100_words": round(100 * detail_count / word_count, 2) if word_count else None,
    }
