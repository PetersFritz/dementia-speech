"""Build a cached list of real-world news events for the recording-date windows this
study needs, so pipelines/score_news.py (not written yet) can later check whether a
participant's "recent news" (STN) answer matches something that was actually in the
news shortly before their recording date.

Two sources, per the Tier 2 TODO (docs/to_do/TODO_dashboard.md):
  - Wikipedia's "Portal:Current events" daily pages -- no API key, public MediaWiki API.
  - The Guardian's UK/world/politics sections -- Guardian Open Platform API. Free, but
    needs a personal key: sign up at https://open-platform.theguardian.com/access/
    (instant, just an email) and set it as GUARDIAN_API_KEY before running this.

Needs outbound internet access to both APIs. HPC compute nodes allocated via sbatch
typically don't have it (that's why this isn't an sbatch script, unlike the extract_*
pipelines) -- run it from a login node or a machine that has both internet and the data
dir mounted. It's lightweight I/O, not compute, but rate-limited (see
REQUEST_SLEEP_SECONDS) so a full run over a year of dates can take a while; consider
nohup/tmux rather than a plain foreground run on a login node you might get disconnected
from. Safe to re-run: each (source, day) is cached to its own file under
data/raw/news_cache/ and skipped if it already exists, so an interrupted run just picks
up where it left off and you never pay for the same day twice.

Reads data/raw/Process_demographics.csv's `assessment_date` column (added by
pipelines/scratch/add_assessment_dates.py) to work out which calendar days actually need
fetching: every day in the WINDOW_DAYS-day lookback before each unique assessment date --
not just the date itself, since score_news.py's "recent" window (also WINDOW_DAYS) means
a participant could correctly cite anything in that whole span, not only same-day news.
WINDOW_DAYS=7 follows GPCOG's own recent-news item, which only looks at the preceding
week -- deliberately narrower than the TODO's original 42-day placeholder, and the reason
this cache stays small enough to build in one sitting even on the Guardian free tier.

Writes one JSON file per (source, day) under data/raw/news_cache/{wikipedia,guardian}/,
plus a combined flat CSV, data/processed/news_events_cache.csv
(date, source, category, headline, summary, url), for score_news.py to read. `summary`
gives the later LLM matcher more than a bare headline to check a participant's answer
against: Wikipedia's `headline` is already a full event sentence (so `summary` just
repeats it there), but a Guardian headline alone (e.g. "Starmer defends welfare reforms")
is too thin on its own -- `summary` there is the Guardian's own short standfirst
(`trailText`), not the full article body (too large to cache cheaply, and a short
standfirst sidesteps any full-text redistribution question that pulling body text would
raise).
"""
import html
import json
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

project_root = Path(__file__).parent.parent
os.chdir(project_root)

from dementia_speech.paths import DATA_DIR

DEMOGRAPHICS_PATH = DATA_DIR / "raw/Process_demographics.csv"
CACHE_DIR = DATA_DIR / "raw/news_cache"
OUTPUT_PATH = DATA_DIR / "processed/news_events_cache.csv"

WINDOW_DAYS = 7  # matches GPCOG's recent-news item -- see module docstring
REQUEST_SLEEP_SECONDS = 1.1  # stay under the Guardian free tier's 1 req/sec limit

GUARDIAN_API_KEY = os.environ.get("GUARDIAN_API_KEY")
GUARDIAN_SECTIONS = "uk-news|world|politics"  # Guardian API OR syntax for `section`

MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 5  # doubles each retry: 5, 10, 20, 40, 80s


def _get_with_retry(url: str, **kwargs) -> requests.Response:
    """requests.get with exponential backoff on 429 -- both APIs can throttle a shared
    university NAT IP well below what our own REQUEST_SLEEP_SECONDS pacing alone assumes,
    so a plain raise_for_status() on the first 429 was cascading into an immediate,
    unthrottled retry of the next day (see cached_fetch) rather than backing off.
    """
    for attempt in range(MAX_RETRIES + 1):
        response = requests.get(url, **kwargs)
        if response.status_code != 429:
            response.raise_for_status()
            return response
        wait = float(response.headers.get("Retry-After", BACKOFF_BASE_SECONDS * 2 ** attempt))
        print(f"    429, waiting {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})")
        time.sleep(wait)
    response.raise_for_status()  # retries exhausted -- raise the last 429 as HTTPError
    return response


def needed_days() -> list[date]:
    """Every day covered by some participant's WINDOW_DAYS-day lookback, deduplicated."""
    demographics = pd.read_csv(DEMOGRAPHICS_PATH)
    if "assessment_date" not in demographics.columns:
        raise SystemExit(
            f"No 'assessment_date' column in {DEMOGRAPHICS_PATH} -- "
            "run pipelines/scratch/add_assessment_dates.py first."
        )
    assessment_dates = pd.to_datetime(demographics["assessment_date"], errors="coerce").dropna()
    if assessment_dates.empty:
        raise SystemExit("No non-missing assessment_date values found.")

    days = set()
    for assessment_date in assessment_dates.dt.date.unique():
        for offset in range(WINDOW_DAYS + 1):
            days.add(assessment_date - timedelta(days=offset))
    return sorted(days)


# ── Wikipedia: Portal:Current events/<Year>_<Month>_<Day> ───────────────────────────

_WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikipedia's API rejects requests with no/generic User-Agent (403) -- needs something
# descriptive per https://meta.wikimedia.org/wiki/User-Agent_policy, contact info optional
# but recommended if this gets reused outside a one-off research cache build.
_HEADERS = {"User-Agent": "dementia-speech-research/1.0 (University of Sheffield)"}
_EXTERNAL_LINK_RE = re.compile(r"\[(https?://\S+)\s+([^\]]*)\]")
_PIPED_WIKILINK_RE = re.compile(r"\[\[[^|\]]+\|([^\]]+)\]\]")
_PLAIN_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def _clean_wikitext(text: str) -> str:
    text = _EXTERNAL_LINK_RE.sub("", text)
    text = _PIPED_WIKILINK_RE.sub(r"\1", text)
    text = _PLAIN_WIKILINK_RE.sub(r"\1", text)
    return text.replace("'''", "").replace("''", "").strip()


def _parse_wikipedia_wikitext(wikitext: str) -> list[dict]:
    """One row per leaf bullet that cites an external source -- the parent bullets in a
    nested chain (e.g. "*[[War]]" / "**[[Battle]]") are just topic breadcrumbs with no
    sentence of their own, so bullets with no external link are skipped rather than
    included as contentless events.
    """
    events = []
    category = None
    for line in wikitext.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("'''") and line.endswith("'''"):
            category = _clean_wikitext(line)
            continue
        if not line.startswith("*"):
            continue
        link_match = _EXTERNAL_LINK_RE.search(line)
        if not link_match:
            continue  # topic-only bullet, no actual event sentence
        url = link_match.group(1)
        headline = _clean_wikitext(line.lstrip("*").strip())
        if headline:
            # `summary` mirrors `headline` here -- the wikitext bullet already is a full
            # event sentence, unlike the Guardian's short `webTitle` (see fetch_guardian_day).
            events.append({"category": category, "headline": headline, "summary": headline, "url": url})
    return events


def fetch_wikipedia_day(day: date) -> list[dict]:
    page_title = f"Portal:Current_events/{day.year}_{day.strftime('%B')}_{day.day}"
    response = _get_with_retry(
        _WIKI_API,
        params={"action": "parse", "page": page_title, "format": "json", "prop": "wikitext"},
        headers=_HEADERS,
        timeout=30,
    )
    payload = response.json()
    if "error" in payload:
        return []  # no portal page for this day (shouldn't normally happen post-2010s)
    wikitext = payload["parse"]["wikitext"]["*"]
    return _parse_wikipedia_wikitext(wikitext)


# ── The Guardian: content search API, UK/world/politics sections ───────────────────

_GUARDIAN_API = "https://content.guardianapis.com/search"
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_guardian_html(text: str) -> str:
    return html.unescape(_HTML_TAG_RE.sub("", text or "")).strip()


def fetch_guardian_day(day: date) -> list[dict]:
    if not GUARDIAN_API_KEY:
        raise SystemExit(
            "GUARDIAN_API_KEY is not set. Sign up for a free key at "
            "https://open-platform.theguardian.com/access/ and export it."
        )
    iso_day = day.isoformat()
    response = _get_with_retry(
        _GUARDIAN_API,
        params={
            "from-date": iso_day,
            "to-date": iso_day,
            "section": GUARDIAN_SECTIONS,
            "page-size": 50,
            "order-by": "newest",
            "show-fields": "trailText",  # short standfirst, not the full article body
            "api-key": GUARDIAN_API_KEY,
        },
        timeout=30,
    )
    results = response.json()["response"]["results"]
    return [
        {
            "category": r["sectionName"],
            "headline": r["webTitle"],
            "summary": _clean_guardian_html(r.get("fields", {}).get("trailText", "")),
            "url": r["webUrl"],
        }
        for r in results
    ]


# ── Caching + combining ──────────────────────────────────────────────────────────────

def cached_fetch(source: str, day: date, fetch_fn) -> list[dict]:
    cache_path = CACHE_DIR / source / f"{day.isoformat()}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    try:
        events = fetch_fn(day)
    finally:
        # Always pace ourselves, even on failure -- otherwise an error on one day falls
        # straight through to an unthrottled request for the next, see _get_with_retry.
        time.sleep(REQUEST_SLEEP_SECONDS)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(events, indent=2))
    return events


def main():
    days = needed_days()
    print(f"{len(days)} unique days needed, {days[0]} to {days[-1]}")

    rows = []
    guardian_quota_exhausted = False
    for i, day in enumerate(days, 1):
        for source, fetch_fn in [("wikipedia", fetch_wikipedia_day), ("guardian", fetch_guardian_day)]:
            if source == "guardian" and guardian_quota_exhausted:
                continue  # don't retry a day we already know will fail -- see below
            try:
                events = cached_fetch(source, day, fetch_fn)
            except requests.exceptions.HTTPError as e:
                # A 429 that survives _get_with_retry's backoff (~2.5 min total) is almost
                # certainly the Guardian free tier's 500-calls/day quota, not a transient
                # per-second throttle -- those resolve well within that window. Stop
                # spending ~2.5 min per remaining day confirming the same thing again;
                # Wikipedia (no daily cap) keeps going. Rerun tomorrow for the rest.
                if source == "guardian" and e.response is not None and e.response.status_code == 429:
                    print(f"  Guardian quota likely exhausted ({e}) -- skipping remaining "
                          "Guardian days this run, rerun tomorrow to fetch the rest.")
                    guardian_quota_exhausted = True
                else:
                    print(f"  ERROR fetching {source} for {day}: {e}")
                continue
            except Exception as e:
                print(f"  ERROR fetching {source} for {day}: {e}")
                continue
            for event in events:
                rows.append({"date": day.isoformat(), "source": source, **event})

        if i % 20 == 0 or i == len(days):
            print(f"  {i}/{len(days)} days done ({len(rows)} events so far)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(rows)} events to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
