"""Event-based scoring of "what has been in the news recently?" (STN) transcripts.

Successor to score_news.py (which is left untouched, along with its output
files, so the two can be compared). Instead of one model call that matches the
whole answer against a keyword-retrieved candidate list and reduces it to a
single 0-3 score, this works event by event, in two model passes (see
dementia_speech.tasks.news.events_schema / events_prompt):

  1. identify_events: the model, prompted as a clinician, lists every distinct
     news event the participant mentions (no headlines shown).
  2. For each event: retrieve candidate articles FOR THAT EVENT, then ask which
     articles (if any) report it and how much concrete detail the participant
     gave. Whether a matched article falls in the last 7 days is plain date
     arithmetic (events_scoring.py), not the model's call. Only the 7 days
     before the assessment date are searched.

Needs a GPU (loads the local Qwen model) -- run through a slurm job, not on a
login node. The dashboard never calls the model; it reads the CSVs.

Outputs (all in data/processed/):
  news_events.csv   one row per event mentioned: what was said, whether it
                    matched an article from the last 7 days (recent_match /
                    no_match), which articles, and the level of detail.
  news_answers.csv  one row per participant: recalled anything recent?, how
                    many events, and the detail level on the recalled ones.
  news_events_audit.jsonl   raw completions, candidate lists, prompt version.

Requires the same inputs as score_news.py: data/raw/text/STN/*.txt,
data/raw/Process_demographics.csv's `assessment_date`, and
data/processed/news_events_cache.csv.
"""
import argparse
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
os.chdir(project_root)

from dementia_speech.paths import DATA_DIR
from dementia_speech.tasks._shared.model import MODEL_PATH
from dementia_speech.tasks.news.events_extract import assess_event, identify_events
from dementia_speech.tasks.news.events_prompt import PROMPT_VERSION
from dementia_speech.tasks.news.events_scoring import WINDOW_DAYS, classify_event, summarize_answer
from dementia_speech.tasks.news.events_verify import sanitize_assessment, sanitize_events
from dementia_speech.tasks.news.retrieval import select_candidates

TRANSCRIPT_DIR = DATA_DIR / "raw/text/STN"
DEMOGRAPHICS_PATH = DATA_DIR / "raw/Process_demographics.csv"
METADATA_PATH = DATA_DIR / "raw/meta-info.csv"
EVENTS_CACHE_PATH = DATA_DIR / "processed/news_events_cache.csv"
EVENTS_OUTPUT_PATH = DATA_DIR / "processed/news_events.csv"
ANSWERS_OUTPUT_PATH = DATA_DIR / "processed/news_answers.csv"
AUDIT_LOG_PATH = DATA_DIR / "processed/news_events_audit.jsonl"

# Columns kept on an events row even when there are no events, so the CSV
# always has the same header.
EVENT_COLUMNS = [
    "participant_id", "diagnosis", "assessment_date", "event_index",
    "description", "evidence_quote", "match_status", "n_matched_articles",
    "best_article_headline", "best_article_date", "best_article_source",
    "best_article_url", "matched_articles", "detail_level", "n_details", "details",
]


def extract_participant_id(filename) -> str:
    """<process_id>__STN.txt -> process_id. process_ID itself routinely
    contains '__' (e.g. 'PROCESS-2_rec__001'), and the task code is always the
    LAST '__'-separated segment -- see score_news.py's own docstring for the
    HPC bug that taking just the first segment caused."""
    return "__".join(filename.stem.split("__")[:-1])


def load_assessment_dates() -> dict[str, date]:
    # dtype=str on process_ID so leading zeros survive (see score_news.py).
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={"process_ID": str})
    parsed = pd.to_datetime(demographics["assessment_date"], errors="coerce")
    valid = parsed.notna()
    return {
        pid: parsed_date.date()
        for pid, parsed_date in zip(demographics.loc[valid, "process_ID"], parsed[valid])
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only score the first N transcripts (a quick trial before the full run). "
             "Overwrites the output files, so a full run afterwards replaces it.",
    )
    args = parser.parse_args()

    EVENTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(METADATA_PATH)
    assessment_dates = load_assessment_dates()
    events_cache = pd.read_csv(EVENTS_CACHE_PATH)

    transcript_files = sorted(TRANSCRIPT_DIR.glob("*.txt"))
    print(f"Found {len(transcript_files)} transcripts")
    if args.limit:
        transcript_files = transcript_files[: args.limit]
        print(f"--limit {args.limit}: scoring only the first {len(transcript_files)}")

    event_rows: list[dict] = []
    answer_rows: list[dict] = []

    # "w", not "a": the audit log always describes the same run as the CSVs.
    with AUDIT_LOG_PATH.open("w") as audit_log:
        for i, txt_path in enumerate(transcript_files, 1):
            participant_id = extract_participant_id(txt_path)
            transcript = txt_path.read_text().strip()

            match = metadata[metadata["IDs"].astype(str).str.contains(participant_id)]
            diagnosis = match.iloc[0]["diagnosis"] if len(match) > 0 else "Unknown"
            word_count = len(transcript.split())

            base = {"participant_id": participant_id, "diagnosis": diagnosis}
            answer_base = {**base, "source_file": txt_path.name, "word_count": word_count}

            assessment_date = assessment_dates.get(participant_id)
            if assessment_date is None:
                # No date -> no candidate search is possible. Missing, never
                # zero: a gap in the measure, not evidence of a wrong answer.
                answer_rows.append({
                    **answer_base, "assessment_date": None, "status": "no_date",
                    **summarize_answer("none", []),
                    "recalled_recent": None,
                })
                continue

            try:
                identified, raw_identify = identify_events(transcript)
                identified = sanitize_events(identified, transcript)
            except Exception as e:
                print(f"  ERROR (identify) on {txt_path.name}: {e}")
                continue

            if identified.status != "extracted":
                answer_rows.append({
                    **answer_base, "assessment_date": assessment_date.isoformat(),
                    "status": "needs_review", **summarize_answer("none", []),
                    "recalled_recent": None,
                })
                audit_log.write(json.dumps({
                    "participant_id": participant_id, "model_name": MODEL_PATH,
                    "prompt_version": PROMPT_VERSION, "raw_identify": raw_identify,
                    "status": "needs_review",
                }) + "\n")
                continue

            participant_events: list[dict] = []
            audit_events: list[dict] = []
            for event_index, event in enumerate(identified.events, 1):
                # Candidates are retrieved FOR THIS EVENT (its description +
                # the participant's own words about it), from the WINDOW_DAYS
                # before the assessment date only -- the only period we care
                # about, and all the news cache holds.
                candidates = select_candidates(
                    f"{event.description} {event.evidence_quote}",
                    events_cache, assessment_date, window_days=WINDOW_DAYS,
                )
                try:
                    assessment, raw_assess = assess_event(
                        transcript, event.description, event.evidence_quote, candidates,
                    )
                    assessment = sanitize_assessment(assessment, transcript, len(candidates))
                except Exception as e:
                    print(f"  ERROR (assess) on {txt_path.name} event {event_index}: {e}")
                    continue

                if assessment.status != "extracted":
                    # Keep the event (so it's still listed) but flag it rather
                    # than silently treating it as a non-match.
                    status, articles = "no_match", []
                    level, details = "none", []
                    review = True
                else:
                    status, articles = classify_event(
                        assessment.matched_candidate_numbers, candidates, assessment_date,
                    )
                    level, details = assessment.detail_level, assessment.details
                    review = False

                row = {
                    **base,
                    "assessment_date": assessment_date.isoformat(),
                    "event_index": event_index,
                    "description": event.description,
                    "evidence_quote": event.evidence_quote,
                    "match_status": status,
                    "n_matched_articles": len(articles),
                    # The model lists matches best-first, so [0] is the article
                    # this event most directly matches (blank if none).
                    "best_article_headline": articles[0]["headline"] if articles else None,
                    "best_article_date": articles[0]["date"] if articles else None,
                    "best_article_source": articles[0]["source"] if articles else None,
                    "best_article_url": articles[0].get("url") if articles else None,
                    "matched_articles": json.dumps(articles, default=str),
                    "detail_level": level,
                    "n_details": len(details),
                    "details": " | ".join(details),
                    "needs_review": review,
                }
                participant_events.append(row)
                audit_events.append({
                    "event_index": event_index,
                    "n_candidates": len(candidates),
                    "candidates": candidates.to_dict("records"),
                    "raw_assess": raw_assess,
                    "parsed": assessment.model_dump(),
                })

            event_rows.extend(participant_events)
            answer_rows.append({
                **answer_base,
                "assessment_date": assessment_date.isoformat(),
                "status": "extracted",
                **summarize_answer(identified.answer_quality, participant_events),
            })
            audit_log.write(json.dumps({
                "participant_id": participant_id,
                "model_name": MODEL_PATH,
                "prompt_version": PROMPT_VERSION,
                "raw_identify": raw_identify,
                "identified": identified.model_dump(),
                "events": audit_events,
            }, default=str) + "\n")

            if i % 20 == 0:
                print(f"  {i}/{len(transcript_files)} done")

    events_df = pd.DataFrame(event_rows, columns=EVENT_COLUMNS + ["needs_review"])
    answers_df = pd.DataFrame(answer_rows)
    events_df.to_csv(EVENTS_OUTPUT_PATH, index=False)
    answers_df.to_csv(ANSWERS_OUTPUT_PATH, index=False)
    print(f"\nSaved: {EVENTS_OUTPUT_PATH}  ({len(events_df)} events)")
    print(f"Saved: {ANSWERS_OUTPUT_PATH}  ({len(answers_df)} participants)")
    print(f"Audit log: {AUDIT_LOG_PATH}")
    print("\nStatus counts:")
    print(answers_df["status"].value_counts())
    print("\nRecalled a recent event:")
    print(answers_df["recalled_recent"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
