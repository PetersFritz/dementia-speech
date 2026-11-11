"""Batch-score "what has been in the news recently?" (STN) transcripts via a local LLM.

Needs a GPU (loads Qwen2.5-14B-Instruct) -- run through
pipelines/slurm/score_news.sbatch, not directly on a login node. The
dashboard never calls the model itself: like score_finance.py and
score_memory_complaints.py, this writes a CSV once and Shiny just reads it
at runtime.

EXPERIMENTAL (per the Tier 2 TODO, docs/to_do/TODO_dashboard.md): hand-rate
50-100 answers against this pipeline's output before showing the resulting
measure in the dashboard as anything more than exploratory. See
docs/news_scoring_rules.md for the full rubric this implements.

Requires:
  - data/raw/text/STN/<process_id>__STN.txt (pipelines/extract_news_transcripts.py)
  - data/raw/Process_demographics.csv's `assessment_date` column
    (pipelines/scratch/add_assessment_dates.py) -- participants without one
    (old-CognoSpeak) are written out with category='no_date', score=None,
    and never reach the model (see the Tier 2 TODO's words/duration
    fallback for those).
  - data/processed/news_events_cache.csv (pipelines/build_news_event_cache.py)
"""
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
os.chdir(project_root)

from dementia_speech.paths import DATA_DIR
from dementia_speech.tasks._shared.model import MODEL_PATH
from dementia_speech.tasks.news.extract import extract
from dementia_speech.tasks.news.prompt import PROMPT_VERSION
from dementia_speech.tasks.news.retrieval import select_candidates
from dementia_speech.tasks.news.scoring import categorize, compute_detail_metrics
from dementia_speech.tasks.news.verify import sanitize_claims

TRANSCRIPT_DIR = DATA_DIR / "raw/text/STN"  # "recent news" transcripts
DEMOGRAPHICS_PATH = DATA_DIR / "raw/Process_demographics.csv"
METADATA_PATH = DATA_DIR / "raw/meta-info.csv"
EVENTS_CACHE_PATH = DATA_DIR / "processed/news_events_cache.csv"
OUTPUT_PATH = DATA_DIR / "processed/news_scores.csv"
# Raw model output + the candidate list actually shown, one line per
# participant -- separate from OUTPUT_PATH so the main CSV stays a tidy
# analysis table. Covers the Tier 2 TODO's "log raw output, model name,
# prompt version" requirement.
AUDIT_LOG_PATH = DATA_DIR / "processed/news_scores_audit.jsonl"


def extract_participant_id(filename) -> str:
    """<process_id>__STN.txt -> process_id (see extract_news_transcripts.py,
    which writes this exact filename from Process_demographics.csv's own
    `process_ID` column -- the join key used below).

    process_ID itself routinely contains '__' (e.g. 'PROCESS-2_rec__001'),
    so the full filename stem can have more than one '__'-separated part
    ('PROCESS-2_rec__001__STN') -- confirmed on the HPC run, 2026-11-11,
    where taking just stem.split('__')[0] truncated every process_ID to
    'PROCESS-2_rec', causing a 100% assessment_date lookup miss. The task
    code ('STN') is always the LAST '__'-separated segment, so process_id
    is everything before that, joined back with '__', not just the first
    segment.
    """
    return "__".join(filename.stem.split("__")[:-1])


def load_assessment_dates() -> dict[str, date]:
    # dtype=str on process_ID: otherwise pandas infers it as numeric and
    # silently drops any leading zero, so EVERY lookup below misses against
    # the (always-string) filename-derived participant_id -- confirmed on
    # the HPC run, 2026-11-11, where this produced 100% 'no_date' rows. Same
    # gotcha pipelines/score_fluency_discrepancy.py already guards against
    # for its own participant_id column.
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={"process_ID": str})
    # zip, not itertuples: itertuples builds a namedtuple, which silently
    # renames any leading-underscore column (e.g. a '_date' column) to a
    # positional fallback name -- `row._date` then raises AttributeError
    # (also confirmed on the HPC run).
    parsed = pd.to_datetime(demographics["assessment_date"], errors="coerce")
    valid = parsed.notna()
    return {
        pid: parsed_date.date()
        for pid, parsed_date in zip(demographics.loc[valid, "process_ID"], parsed[valid])
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(METADATA_PATH)
    assessment_dates = load_assessment_dates()
    events_df = pd.read_csv(EVENTS_CACHE_PATH)

    transcript_files = sorted(TRANSCRIPT_DIR.glob("*.txt"))
    print(f"Found {len(transcript_files)} transcripts")

    rows = []
    with AUDIT_LOG_PATH.open("a") as audit_log:
        for i, txt_path in enumerate(transcript_files, 1):
            participant_id = extract_participant_id(txt_path)
            transcript = txt_path.read_text().strip()

            match = metadata[metadata["IDs"].astype(str).str.contains(participant_id)]
            diagnosis = match.iloc[0]["diagnosis"] if len(match) > 0 else "Unknown"

            base_row = {
                "participant_id": participant_id,
                "diagnosis": diagnosis,
                "source_file": txt_path.name,
            }

            word_count = len(transcript.split())

            assessment_date = assessment_dates.get(participant_id)
            if assessment_date is None:
                rows.append({
                    **base_row, "assessment_date": None, "status": "no_date",
                    "answer_quality": None, "category": "no_date", "score": None,
                    "n_claims": 0, "n_matched_claims": 0,
                    "matched_event_date": None, "matched_event_headline": None,
                    "matched_event_source": None, "summary": None, "evidence_quotes": None,
                    "word_count": word_count, "detail_count": None,
                    "best_claim_detail_count": None, "detail_rate_per_100_words": None,
                })
                continue

            try:
                candidates = select_candidates(transcript, events_df, assessment_date)
                result, raw = extract(transcript, candidates)
                result = sanitize_claims(result, transcript, len(candidates))
                category, score, matched_events, best_claim = categorize(result, candidates, assessment_date)
                detail_metrics = compute_detail_metrics(result, best_claim, word_count)
            except Exception as e:
                print(f"  ERROR on {txt_path.name}: {e}")
                continue

            # The single matched_event_* columns below describe best_claim --
            # the one claim that decided the category/score -- for a quick
            # read of the CSV; every claim and every match (not just this
            # one) is still in the audit log.
            best_event = (
                candidates.iloc[best_claim.matched_candidate_number - 1].to_dict()
                if best_claim is not None else {}
            )

            rows.append({
                **base_row,
                "assessment_date": assessment_date.isoformat(),
                "status": result.status,
                "answer_quality": result.answer_quality,
                "category": category,
                "score": score,
                "n_claims": len(result.claims),
                "n_matched_claims": len(matched_events),
                "matched_event_date": best_event.get("date"),
                "matched_event_headline": best_event.get("headline"),
                "matched_event_source": best_event.get("source"),
                "summary": result.summary,
                "evidence_quotes": " | ".join(c.evidence_quote for c in result.claims if c.evidence_quote),
                "word_count": word_count,
                **detail_metrics,
            })

            audit_log.write(json.dumps({
                "participant_id": participant_id,
                "model_name": MODEL_PATH,
                "prompt_version": PROMPT_VERSION,
                "n_candidates": len(candidates),
                "raw_completion": raw,
                "parsed_result": result.model_dump(),
                "matched_events": matched_events,
                "category": category,
                "score": score,
                "detail_metrics": detail_metrics,
            }) + "\n")

            if i % 20 == 0:
                print(f"  {i}/{len(transcript_files)} done")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Audit log: {AUDIT_LOG_PATH}")
    print("\nCategory counts:")
    print(results_df["category"].value_counts())


if __name__ == "__main__":
    main()
