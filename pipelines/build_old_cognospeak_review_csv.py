"""Build a pre-filled CSV for manually reviewing old-CognoSpeak question/timing extraction.

Why: extract_old_cognospeak_transcripts.py already fuzzy-matches interviewer turns to task
codes, but only keeps the ones that clear MATCH_THRESHOLD, and collapses re-asked questions
down to one anchor per task. That's right for the automated pipeline, but it hides exactly
the turns a human reviewer most needs to see: low-scoring or unmatched ones (e.g. rdp, which
PromptsMapping.csv only documents in stroke-protocol wording and so barely matches for ivamem
participants), and any case where a question was asked more than once.

This script instead writes ONE ROW PER INTERVIEWER/ASSISTANT TURN (every turn parse_label_file
would consider a real question source, i.e. QUESTION_SOURCE_RE -- "Oth" bystander turns are
skipped, same as the main extraction treats them), for all 65 old-system participants, with:
  - the verbatim interviewer text and its timestamp, so you can read what was actually asked
  - the best-guess task code and its match score (blank code if score < MATCH_THRESHOLD, so
    you can tell at a glance which rows the automated pipeline would have kept)
  - a pre-filled guess at the answer's start/end, using the same forward-scan the main
    extraction uses (first PAT turn's start; widened to the start of the next question, or the
    last PAT turn's end if this is the session's last turn)
  - task_code/answer_start/answer_end as the columns to actually EDIT -- they start out equal
    to the guess, so confirming a row means leaving it alone, and correcting one means
    overwriting it
  - a blank "confirmed" column (fill in anything, e.g. "y") and a blank "notes" column

Run from the repo root: python pipelines/build_old_cognospeak_review_csv.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_old_cognospeak_transcripts import (  # noqa: E402
    MATCH_THRESHOLD,
    PARTICIPANT_RE,
    QUESTION_SOURCE_RE,
    best_task_for_turn,
    find_label_file,
    load_old_system_participants,
    load_task_variants,
    parse_label_file,
)

OUTPUT_PATH = Path("data/raw/old_cognospeak_manual_review.csv")


def guess_answer_span(turns: list[tuple[float, float, str, str]], turn_idx: int) -> tuple[float | None, float | None]:
    """Same forward-scan extract_answers() uses, applied from one specific interviewer turn."""
    answer_start, answer_end = None, None
    for start, end, speaker, text in turns[turn_idx + 1:]:
        if QUESTION_SOURCE_RE.match(speaker):
            answer_end = start
            break
        if PARTICIPANT_RE.match(speaker) and text:
            if answer_start is None:
                answer_start = start
            answer_end = end
    return answer_start, answer_end


def main():
    task_variants = load_task_variants()
    participants = load_old_system_participants()

    rows = []
    no_label_file = []
    for process_id, info in sorted(participants.items()):
        label_path = find_label_file(info["full_path"])
        if label_path is None:
            no_label_file.append(process_id)
            continue

        turns = parse_label_file(label_path)
        for turn_idx, (start, end, speaker, text) in enumerate(turns):
            if not QUESTION_SOURCE_RE.match(speaker):
                continue

            best_task, score = best_task_for_turn(text, task_variants)
            task_guess = best_task if best_task and score >= MATCH_THRESHOLD else ""
            answer_start, answer_end = guess_answer_span(turns, turn_idx)

            rows.append({
                "process_id": process_id,
                "system": info["system"],
                "turn_idx": turn_idx,
                "speaker": speaker,
                "question_text": text,
                "question_start": start,
                "question_end": end,
                "match_score": round(score, 2),
                "task_code": task_guess,
                "answer_start": answer_start,
                "answer_end": answer_end,
                "confirmed": "",
                "notes": "",
                "label_file": str(label_path),
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "process_id", "system", "turn_idx", "speaker", "question_text", "question_start",
        "question_end", "match_score", "task_code", "answer_start", "answer_end", "confirmed",
        "notes", "label_file",
    ]
    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_participants = len(participants) - len(no_label_file)
    n_matched = sum(1 for r in rows if r["task_code"])
    n_unmatched = len(rows) - n_matched
    print(f"{n_participants} participants, {len(rows)} interviewer turns written to {OUTPUT_PATH}")
    print(f"  {n_matched} pre-filled with a task_code guess (score >= {MATCH_THRESHOLD})")
    print(f"  {n_unmatched} left blank for you to fill in -- these need the most attention")
    if no_label_file:
        print(f"\nNo label file found for: {no_label_file}")


if __name__ == "__main__":
    main()
