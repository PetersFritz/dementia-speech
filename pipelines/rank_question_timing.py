"""Add a question-order rank prefix to old_cognospeak_question_timing.csv's task column, for
cutting -- so the resulting wav filenames carry the same "Nth question in the recording" prefix
as the review clips (build_review_cut_list.py), making it easy to check a clip against its
position in the label file / manual-review CSV.

Writes a SEPARATE file rather than rewriting old_cognospeak_question_timing.csv in place: that
file's task column is consumed elsewhere as a plain code (e.g. it's what the per-task text files
under data/raw/text/<TASK>/ are named from), so it stays clean. Only the cutting step needs the
rank, via this derived file.

old_cognospeak_question_timing.csv is written sorted by (process_id, task) -- alphabetical, not
chronological -- so getting the rank right means re-sorting by (process_id, start) first.

Run from the repo root: python pipelines/rank_question_timing.py
"""
import csv
from pathlib import Path

TIMING_CSV_PATH = Path("data/raw/old_cognospeak_question_timing.csv")
OUTPUT_PATH = Path("data/raw/old_cognospeak_question_timing_ranked.csv")


def main():
    with TIMING_CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda r: (r["process_id"], float(r["start"])))

    rank = 0
    prev_process_id = None
    for row in rows:
        if row["process_id"] != prev_process_id:
            rank = 0
            prev_process_id = row["process_id"]
        rank += 1
        row["task"] = f"{rank:02d}_{row['task']}"

    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "process_id", "system", "task", "start", "end", "source_wav", "label_file",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} rows ranked -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
