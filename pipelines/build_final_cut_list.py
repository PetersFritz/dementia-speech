"""Turn the hand-reviewed old_cognospeak_manual_review_final.csv into a cut list for
cut_old_cognospeak_by_task.sh, producing the final old-CognoSpeak dataset.

Input schema (as hand-edited in a spreadsheet from old_cognospeak_review_cut_list.csv):
  process_id, system, task, start, end, picture, check, source_wav, label_file, comment
"task" still carries the "<N>_<code>" question-order prefix from the review pass (e.g.
"01_rdp") -- this script strips that prefix, since the final dataset doesn't need it.

Unlike build_review_cut_list.py (which numbers every clip by question order, to make
cross-checking against the transcript easy during review), this is the FINAL pass: review is
done, every row's task code has been hand-corrected, and rows that didn't belong have been
deleted. So filenames here drop the question-order prefix and are just <process_id>__<TASK>.wav
-- clean, stable names for the dataset that will actually get used downstream.

The one wrinkle: a task can legitimately occur twice for one participant (e.g. a reading or
picture-description prompt asked in two parts). Dropping the rank prefix means two such rows
would collide on the same filename, so repeats are disambiguated by appending _2, _3, ... (first
occurrence stays plain).

The picture column (hand-filled "cookie" or "street") disambiguates the two picture-description
prompts -- rows with a non-blank picture get it appended to the task code, e.g.
PDC_COOKIE / PDC_STREET, before the repeat-numbering above is applied.

The check column ("ok" vs. "check again" / "problem" / etc.) is NOT used to filter rows here --
every row with timing gets cut. Rows not marked "ok" are reported at the end so you know which
clips in the output still carry an open question from review.

Usage: python pipelines/build_final_cut_list.py
"""
import csv
import re
from pathlib import Path

INPUT_CSV_PATH = Path("data/raw/old_cognospeak_manual_review_final.csv")
OUTPUT_PATH = Path("data/raw/old_cognospeak_final_cut_list.csv")

RANK_PREFIX_RE = re.compile(r"^\d+_(.+)$")


def strip_rank_prefix(task: str) -> str:
    m = RANK_PREFIX_RE.match(task)
    return m.group(1) if m else task


def main():
    with INPUT_CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    skipped_no_timing = []
    flagged = []
    task_seen_count = {}  # (process_id, task_label) -> count so far

    for row in rows:
        if not row["start"] or not row["end"]:
            skipped_no_timing.append((row["process_id"], row["task"]))
            continue

        task_label = strip_rank_prefix(row["task"].strip()).upper()

        picture = (row.get("picture") or "").strip()
        if picture:
            task_label = f"{task_label}_{picture.upper()}"

        key = (row["process_id"], task_label)
        task_seen_count[key] = task_seen_count.get(key, 0) + 1
        occurrence = task_seen_count[key]
        if occurrence > 1:
            task_label = f"{task_label}_{occurrence}"

        check = (row.get("check") or "").strip().lower()
        if check and check != "ok":
            flagged.append((row["process_id"], task_label, row["check"]))

        out_rows.append({
            "process_id": row["process_id"],
            "system": row["system"],
            "task": task_label,
            "start": row["start"],
            "end": row["end"],
            "source_wav": row["source_wav"],
            "label_file": row["label_file"],
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "process_id", "system", "task", "start", "end", "source_wav", "label_file",
        ])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"{len(out_rows)} clips to cut -> {OUTPUT_PATH}")
    if skipped_no_timing:
        print(f"{len(skipped_no_timing)} rows skipped (no start/end timing):")
        for process_id, task in skipped_no_timing:
            print(f"  {process_id} ({task})")
    if flagged:
        print(f"\n{len(flagged)} rows included but flagged in 'check' (not \"ok\"):")
        for process_id, task_label, check in flagged:
            print(f"  {process_id} {task_label}: check={check!r}")


if __name__ == "__main__":
    main()
