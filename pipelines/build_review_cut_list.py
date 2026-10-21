"""Turn reviewed rows of old_cognospeak_manual_review.csv into a cut list for
cut_old_cognospeak_by_task.sh, so you can listen through a participant's answers right after
reviewing their row in the spreadsheet.

Why a separate step rather than feeding the manual-review CSV straight to the sox script: that
CSV's question_text column has free-text with commas in it, which a naive `IFS=',' read` shell
loop can't parse safely. This script does the (robust, Python-csv) parsing and writes a minimal
CSV with exactly the 7 columns the existing sox script already expects -- no free text, so a
shell loop can read it safely.

Each row becomes <N>_<task>, N being the 1-based rank of that row among ALL of this
participant's question-turn rows in the review CSV (in transcript order, whether or not that
particular row got cut) -- so clip "3_mem" is the 3rd question asked in the recording, matching
up directly against the 3rd row of that participant's block in old_cognospeak_manual_review.csv
or the 3rd INT/ASS turn in the label file, for easy cross-checking. Zero-padded to 2 digits so
filenames still sort in question order (not "1, 10, 11, ..., 2").

A task can legitimately appear more than once in a transcript (e.g. a reading or
picture-description prompt asked in two parts) -- every row with both answer timestamps filled
in gets its own clip, so you can listen to each occurrence separately.

A blank task_code does NOT mean there's nothing to listen to -- it means the match score was
below MATCH_THRESHOLD, which is exactly the case that most needs a human ear (e.g. "rdp" is
badly under-matched for ivamem participants: PromptsMapping.csv only documents the
stroke-protocol wording for it, so ivamem's actual "can you please read this paragraph out
aloud" scores 0.4, just under threshold, even though it's obviously a reading prompt). Those
rows are included here too, labelled "<N>_unmatched" instead of a task code, as long as a PAT
turn was found to time -- you listen, then go fill in the right task_code yourself in
old_cognospeak_manual_review.csv. Only rows with NO guessed answer span at all (e.g. a
re-prompt where the participant never actually answered) have nothing to cut and are skipped.

Usage: python pipelines/build_review_cut_list.py PROCESS-2_rec__009 [more_process_ids...]
       python pipelines/build_review_cut_list.py --all   (every participant in the review CSV)
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_old_cognospeak_transcripts import load_old_system_participants  # noqa: E402

REVIEW_CSV_PATH = Path("data/raw/old_cognospeak_manual_review.csv")
OUTPUT_PATH = Path("data/raw/old_cognospeak_review_cut_list.csv")


def main():
    if len(sys.argv) < 2:
        print("Usage: python pipelines/build_review_cut_list.py <process_id> [...] | --all")
        sys.exit(1)

    participants = load_old_system_participants()

    with REVIEW_CSV_PATH.open(newline="") as f:
        review_rows = list(csv.DictReader(f))

    if sys.argv[1] == "--all":
        wanted_ids = None
    else:
        wanted_ids = set(sys.argv[1:])

    out_rows = []
    skipped_no_timing = []
    question_rank = {}  # process_id -> count of question-turn rows seen so far (1-based)
    for row in review_rows:
        if wanted_ids is not None and row["process_id"] not in wanted_ids:
            continue

        question_rank[row["process_id"]] = question_rank.get(row["process_id"], 0) + 1
        rank = question_rank[row["process_id"]]

        if not row["answer_start"] or not row["answer_end"]:
            skipped_no_timing.append((row["process_id"], rank, row["task_code"]))
            continue

        task_label = row["task_code"] or "unmatched"
        info = participants[row["process_id"]]
        out_rows.append({
            "process_id": row["process_id"],
            "system": info["system"],
            "task": f"{rank:02d}_{task_label}",
            "start": row["answer_start"],
            "end": row["answer_end"],
            "source_wav": info["full_path"],
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
        print(f"{len(skipped_no_timing)} rows skipped (no PAT turn found to time -- nothing to cut):")
        for process_id, rank, task_code in skipped_no_timing:
            print(f"  {process_id} question #{rank} (task_code={task_code!r})")


if __name__ == "__main__":
    main()
