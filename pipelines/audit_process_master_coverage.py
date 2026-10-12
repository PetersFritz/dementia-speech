"""Audit which (participant, task) question files exist in PROCESS_Master.

Checks every question_type in data/raw/matching_prompts.csv against every participant in
data/raw/Process_demographics.csv -- i.e. whether PROCESS_Master/files/<process_id>/ actually
has a *_{code}.txt file for that participant's system and task. Surfaces gaps like the finance
one found for PROCESS-2_rec__085 and PROCESS-2_rec__243 (old-CognoSpeak participants whose
per-question split apparently missed that question, even though the full combined dialogue
transcript has it) across every task, not just one.

Pure read-only file-existence checks -- cheap enough to run on a login node, no sbatch needed.
Checks each participant's file *count* first (one directory listing) against how many task
files their system should produce; only runs the slower per-task glob for participants whose
count doesn't match, rather than globbing all 10 tasks for every one of the 400 participants.
Run from the repo root: python pipelines/audit_process_master_coverage.py
"""
import csv
import os
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
os.chdir(project_root)

PROCESS_MASTER_DIR = Path("/mnt/parscratch/users/acp23fp/PROCESS_Master/files")
DEMOGRAPHICS_PATH = Path("data/raw/Process_demographics.csv")
MATCHING_PROMPTS_PATH = Path("data/raw/matching_prompts.csv")
OUTPUT_PATH = Path("data/processed/process_master_coverage.csv")


def load_task_system_codes() -> dict[str, dict[str, str]]:
    """{question_type: {system: code}} for every task row in matching_prompts.csv."""
    with MATCHING_PROMPTS_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    tasks = {}
    for row in rows:
        task = row["question_type"]
        systems = [k for k in row if k not in ("question_type", "PromptLongCode")]
        tasks[task] = {system: row[system] for system in systems if row[system]}
    return tasks


def load_participant_systems() -> dict[str, str]:
    """{process_id: system_name}, with a trailing " (inferred)" marker stripped."""
    with DEMOGRAPHICS_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        row["process_ID"]: row["system"].replace(" (inferred)", "").strip()
        for row in rows
    }


def check_file(folder: Path, code: str | None) -> str:
    """Detailed per-task check -- only called once the cheap count check below has
    already shown a participant's folder doesn't have the expected number of files."""
    if not code:
        return "no_code_for_system"
    matches = list(folder.glob(f"*_{code}.txt"))
    if len(matches) == 1:
        return "ok"
    if len(matches) == 0:
        return "missing"
    return "multiple_matches"


def expected_count(system: str, task_codes: dict[str, dict[str, str]]) -> int:
    """How many task files a participant on this system should have."""
    return sum(1 for codes in task_codes.values() if codes.get(system))


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    task_codes = load_task_system_codes()
    participant_systems = load_participant_systems()
    tasks = sorted(task_codes)

    rows = []
    for process_id, system in sorted(participant_systems.items()):
        folder = PROCESS_MASTER_DIR / process_id

        if not folder.is_dir():
            rows.extend({
                "process_id": process_id, "system": system, "task": task,
                "code": task_codes[task].get(system, ""), "status": "folder_missing",
            } for task in tasks)
            continue

        # Cheap first pass: one directory listing, compared against how many task
        # files this participant's system should produce. Only participants whose
        # count doesn't match pay for the slower per-task glob below.
        actual = len(list(folder.glob("*.txt")))
        expected = expected_count(system, task_codes)

        if actual == expected:
            rows.extend({
                "process_id": process_id, "system": system, "task": task,
                "code": task_codes[task].get(system, ""),
                "status": "ok" if task_codes[task].get(system) else "no_code_for_system",
            } for task in tasks)
            continue

        # Count mismatch -- figure out exactly which task(s) are missing or duplicated.
        rows.extend({
            "process_id": process_id, "system": system, "task": task,
            "code": task_codes[task].get(system, ""),
            "status": check_file(folder, task_codes[task].get(system)),
        } for task in tasks)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH} ({len(df)} rows: {len(participant_systems)} participants x {len(tasks)} tasks)")
    print("\nCoverage per task:")
    print(df.groupby("task")["status"].value_counts().unstack(fill_value=0))

    gaps = df[df["status"] != "ok"]
    if len(gaps):
        print(f"\n{len(gaps)} gaps found:")
        for _, r in gaps.iterrows():
            print(f"  {r['process_id']} ({r['system']}) / {r['task']}: {r['status']}")
    else:
        print("\nNo gaps found -- full coverage.")


if __name__ == "__main__":
    main()
