"""Extract Q2 ("memory-problems") transcripts from PROCESS_Master into data/raw/text/MEM.

PROCESS_Master/files/<process_id>/ keeps each participant's original (per-system) file
names rather than renaming them to process_id. Which file is the "memory-problems" answer
depends on which system collected that participant -- data/raw/Process_demographics.csv's
"system" column gives that, and data/raw/matching_prompts.csv maps (task, system) to the
question code baked into the filename (e.g. "_Q2.txt" for cchat, "_q2.txt" for the
IVA-derived systems). This writes data/raw/text/MEM/<process_id>__MEM.txt, matching the
existing data/raw/text/{PFT,CTD,SFT} convention, so score_memory_complaints.py needs no
changes. Run on the HPC (login node is fine -- no GPU needed, just reads /mnt/parscratch).
"""
import csv
import os
import shutil
from pathlib import Path

project_root = Path(__file__).parent.parent
os.chdir(project_root)

PROCESS_MASTER_DIR = Path("/mnt/parscratch/users/acp23fp/PROCESS_Master/files")
DEMOGRAPHICS_PATH = Path("data/raw/Process_demographics.csv")
MATCHING_PROMPTS_PATH = Path("data/raw/matching_prompts.csv")
OUTPUT_DIR = Path("data/raw/text/MEM")

TASK_QUESTION_TYPE = "mem"  # matching_prompts.csv row for "memory-problems"


def load_system_to_code() -> dict[str, str]:
    """{system_name: question_code} for the memory-problems task, e.g. {"cchat": "Q2", "iva2": "q2", ...}."""
    with MATCHING_PROMPTS_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if r["question_type"] == TASK_QUESTION_TYPE)
    systems = [k for k in row if k not in ("question_type", "PromptLongCode")]
    return {system: row[system] for system in systems if row[system]}


def load_participant_systems() -> dict[str, str]:
    """{process_id: system_name}, with a trailing " (inferred)" marker stripped."""
    with DEMOGRAPHICS_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        row["process_ID"]: row["system"].replace(" (inferred)", "").strip()
        for row in rows
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    system_to_code = load_system_to_code()
    participant_systems = load_participant_systems()

    copied, skipped = [], []
    for process_id, system in sorted(participant_systems.items()):
        code = system_to_code.get(system)
        if not code:
            skipped.append((process_id, system, "no question code for this system"))
            continue

        folder = PROCESS_MASTER_DIR / process_id
        if not folder.is_dir():
            skipped.append((process_id, system, "participant folder not found"))
            continue

        matches = sorted(folder.glob(f"*_{code}.txt"))
        if len(matches) != 1:
            skipped.append((process_id, system, f"found {len(matches)} files matching *_{code}.txt"))
            continue

        dest = OUTPUT_DIR / f"{process_id}__MEM.txt"
        shutil.copy2(matches[0], dest)
        copied.append(process_id)

    print(f"Copied {len(copied)} transcripts to {OUTPUT_DIR}")
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for process_id, system, reason in skipped:
            print(f"  {process_id} ({system}): {reason}")


if __name__ == "__main__":
    main()
