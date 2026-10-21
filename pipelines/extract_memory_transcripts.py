"""Extract "memory-problems" (MEM) transcripts from PROCESS_Master into data/raw/text/MEM.

PROCESS_Master/files/<process_id>/ now names every file uniformly as
<process_id>__<TASK>.{wav,txt} (both the old-CognoSpeak and cchat participants were
renamed onto this scheme), so finding the memory-problems answer for any participant is
just a glob on *__MEM.txt -- no more per-system question-code lookup needed. This writes
data/raw/text/MEM/<process_id>__MEM.txt, matching the existing data/raw/text/{PFT,CTD,SFT}
convention, so score_memory_complaints.py needs no changes. Run on the HPC (login node is
fine -- no GPU needed, just reads /mnt/parscratch).
"""
import os
import shutil
from pathlib import Path

project_root = Path(__file__).parent.parent
os.chdir(project_root)

PROCESS_MASTER_DIR = Path("/mnt/parscratch/users/acp23fp/PROCESS_Master/files")
OUTPUT_DIR = Path("data/raw/text/MEM")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    copied, skipped = [], []
    for folder in sorted(PROCESS_MASTER_DIR.iterdir()):
        if not folder.is_dir():
            continue
        process_id = folder.name

        matches = sorted(folder.glob("*__MEM.txt"))
        if len(matches) != 1:
            skipped.append((process_id, f"found {len(matches)} files matching *__MEM.txt"))
            continue

        dest = OUTPUT_DIR / f"{process_id}__MEM.txt"
        shutil.copy2(matches[0], dest)
        copied.append(process_id)

    print(f"Copied {len(copied)} transcripts to {OUTPUT_DIR}")
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for process_id, reason in skipped:
            print(f"  {process_id}: {reason}")


if __name__ == "__main__":
    main()
