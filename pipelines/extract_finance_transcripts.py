"""Extract "finance" (FIN) transcripts from PROCESS_Master into data/raw/text/FIN.

See pipelines/extract_memory_transcripts.py for the full explanation -- PROCESS_Master/files
now names every file uniformly as <process_id>__<TASK>.{wav,txt}, so this is just a glob on
*__FIN.txt, no per-system question-code lookup needed. Writes
data/raw/text/FIN/<process_id>__FIN.txt. Run on the HPC (login node is fine -- no
GPU needed, just reads /mnt/parscratch).
"""
import os
import shutil
from pathlib import Path

project_root = Path(__file__).parent.parent
os.chdir(project_root)

PROCESS_MASTER_DIR = Path("/mnt/parscratch/users/acp23fp/DementiaSpeech_HPC/PROCESS_Master/files")
OUTPUT_DIR = Path("data/raw/text/FIN")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    copied, skipped = [], []
    for folder in sorted(PROCESS_MASTER_DIR.iterdir()):
        if not folder.is_dir():
            continue
        process_id = folder.name

        matches = sorted(folder.glob("*__FIN.txt"))
        if len(matches) != 1:
            skipped.append((process_id, f"found {len(matches)} files matching *__FIN.txt"))
            continue

        dest = OUTPUT_DIR / f"{process_id}__FIN.txt"
        shutil.copy2(matches[0], dest)
        copied.append(process_id)

    print(f"Copied {len(copied)} transcripts to {OUTPUT_DIR}")
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for process_id, reason in skipped:
            print(f"  {process_id}: {reason}")


if __name__ == "__main__":
    main()
