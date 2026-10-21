"""Extract every task a participant actually answered (not just the 10 "interview" questions --
also fluency, picture description, paragraph reading, etc.: whatever's in PromptsMapping.csv)
for old CognoSpeak participants, directly from manually-reviewed label transcripts, rather than
trusting PROCESS_Master's pre-split per-question files.

Why: PROCESS_Master's old-CognoSpeak files are split and named using a positional q-number
assigned by an earlier process, and that numbering has been shown to be unreliable for at
least the ivastroke_* systems -- e.g. PROCESS-2_rec__017's file named "*_q8.txt" actually
contains the "long-term-job" answer, not finance (confirmed against data/raw/questions.txt
and the participant's own label transcript, where ltjs / lstj / fin appear as three separate
turns in that order, apparently compressed into one position somewhere in PROCESS_Master's
split). This script sidesteps q-numbers entirely: it reads each participant's label transcript
(one turn per line, with the interviewer's actual spoken question, not just a code) and
identifies each task by fuzzy-matching the interviewer's turn text against every known
phrasing of that task across all systems (data/raw/PromptsMapping.csv) -- the same technique
notebooks/04_MatchFiles.ipynb already uses for participant matching.

Label source: CognoSpeak/Audacity-labels and IVA/Audacity-labels (per-system, matching the
Wav-files split), NOT semi-automatic-man-labels -- manual diff against PROCESS-2_rec__009
(261-M043-2021-2-8-11-3-51) showed semi-automatic-man-labels silently drops whole interviewer
turns (its own weekend/"stn" question is missing entirely, with the two surrounding PAT answers
run together as if the participant had said both in one turn) that Audacity-labels keeps. The
tradeoff: Audacity-labels tags each INT turn with the OLD, already-distrusted positional code
("INT q4:...", "INT gp:...") right in the speaker field -- parse_label_file() strips that off
and keeps only the speaker role, since fuzzy content-matching never trusted the code anyway.
semi-automatic-man-labels is kept last as a fallback for the rare recording missing from both
Audacity-labels dirs. cchat participants are untouched -- their existing PROCESS_Master-based
extraction is reliable and unaffected.

Writes/overwrites data/raw/text/<TASK_UPPER>/<process_id>__<TASK_UPPER>.txt for every task
confidently matched -- including re-extracting mem/fin for these 65 participants, since their
previous PROCESS_Master-derived versions can't be trusted either. A task genuinely not asked
by a participant's system (e.g. wrm/lts/ltlj for ivastroke) correctly gets no file, same as
matching_prompts.csv already shows via blank cells for that system.

Also writes data/raw/old_cognospeak_question_timing.csv: one row per (process_id, task) with
the start/end seconds of the participant's answer in the ORIGINAL recording (full_path) --
the new, content-verified replacement for the old, unreliable q-number-based mapping. This is
what the next step (re-cutting + re-transcribing with Whisper) should cut from: start is the
first PAT turn's start, end is the last PAT turn's end, i.e. the answer only -- it does not
include the interviewer's question audio.

Pure text processing, no GPU -- runs locally or on a login node.
Run from the repo root: python pipelines/extract_old_cognospeak_transcripts.py
"""
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).parent.parent
os.chdir(project_root)

DEMOGRAPHICS_PATH = Path("data/raw/Process_demographics.csv")
PROMPTS_MAPPING_PATH = Path("data/raw/PromptsMapping.csv")
TEXT_DIR = Path("data/raw/text")
TIMING_OUTPUT_PATH = Path("data/raw/old_cognospeak_question_timing.csv")

LABEL_DIRS = [
    Path("/mnt/xdrive/cchat/Shared/CognoSpeak_OLD/CognoSpeak/Audacity-labels"),
    Path("/mnt/xdrive/cchat/Shared/CognoSpeak_OLD/IVA/Audacity-labels"),
    Path("/mnt/xdrive/cchat/Shared/CognoSpeak_OLD/semi-automatic-man-labels"),
]

OLD_SYSTEMS = {
    "iva2", "iva3", "ivamem_fl_ct_gp_cp1", "ivamem_fl_ct_gp_cp2",
    "ivastroke_fl_ct_gp_cp1", "ivastroke_fl_ct_gp_cp2",
}

MATCH_THRESHOLD = 0.5
# Used when scanning for a turn that might BE a question -- permissive, since an assisting
# third party ("Oth") occasionally reads or relays part of the question too.
INTERVIEWER_RE = re.compile(r"^(int|ass|oth)", re.IGNORECASE)
# Used when deciding whether to STOP collecting a participant's answer -- stricter: only the
# actual interviewer/assistant starting a genuinely new question ends collection. "Oth" turns
# mid-answer (e.g. a family member saying "I'll hold it to yer") are bystander interjections,
# not a new question, so they're skipped rather than treated as the answer's end.
QUESTION_SOURCE_RE = re.compile(r"^(int|ass)", re.IGNORECASE)
PARTICIPANT_RE = re.compile(r"^pat", re.IGNORECASE)

# Same word-overlap method as notebooks/04_MatchFiles.ipynb -- content words (stopwords and
# words <=2 chars dropped) compared by Jaccard similarity.
STOPWORDS = set(
    "the a an to and or of in on for as you can that this it is are with your "
    "please have has been many much give as detail possible".split()
)


def normalize_prompt(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def content_words(normalized_text: str) -> set[str]:
    return {w for w in normalized_text.split() if w not in STOPWORDS and len(w) > 2}


def jaccard(a_words: set[str], b_words: set[str]) -> float:
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


def load_task_variants() -> dict[str, list[set[str]]]:
    """{task_code: [content-word sets, one per known phrasing]} for EVERY question code in
    PromptsMapping.csv (not just the 10 "interview" tasks) -- every task a participant might
    have answered: memory/finance/etc., but also fluency (sft/pft), picture description
    (pdc/pd1/pd2), paragraph reading (rdp), months-in-reverse (myr), how-to-make-tea (hct),
    future planning (fpl), and whatever else is in there. Content-based matching doesn't need
    to know which system a participant is on."""
    with PROMPTS_MAPPING_PATH.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    texts = defaultdict(set)
    for row in rows:
        code, text = row["PromptCode"], row["PromptText"]
        if code and text:
            texts[code].add(normalize_prompt(text))
    return {code: [content_words(t) for t in variants] for code, variants in texts.items()}


def load_old_system_participants() -> dict[str, dict[str, str]]:
    """{process_id: {"system", "full_path"}} for every participant on an old (non-cchat) system."""
    with DEMOGRAPHICS_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        system = row["system"].replace(" (inferred)", "").strip()
        if system in OLD_SYSTEMS:
            out[row["process_ID"]] = {"system": system, "full_path": row["full_path"]}
    return out


def find_label_file(full_path: str) -> Path | None:
    stem = Path(full_path).stem
    for label_dir in LABEL_DIRS:
        candidate = label_dir / f"{stem}.labels.txt"
        if candidate.exists():
            return candidate
    return None


def parse_label_file(path: Path) -> list[tuple[float, float, str, str]]:
    """Return [(start, end, speaker, text), ...] turns in order.

    Audacity-labels tags an INT turn as e.g. "INT q4:WHAT HAS BEEN..." or "INT gp:PLEASE
    READ..." -- the word(s) between the speaker and the colon are the old, already-distrusted
    positional code, not part of the speaker or the text, so they're discarded here rather than
    parsed as part of either. semi-automatic-man-labels has no such code ("Int1: Please
    read...") and is handled the same way: splitting on the first colon and taking the first
    whitespace-separated word before it as the speaker works for both.
    """
    turns = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        field = parts[2].strip()
        if ":" not in field:
            continue
        speaker_part, text = field.split(":", 1)
        speaker_words = speaker_part.strip().split()
        if not speaker_words:
            continue
        speaker, text = speaker_words[0], text.strip()
        if text:
            turns.append((float(parts[0]), float(parts[1]), speaker, text))
    return turns


def best_task_for_turn(text: str, task_variants: dict[str, list[set[str]]]) -> tuple[str | None, float]:
    """Argmax task + its score for one interviewer turn, scored against every task at once --
    needed because some tasks' phrasings overlap heavily (e.g. ltlj "tell me about your last
    job" vs lstj "tell me about your last job or your current job" score ~0.8 against each
    other), so independent per-task thresholding would double-assign the same turn to both;
    picking the single best-fitting task avoids that."""
    words = content_words(normalize_prompt(text))
    if not words:
        return None, 0.0
    best_task, best_score = None, 0.0
    for task, variants in task_variants.items():
        score = max((jaccard(words, v) for v in variants), default=0.0)
        if score > best_score:
            best_task, best_score = task, score
    return best_task, best_score


def extract_answers(turns: list[tuple[float, float, str, str]], task_variants: dict) -> dict[str, dict]:
    """{task_code: {"text", "start", "end"}} for every task confidently matched in this
    transcript. start = first PAT turn's start; end = start of the NEXT interviewer turn
    (not the last PAT turn's own end) -- same widening cut-mci-data.sh's timing index uses
    (end of question to start of next question), rather than cutting flush against the
    transcript's turn boundary. semi-automatic-man-labels' turn timestamps are approximate,
    and cutting tight against them was clipping trailing speech right at the edge, producing
    near-silent clips; the extra few seconds of natural pause/trail-off this captures instead
    fixes that without reverting to trusting the old, sometimes-wrong positional tags."""
    anchors = {}  # task -> turn index of its LAST matching interviewer turn (handles re-reads)
    for i, (start, end, speaker, text) in enumerate(turns):
        if not INTERVIEWER_RE.match(speaker):
            continue
        task, score = best_task_for_turn(text, task_variants)
        if task and score >= MATCH_THRESHOLD:
            anchors[task] = i

    answers = {}
    for task, anchor in anchors.items():
        parts, answer_start, answer_end = [], None, None
        for start, end, speaker, text in turns[anchor + 1:]:
            if QUESTION_SOURCE_RE.match(speaker):
                answer_end = start  # widen to the start of the next question
                break
            if PARTICIPANT_RE.match(speaker) and text:
                parts.append(text)
                if answer_start is None:
                    answer_start = start
                answer_end = end  # fallback if this is the session's last turn
            # anything else (e.g. "Oth" bystander interjections) is skipped: not part of the
            # answer, but not a new question either, so collection just continues past it
        answer = " ".join(parts).strip()
        if answer:
            answers[task] = {"text": answer, "start": answer_start, "end": answer_end}
    return answers


def main():
    task_variants = load_task_variants()
    participants = load_old_system_participants()

    counts = defaultdict(int)
    no_label_file = []
    timing_rows = []
    for process_id, info in sorted(participants.items()):
        full_path = info["full_path"]
        label_path = find_label_file(full_path)
        if label_path is None:
            no_label_file.append(process_id)
            continue

        turns = parse_label_file(label_path)
        answers = extract_answers(turns, task_variants)

        for task, result in answers.items():
            out_dir = TEXT_DIR / task.upper()
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{process_id}__{task.upper()}.txt").write_text(result["text"])
            counts[task] += 1
            timing_rows.append({
                "process_id": process_id,
                "system": info["system"],
                "task": task,
                "start": result["start"],
                "end": result["end"],
                "source_wav": full_path,
                "label_file": str(label_path),
            })

    TIMING_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TIMING_OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "process_id", "system", "task", "start", "end", "source_wav", "label_file",
        ])
        writer.writeheader()
        writer.writerows(sorted(timing_rows, key=lambda r: (r["process_id"], r["task"])))

    n_with_labels = len(participants) - len(no_label_file)
    print(f"{len(participants)} old-system participants, {len(no_label_file)} missing a label file")
    print("\nExtracted per task:")
    for task in sorted(task_variants):
        print(f"  {task}: {counts[task]}/{n_with_labels}")
    if no_label_file:
        print("\nNo label file found for:", no_label_file)
    print(f"\nSaved timing/mapping file: {TIMING_OUTPUT_PATH} ({len(timing_rows)} rows)")


if __name__ == "__main__":
    main()
