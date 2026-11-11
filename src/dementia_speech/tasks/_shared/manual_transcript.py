"""Parsing for this project's manually-transcribed CognoSpeak transcripts
(data/raw/process/<TASK>/) -- shared across tasks because the transcription
convention itself isn't task-specific: speaker turns tagged at the start of a
line ("Pat:", "Oth"/"Oth1"/"Oth2", "Int:"), and transcriber annotations
(pause durations, non-speech sounds, the task buzzer, unclear speech) in
parentheses or angle brackets. Originally written for SFT_new/scoring.py;
moved here once picture_description/scoring.py needed the exact same parsing
for Cookie Theft's own manual transcripts (data/raw/process/CTD/) rather than
duplicating the regexes a second time.
"""
import re

# A transcript with no tag at all is treated as one continuous Pat turn by
# parse_speaker_turns() -- true for a meaningful minority of files (144/400
# for SFT), not an edge case to special-case away.
_SPEAKER_TURN_RE = re.compile(r"(?m)^\s*(Pat|Oth\d*|Int)\s*:\s*")

# Parenthetical/angle-bracket annotations are always noise in these
# transcripts -- pause durations ("(3 seconds)"), the task buzzer
# ("(buzzer sounds)"), unclear speech ("(?)"), and non-speech sounds
# ("(sighs)", "(laughs)", "(tuts)", ...), plus a couple of "<laugh>"/
# "<lugh>" tags. Stripped wholesale rather than enumerated by phrase, since
# every instance found across all 400 SFT transcripts was non-content.
_ANNOTATION_RE = re.compile(r"\([^)]*\)|<[^>]*>")


def parse_speaker_turns(text: str) -> list[tuple[str, str]]:
    """Split a transcript into [(speaker, turn_text), ...] in order.

    Speaker is normalised to its base tag ("Oth2" -> "Oth") -- callers only
    care about Pat vs. not-Pat, not which bystander/assistant spoke. A
    transcript with no speaker tag at all is returned as a single
    ("Pat", text) turn. Several files start with untagged speech and only
    pick up a tag later (e.g. an "Oth:" interjection midway through) -- any
    text before the first tag is the patient's own, so it becomes a
    leading ("Pat", ...) turn rather than being silently dropped.
    """
    matches = list(_SPEAKER_TURN_RE.finditer(text))
    if not matches:
        return [("Pat", text)]

    turns = []
    if matches[0].start() > 0:
        turns.append(("Pat", text[:matches[0].start()]))
    for i, m in enumerate(matches):
        speaker = "Oth" if m.group(1).startswith("Oth") else m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        turns.append((speaker, text[start:end]))
    return turns


def strip_annotations(text: str) -> str:
    """Remove parenthetical/angle-bracket annotations -- see _ANNOTATION_RE."""
    return _ANNOTATION_RE.sub(" ", text)


def extract_patient_speech(raw_text: str) -> str:
    """The participant's own speech only, annotations stripped, turns joined
    with a space -- bystander/assistant/interviewer turns (Oth/Int) are
    excluded entirely, not just flagged, same as SFT_new.scoring.score_trial.
    """
    turns = parse_speaker_turns(raw_text)
    return " ".join(
        strip_annotations(text) for speaker, text in turns if speaker == "Pat"
    )
