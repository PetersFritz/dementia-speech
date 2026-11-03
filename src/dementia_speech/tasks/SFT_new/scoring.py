"""Score semantic (category) fluency trials from manually-transcribed free text.

Unlike PFT_new, there's no word-level TextGrid for SFT -- only a manually
transcribed free-text transcript per participant (data/raw/process/SFT/),
with inline speaker tags, pause durations, and non-speech sounds annotated
by the transcriber. This module parses that format directly rather than
iterating aligned word tiers: splits speaker turns, strips annotations, and
tokenizes the patient's own speech into candidate animal-name items.

Classification codes (deliberately a smaller set than PFT_new's Leslie
codes -- see the SFT scoring discussion: no nonword/proper-noun/similarity
codes, since the animal list is itself the only dictionary that matters
here):
  C   -- correct, first mention of a known animal
  R   -- repetition of an earlier 'C' animal
  OTH -- intrusion (not a known animal)
  F   -- filled pause (um, er, ...)
"""
import re

import pandas as pd

from dementia_speech.tasks.PFT_new.rules import is_filler, is_repetition
from dementia_speech.tasks.SFT_new.rules import canonical_key, get_aoa, is_animal, normalize_word

_CLASSIFICATION_COLUMN = "classification"

# Speaker turns are tagged at the start of a line, e.g. "Pat: Cat, dog..." or
# "Oth: Keep going...". A transcript with no tag at all (144/400 files) is
# treated as one continuous Pat turn by parse_speaker_turns().
_SPEAKER_TURN_RE = re.compile(r"(?m)^\s*(Pat|Oth\d*|Int)\s*:\s*")

# Parenthetical/angle-bracket annotations are always noise in these
# transcripts -- pause durations ("(3 seconds)"), the task buzzer
# ("(buzzer sounds)"), unclear speech ("(?)"), and non-speech sounds
# ("(sighs)", "(laughs)", "(tuts)", ...), plus a couple of "<laugh>"/
# "<lugh>" tags. Stripped wholesale rather than enumerated by phrase, since
# every instance found across all 400 transcripts was non-content.
_ANNOTATION_RE = re.compile(r"\([^)]*\)|<[^>]*>")

# Segments are split on sentence/list punctuation. ";" shows up occasionally
# in place of a comma (e.g. "shark; oh I've said that.").
_SEGMENT_SPLIT_RE = re.compile(r"[,.;?!]+")

# Pure connective/grammatical glue words -- dropped silently rather than
# scored as an intrusion. "camel and sheep" (no Oxford comma) is two correct
# items joined by "and", not three items including "and" as an off-target
# word. Not the same thing as a filled pause (um/er/...), which IS scored
# (as 'F') via is_filler().
_CONNECTIVE_STOPWORDS = {
    "and", "or", "a", "an", "the", "to", "of", "in", "on", "for", "as",
    "is", "are", "it", "that", "this",
}


def parse_speaker_turns(text: str) -> list[tuple[str, str]]:
    """Split a transcript into [(speaker, turn_text), ...] in order.

    Speaker is normalised to its base tag ("Oth2" -> "Oth") -- scoring only
    cares about Pat vs. not-Pat, not which bystander/assistant spoke. A
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


def tokenize_pat_turn(text: str) -> list[str]:
    """Split a (annotation-stripped) turn into raw candidate segments."""
    return [s.strip() for s in _SEGMENT_SPLIT_RE.split(text) if s.strip()]


# How many consecutive words a single animal name can span (e.g. "guinea
# pig", "polar bear", "red admiral butterfly") -- the scheme's own compound
# names are written as one concatenated token, so there's no way to know
# this except by trying.
_MAX_PHRASE_WORDS = 3


def _classify_segment_words(
    words: list[str], previous_correct: list[str]
) -> list[tuple[str, str]]:
    """Classify one segment's remaining words (filler/stopwords already
    removed) into [(raw_word_or_phrase, classification), ...].

    If NONE of the words are animals even individually, the whole phrase is
    scored as a single intrusion (e.g. a narration aside like "and through
    Africa we went" is one off-target utterance, not five). Otherwise,
    scans left to right taking the longest animal-name match at each
    position (up to _MAX_PHRASE_WORDS words) before falling back to a
    single word -- needed because a whole-phrase-or-split-every-word
    approach breaks on e.g. "mouse guinea pig" (two items, no comma between
    them, one of which is itself two words): greedy matching gets "mouse"
    then "guinea pig", instead of wrongly splitting "guinea" off on its own.
    """
    if not words:
        return []

    if not any(is_animal(w) for w in words):
        return [(" ".join(words), "OTH")]

    results = []
    i, n = 0, len(words)
    while i < n:
        for window in range(min(_MAX_PHRASE_WORDS, n - i), 0, -1):
            phrase = " ".join(words[i:i + window])
            if is_animal(phrase):
                key = canonical_key(phrase)
                classification = "R" if is_repetition(key, previous_correct) else "C"
                if classification == "C":
                    previous_correct.append(key)
                results.append((phrase, classification))
                i += window
                break
        else:
            results.append((words[i], "OTH"))
            i += 1
    return results


def score_trial(raw_text: str) -> tuple[pd.DataFrame, int]:
    """Score a semantic fluency (SFT) trial from its manually-transcribed text.

    Returns (scored_df, oth_turn_count): scored_df has one row per scored
    item ('word' and 'classification' columns). oth_turn_count is how many
    non-Pat (bystander/assistant) turns occurred -- tracked as metadata,
    but that speech is excluded entirely from scoring, not just flagged.
    """
    turns = parse_speaker_turns(raw_text)
    oth_turn_count = sum(1 for speaker, _ in turns if speaker != "Pat")

    rows = []
    previous_correct: list[str] = []
    for speaker, turn_text in turns:
        if speaker != "Pat":
            continue

        cleaned = strip_annotations(turn_text)
        for segment in tokenize_pat_turn(cleaned):
            remaining = []
            for word in segment.split():
                norm = normalize_word(word)
                if not norm or norm in _CONNECTIVE_STOPWORDS:
                    continue
                if is_filler(word):
                    rows.append({"word": word, _CLASSIFICATION_COLUMN: "F", "aoa": get_aoa(word)})
                    continue
                remaining.append(word)

            for word_or_phrase, classification in _classify_segment_words(remaining, previous_correct):
                rows.append({
                    "word": word_or_phrase,
                    _CLASSIFICATION_COLUMN: classification,
                    # None for any multi-word phrase (e.g. "guinea pig") --
                    # the Kuperman norms are single-word only.
                    "aoa": get_aoa(word_or_phrase),
                })

    scored_df = pd.DataFrame(rows, columns=["word", _CLASSIFICATION_COLUMN, "aoa"])
    return scored_df, oth_turn_count


def compute_metrics(scored_df: pd.DataFrame, oth_turn_count: int) -> dict:
    """Summarize classification counts/rates for one participant's trial."""
    if len(scored_df) == 0:
        return {
            "total_items": 0, "C": 0, "R": 0, "OTH": 0, "F": 0,
            "R_rate": None, "OTH_rate": None, "F_rate": None,
            "mean_aoa_correct": None, "aoa_n": 0,
            "oth_turn_count": oth_turn_count,
        }

    counts = scored_df[_CLASSIFICATION_COLUMN].value_counts().to_dict()
    total = len(scored_df)
    c, r, oth, f = (counts.get(code, 0) for code in ("C", "R", "OTH", "F"))

    # Mean age-of-acquisition (years, Kuperman et al. 2012) of correct
    # animal names with a known rating; aoa_n is how many 'C' items matched
    # the norms (multi-word names like "guinea pig" never do), since a low
    # aoa_n makes the mean unreliable.
    correct_aoa = scored_df.loc[scored_df[_CLASSIFICATION_COLUMN] == "C", "aoa"].dropna()

    return {
        "total_items": total,
        "C": c, "R": r, "OTH": oth, "F": f,
        "R_rate": round(100 * r / total, 2),
        "OTH_rate": round(100 * oth / total, 2),
        "F_rate": round(100 * f / total, 2),
        "mean_aoa_correct": float(correct_aoa.mean()) if len(correct_aoa) else None,
        "aoa_n": int(len(correct_aoa)),
        "oth_turn_count": oth_turn_count,
    }
