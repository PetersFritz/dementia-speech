"""Score semantic (category) fluency trials from manually-transcribed free text.

Unlike PFT_new, there's no word-level TextGrid for SFT -- only a manually
transcribed free-text transcript per participant (data/raw/process/SFT/),
with inline speaker tags, pause durations, and non-speech sounds annotated
by the transcriber. This module parses that format directly rather than
iterating aligned word tiers: splits speaker turns, strips annotations, and
tokenizes the patient's own speech into candidate animal-name items.

Classification codes (a smaller set than PFT_new's Leslie codes -- no
proper-noun or similarity codes, since the animal list is itself the only
dictionary that matters here, and there's no clean animal-naming analog for
"morphological/synonym variant of an earlier correct word." A nonword code
*was* added (2026-11-17), splitting what used to be a single catch-all
"intrusion" bucket -- sampling it showed a real mix of genuine real-word
intrusions ("car", narration asides like "I can't think") and fragments/
false starts that were never real word attempts ("alpac", "de-de", "ka"),
the same distinction PFT_new's N vs. OTH draws):
  C   -- correct, first mention of a known animal
  R   -- repetition of an earlier 'C' animal
  N   -- nonword: not a known animal, and not a real English word either
         (fragment/false start -- see PFT_new.rules.is_real_word)
  OTH -- intrusion: not a known animal, but a real word/narration aside
  F   -- filled pause (um, er, ...)
"""
import re

import pandas as pd

from dementia_speech.tasks._shared.manual_transcript import parse_speaker_turns, strip_annotations
from dementia_speech.tasks.PFT_new.rules import is_filler, is_real_word, is_repetition
from dementia_speech.tasks.SFT_new.clustering import compute_cluster_metrics
from dementia_speech.tasks.SFT_new.rules import canonical_key, get_aoa, is_animal, normalize_word

_CLASSIFICATION_COLUMN = "classification"

# Segments are split on sentence/list punctuation. ";" shows up occasionally
# in place of a comma (e.g. "shark; oh I've said that.").
_SEGMENT_SPLIT_RE = re.compile(r"[,.;?!]+")

# is_real_word()'s dictionaries (WordNet + nltk's words corpus) don't cover
# contractions or possessives at all -- "can't", "it's", "cetti's" all fail
# it even though they're ordinary English, not fragments/false starts. Found
# while sampling this task's narration-heavy speech for the new 'N' code
# (contractions are rare in PFT's rapid-fire word lists, so this never
# surfaced there). A word ending in one of these suffixes is accepted as a
# real word regardless of its stem -- the clitic itself is what makes it
# grammatical ("cetti's", a possessive of a proper noun, is just as
# legitimate as "dog's"), unlike a genuine fragment such as "ka" or "de-de".
_CONTRACTION_SUFFIXES = ("'s", "'t", "'re", "'ve", "'ll", "'d", "'m")


def _is_real_word_or_contraction(word: str) -> bool:
    normalized = word.replace("’", "'")
    if normalized.lower().endswith(_CONTRACTION_SUFFIXES):
        return True
    return is_real_word(word)

# Pure connective/grammatical glue words -- dropped silently rather than
# scored as an intrusion. "camel and sheep" (no Oxford comma) is two correct
# items joined by "and", not three items including "and" as an off-target
# word. Not the same thing as a filled pause (um/er/...), which IS scored
# (as 'F') via is_filler().
_CONNECTIVE_STOPWORDS = {
    "and", "or", "a", "an", "the", "to", "of", "in", "on", "for", "as",
    "is", "are", "it", "that", "this",
}


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
    Africa we went" is one off-target utterance, not five) -- always 'OTH',
    never 'N': a multi-word phrase is narration, not a single failed word-
    finding attempt, and a single-word segment that isn't a real word (e.g.
    "ka") gets 'N' via the same check used below. Otherwise, scans left to
    right taking the longest animal-name match at each position (up to
    _MAX_PHRASE_WORDS words) before falling back to a single word -- needed
    because a whole-phrase-or-split-every-word approach breaks on e.g.
    "mouse guinea pig" (two items, no comma between them, one of which is
    itself two words): greedy matching gets "mouse" then "guinea pig",
    instead of wrongly splitting "guinea" off on its own. A non-matching
    single word at any position is 'N' if it isn't a real English word
    either (fragment/false start), 'OTH' otherwise (genuine intrusion).
    """
    if not words:
        return []

    if not any(is_animal(w) for w in words):
        phrase = " ".join(words)
        if len(words) == 1 and not _is_real_word_or_contraction(words[0]):
            return [(phrase, "N")]
        return [(phrase, "OTH")]

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
            word = words[i]
            results.append((word, "N" if not _is_real_word_or_contraction(word) else "OTH"))
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
            "total_items": 0, "C": 0, "R": 0, "OTH": 0, "F": 0, "N": 0,
            "errors": 0, "error_rate": None,
            "R_rate": None, "OTH_rate": None, "F_rate": None, "N_rate": None,
            "mean_aoa_correct": None, "aoa_n": 0,
            "n_clusters": 0, "n_switches": 0,
            "mean_cluster_size": None, "largest_cluster_size": 0,
            "oth_turn_count": oth_turn_count,
        }

    counts = scored_df[_CLASSIFICATION_COLUMN].value_counts().to_dict()
    total = len(scored_df)
    c, r, oth, f, n = (counts.get(code, 0) for code in ("C", "R", "OTH", "F", "N"))

    # Mean age-of-acquisition (years, Kuperman et al. 2012) of correct
    # animal names with a known rating; aoa_n is how many 'C' items matched
    # the norms (multi-word names like "guinea pig" never do), since a low
    # aoa_n makes the mean unreliable.
    correct_aoa = scored_df.loc[scored_df[_CLASSIFICATION_COLUMN] == "C", "aoa"].dropna()

    # Semantic cluster/switch analysis (Troyer et al. 1997 / Zemla et al.
    # 2020) over the ordered sequence of correct animal names -- see
    # SFT_new/clustering.py.
    correct_words_ordered = scored_df.loc[scored_df[_CLASSIFICATION_COLUMN] == "C", "word"].tolist()
    cluster_metrics = compute_cluster_metrics(correct_words_ordered)

    return {
        "total_items": total,
        "C": c, "R": r, "OTH": oth, "F": f, "N": n,
        # Same definition as PFT_new's errors/error_rate (see its docstring):
        # everything except correct responses and filled pauses (R + OTH + N
        # here -- SFT has no PN/SIM codes).
        "errors": r + oth + n,
        "error_rate": round(100 * (r + oth + n) / total, 2),
        "R_rate": round(100 * r / total, 2),
        "OTH_rate": round(100 * oth / total, 2),
        "F_rate": round(100 * f / total, 2),
        "N_rate": round(100 * n / total, 2),
        "mean_aoa_correct": float(correct_aoa.mean()) if len(correct_aoa) else None,
        "aoa_n": int(len(correct_aoa)),
        **cluster_metrics,
        "oth_turn_count": oth_turn_count,
    }
