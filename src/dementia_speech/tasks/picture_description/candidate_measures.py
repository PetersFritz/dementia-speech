"""Candidate extra measures for the Cookie Theft tab, computed for inspection
before deciding which ones to keep (see docs/cookie_theft_scoring.md once that
decision is made). These are additive to the existing IU/graph/lexical
measures in scoring.py, graph.py and lexical.py -- nothing here replaces them.

Several of these are crude proxies, not validated clinical measures, because
the underlying data is limited:
- Pause/disfluency annotations ("(3 seconds)", filled pauses) exist in the
  manual transcripts but only ~58% of CTD transcripts use that convention
  (see has_pause_annotations) -- missing is NOT the same as "no pauses".
- Speech duration comes from the raw .wav via energy-based silence trimming
  (librosa.effects.split), not forced alignment or diarization. A small
  minority of CTD recordings (~4-5%) include interviewer/bystander turns
  (Oth/Int) in the transcript -- since the audio isn't diarized, any such
  speech on the recording inflates speech_duration_seconds for those
  participants. Flag before trusting duration-derived rates for a given case.
- grammatical_accuracy_proxy and the syntactic-complexity metrics are spaCy
  dependency-parse heuristics, not a validated grammaticality/error detector
  (no grammar-checking tool is available in this environment). They flag
  clause *form* (is there a subject + finite verb), not correctness.
- Deliberately NOT computed here: semantic errors, referential clarity.
  Both need either manual annotation or an LLM-based judgement against the
  picture's ground truth, which is a separate scoping decision, not a
  parameter choice -- left out rather than faked with a weak proxy.
"""
import re
from collections import Counter
from functools import lru_cache

import nltk
import spacy

from dementia_speech.tasks._shared.manual_transcript import parse_speaker_turns
from dementia_speech.tasks.picture_description.graph import get_quadrant
from dementia_speech.tasks.picture_description.lexical import _AUXILIARY_VERBS
from dementia_speech.tasks.picture_description.scoring import tokenize_words

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _NLP


# ── Syllables ─────────────────────────────────────────────────────────────

_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")


@lru_cache(maxsize=1)
def _cmudict():
    from nltk.corpus import cmudict
    return cmudict.dict()


def count_syllables(word: str) -> int:
    """Syllable count via CMU Pronouncing Dictionary, falling back to a
    vowel-group heuristic for out-of-dictionary words (informal spellings,
    names) -- crude but keeps every word counted rather than dropped."""
    word = word.lower()
    entry = _cmudict().get(word)
    if entry:
        return max(1, sum(1 for phoneme in entry[0] if phoneme[-1].isdigit()))
    groups = _VOWEL_GROUP_RE.findall(word)
    n = len(groups)
    if word.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def compute_syllable_metrics(transcript: str) -> dict:
    words = tokenize_words(transcript)
    return {"total_syllables": sum(count_syllables(w) for w in words)}


# ── Utterances ───────────────────────────────────────────────────────────

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_HAS_PUNCTUATION_RE = re.compile(r"[.!?]")


def compute_utterance_metrics(transcript: str, total_words: int) -> dict:
    """Utterance count and MLU from sentence-ending punctuation. Transcripts
    with no punctuation at all (run-on dictation) collapse to one utterance
    -- flagged via has_sentence_punctuation so those rows can be excluded
    rather than silently treated as "one long utterance"."""
    has_punctuation = bool(_HAS_PUNCTUATION_RE.search(transcript))
    utterances = [u.strip() for u in _SENTENCE_SPLIT_RE.split(transcript) if u.strip()]
    n = len(utterances)
    return {
        "num_utterances": n,
        "mean_length_utterance": round(total_words / n, 2) if n else None,
        "has_sentence_punctuation": has_punctuation,
    }


# ── Region / left-right ─────────────────────────────────────────────────────

def compute_region_metrics(iu_results: dict, coordinates_raw: dict, img_width: int, img_height: int) -> dict:
    """Presence-based spatial measures: which quadrant(s) of the picture the
    mentioned content units fall in, and the left/right split. Distinct from
    graph.py's avg_x/spatial_left_bias, which are order-of-mention measures
    on the full repeat-aware sequence -- these are simple presence counts."""
    mentioned = [iu for iu, v in iu_results.items() if v["present"]]
    region_counts = Counter()
    left = right = 0
    for iu in mentioned:
        if iu not in coordinates_raw:
            continue
        x, y = coordinates_raw[iu]
        region_counts[get_quadrant((x, y), img_width, img_height)] += 1
        if x < img_width / 2:
            left += 1
        else:
            right += 1

    all_quadrants = {
        get_quadrant(coord, img_width, img_height) for coord in coordinates_raw.values()
    }
    mentioned_quadrants = set(region_counts.keys())
    region_coverage = (
        round(len(mentioned_quadrants) / len(all_quadrants), 3) if all_quadrants else None
    )

    return {
        "content_units_left": left,
        "content_units_right": right,
        "left_right_cu_ratio": round(left / right, 3) if right else None,
        "region_coverage": region_coverage,
        "content_units_by_region": dict(sorted(region_counts.items())),
    }


# ── Grammar / syntactic complexity (spaCy dependency parse) ────────────────

_SUBORDINATION_DEPS = {"advcl", "ccomp", "relcl", "acl"}
_SUBJECT_DEPS = {"nsubj", "nsubjpass", "expl"}


def _max_dep_depth(sent) -> int:
    def depth(tok):
        d = 0
        node = tok
        while node.head != node and d < 50:
            d += 1
            node = node.head
        return d
    return max((depth(t) for t in sent), default=0)


def compute_grammar_complexity_metrics(transcript: str) -> dict:
    """Crude spaCy-parse proxies, NOT a validated grammar checker -- see
    module docstring. grammatical_accuracy_proxy = fraction of sentences
    with a finite-verb root plus an overt subject (clause *form*, not
    correctness)."""
    doc = _get_nlp()(transcript)
    sents = [s for s in doc.sents if any(not t.is_space for t in s)]
    if not sents:
        return {
            "grammatical_accuracy_proxy": None,
            "subordinate_clause_rate": None,
            "mean_dependency_depth": None,
        }

    well_formed = 0
    subordinate = 0
    depths = []
    for sent in sents:
        root = sent.root
        has_subject = any(tok.dep_ in _SUBJECT_DEPS for tok in sent)
        if root.pos_ in ("VERB", "AUX") and has_subject:
            well_formed += 1
        if any(tok.dep_ in _SUBORDINATION_DEPS for tok in sent):
            subordinate += 1
        depths.append(_max_dep_depth(sent))

    n = len(sents)
    return {
        "grammatical_accuracy_proxy": round(well_formed / n, 3),
        "subordinate_clause_rate": round(subordinate / n, 3),
        "mean_dependency_depth": round(sum(depths) / n, 2),
    }


# ── Repetitions / revisions ─────────────────────────────────────────────────

_IMMEDIATE_REPEAT_RE = re.compile(r"\b(\w+)\b[\s,]+\1\b")


def compute_repetition_metrics(transcript: str, total_words: int) -> dict:
    """Immediate word repetition ("the the") only -- a reliable, convention-
    independent signal. Does NOT catch revision patterns some transcribers
    mark with a comma ("drop, um is getting") since that convention isn't
    used consistently across transcribers, so would undercount for files
    that don't use it rather than giving a comparable rate."""
    hits = len(_IMMEDIATE_REPEAT_RE.findall(transcript))
    return {
        "immediate_word_repetitions": hits,
        "repetition_rate_per_100_words": round(100 * hits / total_words, 2) if total_words else None,
    }


# ── Pauses (from transcriber pause annotations) and filled pauses ──────────

_PAUSE_RE = re.compile(r"\((\d+)\s*seconds?\)")
_FILLED_PAUSE_TERMS = {"um", "uh", "er", "erm", "mm", "hmm"}


def compute_text_pause_metrics(raw_text: str) -> dict:
    """Pause counts/durations from transcriber annotations like "(3
    seconds)", Pat turns only. Only ~58% of CTD transcripts use this
    convention -- has_pause_annotations=False means "not annotated", not
    "no pauses"; don't treat it as zero."""
    pat_text = " ".join(text for speaker, text in parse_speaker_turns(raw_text) if speaker == "Pat")
    seconds = [int(m) for m in _PAUSE_RE.findall(pat_text)]
    if not seconds:
        return {"pause_count": None, "total_pause_seconds": None, "has_pause_annotations": False}
    return {
        "pause_count": len(seconds),
        "total_pause_seconds": sum(seconds),
        "has_pause_annotations": True,
    }


def compute_filled_pause_metrics(transcript: str, total_words: int) -> dict:
    """Filled-pause ("um"/"uh"/"er"/"mm") rate -- a word-finding-difficulty
    proxy independent of transcription convention, unlike the pause-duration
    annotations above (those come from the transcriber; these are spoken
    words already in the cleaned transcript)."""
    words = tokenize_words(transcript)
    hits = sum(1 for w in words if w in _FILLED_PAUSE_TERMS)
    return {
        "filled_pause_count": hits,
        "filled_pause_rate_per_100_words": round(100 * hits / total_words, 2) if total_words else None,
    }


# ── Noun/verb ratio ──────────────────────────────────────────────────────────

_NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
_VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}


def compute_noun_verb_ratio(transcript: str) -> dict:
    words = tokenize_words(transcript)
    if not words:
        return {"noun_count": None, "verb_count": None, "noun_verb_ratio": None}
    tagged = nltk.pos_tag(words)
    nouns = sum(1 for _, tag in tagged if tag in _NOUN_TAGS)
    verbs = sum(1 for word, tag in tagged if tag in _VERB_TAGS and word not in _AUXILIARY_VERBS)
    return {
        "noun_count": nouns,
        "verb_count": verbs,
        "noun_verb_ratio": round(nouns / verbs, 3) if verbs else None,
    }


# ── Audio-derived speech duration ───────────────────────────────────────────

def _rms_split(y, sr, top_db: float = 30.0, frame_ms: float = 20.0, hop_ms: float = 10.0):
    """Non-silent (start, end) sample-index intervals by frame-wise RMS energy
    vs. a threshold `top_db` below the signal's peak frame RMS -- the same
    approach as librosa.effects.split, reimplemented directly on top of
    soundfile/numpy because librosa's import chain is broken in this conda
    env (pkg_resources -> jaraco -> backports.tarfile, unrelated to this
    project -- see candidate_measures module usage notes)."""
    import numpy as np

    frame_len = max(1, int(sr * frame_ms / 1000))
    hop_len = max(1, int(sr * hop_ms / 1000))
    n_frames = max(0, (len(y) - frame_len) // hop_len + 1)
    if n_frames <= 0:
        return []

    rms = np.empty(n_frames)
    for i in range(n_frames):
        start = i * hop_len
        frame = y[start:start + frame_len]
        rms[i] = np.sqrt(np.mean(frame ** 2)) if frame.size else 0.0

    peak = rms.max()
    if peak <= 0:
        return []
    threshold = peak * (10 ** (-top_db / 20))
    is_speech = rms >= threshold

    intervals = []
    frame_start = None
    for i, speech in enumerate(is_speech):
        if speech and frame_start is None:
            frame_start = i
        elif not speech and frame_start is not None:
            intervals.append((frame_start * hop_len, i * hop_len + frame_len))
            frame_start = None
    if frame_start is not None:
        intervals.append((frame_start * hop_len, len(y)))
    return intervals


def compute_audio_duration_metrics(wav_path, top_db: float = 30.0) -> dict:
    """Participant speech duration via energy-based silence trimming, not
    forced alignment or diarization -- see module docstring for the
    interviewer-contamination caveat on the small minority of recordings
    with Oth/Int turns."""
    import soundfile as sf

    try:
        y, sr = sf.read(str(wav_path), always_2d=False)
    except Exception as exc:
        return {
            "speech_duration_seconds": None,
            "total_audio_duration_seconds": None,
            "audio_load_error": str(exc),
        }
    if y.ndim > 1:
        y = y.mean(axis=1)
    total_duration = len(y) / sr
    intervals = _rms_split(y, sr, top_db=top_db)
    speech_duration = sum(end - start for start, end in intervals) / sr
    return {
        "speech_duration_seconds": round(float(speech_duration), 2),
        "total_audio_duration_seconds": round(float(total_duration), 2),
    }
