"""Scoring rules for phonemic fluency (letter P), rebuilt against the Leslie scoring criteria."""

"""
Import all the necessary libraries and modules
"""

import re
import pandas as pd
from pathlib import Path
import pycountry
from nltk.corpus import wordnet as wn
from nltk.corpus import words as _nltk_words
from nltk.stem import PorterStemmer
import pycountry

STEMMER = PorterStemmer()

# Reference-data paths are anchored to this file's location, not the process's
# working directory -- pipelines/*.py os.chdir() to the project root before
# importing this module, but the Shiny app doesn't, so a CWD-relative path
# would break there.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

"""
Filled pauses -> Leslie code 'F'. Discourse markers (well, so, right, yeah,
no, ...) are deliberately NOT included here -- they're scored through the
normal classification hierarchy instead.
"""

FILLED_PAUSES = {
    "er", "erm", "um", "umm", "uhm", "uh", "oh", "ah", "mm", "hmm", "hm", "err", "uhh", "uhhh", "ummm", "uhmmm",
    "phew", "pooh",
}


def is_filler(word: str) -> bool:
    """Check if word is a filled pause (Leslie code 'F')."""
    return word.lower().strip() in FILLED_PAUSES


"""
Load excel file with human names for proper noun detection
"""

_ONS_PATH = _PROJECT_ROOT / "data/raw/reference/babynames1996to2025.xlsx"

def _load_ons_names(path: Path = _ONS_PATH) -> set[str]:
    df_boy = pd.read_excel(path, sheet_name="Table_2")
    boy = df_boy.iloc[4:].copy()
    boy.columns = df_boy.iloc[3]
    boy = boy[["Name"]]

    df_girl = pd.read_excel(path, sheet_name="Table_1")
    girl = df_girl.iloc[4:].copy()
    girl.columns = df_girl.iloc[3]
    girl = girl[["Name"]]

    all_names = pd.concat([boy, girl], ignore_index=True).dropna(subset=["Name"])
    return set(all_names["Name"].astype(str).str.strip().str.lower())

NAME_SET = _load_ons_names()

# Words the WordNet/instance check would treat as "ordinary", but manual
# review confirmed they should be scored as names in this data.
FORCE_NAME_EXCEPTIONS = {
    "paddy",
    "pauline",
    "penelope",
    "peter",
    "phoebe"
}

"""
Load age-of-acquisition (AoA) norms (Kuperman et al., 2012, Behavior Research
Methods) for the age-of-acquisition metric. Mean AoA rating is in years.
"""

_AOA_PATH = _PROJECT_ROOT / "data/raw/reference/Kuperman2012.csv"

_AOA_WORD_COLS = {"word", "word_english"}
_AOA_RATING_COLS = {"rating.mean", "aoa_mean"}


def _load_aoa_norms(path: Path = _AOA_PATH) -> dict[str, float]:
    if not path.exists():
        return {}

    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)

    word_col = next((c for c in df.columns if str(c).strip().lower() in _AOA_WORD_COLS), None)
    aoa_col = next((c for c in df.columns if str(c).strip().lower() in _AOA_RATING_COLS), None)
    if word_col is None or aoa_col is None:
        raise ValueError(
            f"Expected a word column ({_AOA_WORD_COLS}) and an AoA rating column "
            f"({_AOA_RATING_COLS}) in {path}, found: {list(df.columns)}"
        )

    norms = df[[word_col, aoa_col]].dropna()
    norms[word_col] = norms[word_col].astype(str).str.strip().str.lower()
    return dict(zip(norms[word_col], norms[aoa_col].astype(float)))


AOA_NORMS = _load_aoa_norms()


def get_aoa(word: str) -> float | None:
    """Look up the mean age-of-acquisition rating (years) for a word.

    Returns None if the AoA norms file isn't present or the word isn't in it.
    """
    return AOA_NORMS.get(word.lower().strip())


"""
Load a UK-English IPA pronouncing dictionary (Montreal Forced Aligner's
english_uk_mfa lexicon) for phonemic cluster/switch analysis (Troyer et al.,
1997). Some words have several weighted pronunciation variants (e.g. "paint"
-> /p ej n t/ 0.99, /p ej n/ 0.86); we keep only the highest-probability
variant per word.
"""

_PHONE_DICT_PATH = _PROJECT_ROOT / "data/raw/reference/english_uk_mfa.dict"

# Vowel phones in this dictionary's IPA inventory (monophthongs + diphthongs).
# Everything else (including glides j/w, which only appear standalone as
# consonants -- diphthongs like "aj"/"aw"/"ej" are single tokens) is a
# consonant. Used to find the rime (nucleus + coda) of a word for rhyme
# matching.
VOWEL_PHONES = {
    "a", "aj", "aw", "e", "ej", "i", "iː", "ɐ", "ɑ", "ɑː", "ɒ", "ɒː", "ɔj",
    "ə", "əw", "ɛ", "ɛː", "ɜ", "ɜː", "ɪ", "ʉ", "ʉː", "ʊ",
}


def _load_phone_dict(path: Path = _PHONE_DICT_PATH) -> dict[str, list[str]]:
    if not path.exists():
        return {}

    best_prob: dict[str, float] = {}
    phones: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            word = fields[0].strip().lower()
            prob = float(fields[1]) if len(fields) > 2 else 1.0
            if word not in best_prob or prob > best_prob[word]:
                best_prob[word] = prob
                phones[word] = fields[-1].split()
    return phones


PHONE_DICT = _load_phone_dict()


def get_phones(word: str) -> list[str] | None:
    """Look up the (highest-probability) IPA phone sequence for a word.

    Returns None if the pronouncing dictionary isn't present or the word
    isn't in it.
    """
    return PHONE_DICT.get(word.lower().strip())


def _rime(phones: list[str]) -> list[str]:
    """The rime of a word: from its last vowel phone to the end."""
    for i in range(len(phones) - 1, -1, -1):
        if phones[i] in VOWEL_PHONES:
            return phones[i:]
    return phones


# A rime of just a bare, coda-less schwa isn't a meaningful rhyme anchor --
# this phone set doesn't mark stress, so we can't tell a genuinely rhyming
# stressed vowel from an incidental unstressed one, and a coda-less schwa is
# specifically the *reduced/unstressed* vowel. It's also extremely common:
# english_uk_mfa is non-rhotic, so any "-er"/"-or"/"-ar" ending (doctor,
# water, sugar, camera, banana, paper, ...) collapses to a trailing bare
# schwa. Without this guard, huge numbers of unrelated words would spuriously
# "rhyme" on nothing but that shared reduced ending.
_WEAK_RIME = ["ə"]


def is_phonemic_cluster_match(word_a: str, word_b: str) -> bool:
    """Check if two words are phonemically related (Troyer et al. cluster criteria).

    True if the words share their first two phonemes, rhyme (share a rime --
    last vowel phone through the end of the word, excluding a bare schwa
    rime -- see `_WEAK_RIME`), or are homonyms (identical full phone
    sequence). Conservative subset of the original four Troyer criteria:
    doesn't attempt the "differ by one phoneme" rule.

    Returns False (not merged into a cluster) if either word is missing from
    the pronouncing dictionary, rather than guessing.
    """
    phones_a = get_phones(word_a)
    phones_b = get_phones(word_b)
    if not phones_a or not phones_b:
        return False

    if phones_a == phones_b:
        return True

    if phones_a[:2] == phones_b[:2]:
        return True

    rime_a = _rime(phones_a)
    if rime_a == _WEAK_RIME:
        return False
    return rime_a == _rime(phones_b)


"""
Load a list with all country names for proper noun detection
"""

COUNTRY_SET = set(c.name.lower() for c in pycountry.countries)
SINGLE_WORD_COUNTRIES = {c for c in COUNTRY_SET if ' ' not in c}
MULTI_WORD_COUNTRIES = {c for c in COUNTRY_SET if ' ' in c}
_MAX_COUNTRY_PHRASE_LEN = max(len(c.split()) for c in MULTI_WORD_COUNTRIES)


"""
All scoring rules for phonemic fluency (letter P) are implemented as functions below.
"""

def starts_with_target_letter(word: str, target_letter: str = "p") -> bool:
    """Check if word starts with target letter (case-insensitive, orthographic).

    Parameters
    ----------
    word : str
        The word to check
    target_letter : str
        The target letter (default: "p")

    Returns
    -------
    bool
        True if word starts with target letter
    """
    word_lower = word.lower().strip()
    target_lower = target_letter.lower()

    if not word_lower or not word_lower[0].isalpha():
        return False

    return word_lower[0] == target_lower

_WORDS_CORPUS = {w.lower() for w in _nltk_words.words()}

def is_real_word(word: str) -> bool:
    word_lower = word.lower().strip()
    # Single letters get spurious hits in both WordNet (e.g. "p" = phosphorus
    # symbol) and the words corpus (single letters are listed as headwords)
    # even though a bare letter isn't a word attempt. Checked before either
    # dictionary lookup so it applies regardless of which one would have
    # matched.
    if len(word_lower) <= 1:
        return False
    # WordNet alone misses plenty of ordinary words (e.g. "per", "punchy");
    # nltk.corpus.words is a much larger flat dictionary (237k entries) that
    # catches those, at the cost of also accepting some obscure/archaic
    # entries and proper nouns -- accepted tradeoff, no frequency filtering.
    return len(wn.synsets(word_lower)) > 0 or word_lower in _WORDS_CORPUS

def has_ordinary_sense(word: str) -> bool:
    for s in wn.synsets(word.lower().strip()):
        if not s.instance_hypernyms():
            return True
    return False
        
def is_person_name(word: str) -> bool: 
    word_lower = word.lower().strip()
    if word_lower in FORCE_NAME_EXCEPTIONS:
        return True
    if word_lower not in NAME_SET:
        return False
    return not has_ordinary_sense(word_lower)

def is_country_name(word: str) -> bool:
    return word.lower().strip() in SINGLE_WORD_COUNTRIES

def is_repetition(word: str, previous_correct_words: list[str]) -> bool:
    """Check if word is an exact repeat of an earlier *correct* word.

    Only compares against previously-correct words (not errors), per the
    Leslie code definition: "R = repetition of previous correct word".

    Parameters
    ----------
    word : str
        Current word
    previous_correct_words : list[str]
        Words already classified 'C' earlier in the same trial

    Returns
    -------
    bool
        True if word exactly repeats an earlier correct word
    """
    word_lower = word.lower().strip()
    return word_lower in {w.lower().strip() for w in previous_correct_words}


def is_similar_form_or_meaning(word: str, previous_correct_words: list[str]) -> bool:
    """Check if word is a morphological variant or synonym of an earlier correct word.

    Two checks against each earlier correct word:
    - shared word stem (PorterStemmer) — catches inflectional/derivational
      variants, e.g. "paint" / "painting"
    - shared WordNet synset — catches exact synonyms, e.g. "pretty" / "pleasant"
      only if they co-occur in at least one sense's synonym set

    This is intentionally conservative: it will not catch broader semantic
    relatedness (hypernyms/hyponyms), only literal shared-synset synonymy.

    Parameters
    ----------
    word : str
        Current word
    previous_correct_words : list[str]
        Words already classified 'C' earlier in the same trial

    Returns
    -------
    bool
        True if word shares a stem or a WordNet synset with an earlier
        correct word (and is not identical to it — identical is 'R')
    """
    word_lower = word.lower().strip()
    word_stem = STEMMER.stem(word_lower)
    word_synsets = set(wn.synsets(word_lower))

    for prev in previous_correct_words:
        prev_lower = prev.lower().strip()
        if prev_lower == word_lower:
            continue

        if STEMMER.stem(prev_lower) == word_stem:
            return True

        if word_synsets and word_synsets & set(wn.synsets(prev_lower)):
            return True

    return False


def find_country_phrase_indices(words: list[str]) -> set[int]:
    """Indices in an ordered word list that are part of a multi-word country name."""
    matched = set()
    lowered = [w.lower().strip() for w in words]
    for n in range(2, _MAX_COUNTRY_PHRASE_LEN + 1):
        for i in range(len(lowered) - n + 1):
            span = ' '.join(lowered[i:i + n])
            if span in MULTI_WORD_COUNTRIES:
                matched.update(range(i, i + n))
    return matched

#check some of the output files 

