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

_ONS_PATH = Path("data/raw/reference/babynames1996to2025.xlsx")

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

