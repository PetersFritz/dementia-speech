"""Scoring rules for phonemic fluency (letter P)."""
import re
from typing import Set
from nltk.corpus import wordnet
try:
    from nltk.stem import PorterStemmer
    STEMMER = PorterStemmer()
    HAS_STEMMER = True
except ImportError:
    HAS_STEMMER = False
    STEMMER = None

def clean_word(word: str) -> str:
    """Strip punctuation and whitespace from a word.
    
    e.g. "park," -> "park", "pet." -> "pet"
    """
    return re.sub(r'[^\w]', '', word).strip().lower()

def starts_with_target_letter(word: str, target_letter: str = "p") -> bool:
    """Check if word starts with target letter (case-insensitive, orthographic).
    
    For letter P, we accept both 'p' and 'ph' starts since 'ph' is pronounced /f/
    but in standard phonemic fluency scoring for P, we typically accept it.
    
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
    
    # Handle empty or invalid input
    if not word_lower or not word_lower[0].isalpha():
        return False
    
    return word_lower[0] == target_lower

def is_nonword(word: str) -> bool:
    """Check if word is not in WordNet.
    
    Parameters
    ----------
    word : str
        Word to check
        
    Returns
    -------
    bool
        True if word is NOT in WordNet (i.e. not a real word)
    """
    return len(wordnet.synsets(word.lower().strip())) == 0


def is_repetition(word: str, previous_words: list[str]) -> bool:
    """Check if word is an exact repetition of a previous word.
    
    Case-insensitive comparison.
    
    Parameters
    ----------
    word : str
        Current word
    previous_words : list[str]
        All words produced before this one in the trial
        
    Returns
    -------
    bool
        True if word is an exact repetition
    """
    word_lower = word.lower().strip()
    previous_lower = [w.lower().strip() for w in previous_words]
    return word_lower in previous_lower


def is_variant(word: str, previous_words: list[str]) -> bool:
    """Check if word is a morphological variant of a previous word.
    
    Uses stemming to detect variants like cat/cats, run/running.
    
    Parameters
    ----------
    word : str
        Current word
    previous_words : list[str]
        All words produced before this one in the trial
        
    Returns
    -------
    bool
        True if word shares a stem with a previous word
    """
    if not HAS_STEMMER:
        # Fallback: simple suffix stripping if NLTK not available
        return _simple_variant_check(word, previous_words)
    
    word_stem = STEMMER.stem(word.lower().strip())
    
    for prev_word in previous_words:
        prev_stem = STEMMER.stem(prev_word.lower().strip())
        if word_stem == prev_stem and word.lower() != prev_word.lower():
            return True
    
    return False


def _simple_variant_check(word: str, previous_words: list[str]) -> bool:
    """Simple variant detection without NLTK (fallback).
    
    Strips common suffixes and checks for matches.
    """
    suffixes = ['s', 'es', 'ed', 'ing', 'er', 'est', 'ly']
    
    def strip_suffix(w: str) -> str:
        w_lower = w.lower().strip()
        for suffix in suffixes:
            if w_lower.endswith(suffix) and len(w_lower) > len(suffix) + 2:
                return w_lower[:-len(suffix)]
        return w_lower
    
    word_base = strip_suffix(word)
    
    for prev_word in previous_words:
        prev_base = strip_suffix(prev_word)
        if word_base == prev_base and word.lower() != prev_word.lower():
            return True
    
    return False


def classify_word(
    word: str,
    previous_words: list[str],
    target_letter: str = "p"
) -> str:
    """Classify a word according to phonemic fluency scoring rules.
    
    Classification hierarchy (checked in order):
    1. repetition - exact match to previous word
    2. variant - morphological variant of previous word
    3. intrusion - doesn't start with target letter
    4. correct - valid word starting with target letter
    
    Parameters
    ----------
    word : str
        Word to classify
    previous_words : list[str]
        All words produced before this one
    target_letter : str
        Target letter for the trial (default: "p")
        
    Returns
    -------
    str
        Classification: 'correct', 'intrusion', 'repetition', or 'variant'
    """
    word = clean_word(word)
    previous_words = [clean_word(w) for w in previous_words]
    
    if is_repetition(word, previous_words):
        return 'repetition'
    
    if is_variant(word, previous_words):
        return 'variant'
    
    if not starts_with_target_letter(word, target_letter):
        return 'intrusion'
    
    if is_nonword(word):
        return 'nonword'
    
    return 'correct'