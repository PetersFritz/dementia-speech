"""Lexical-semantic measures for Cookie Theft picture descriptions.

Computed directly from the running transcript text, independent of the IU
checklist. These follow the lexico-semantic category of connected-speech
measures identified in Boschi et al. (2017, Frontiers in Psychology,
"Connected Speech in Neurodegenerative Language Disorders: A Review") and
the "empty speech" literature (Giles et al., 1996, "Performance on the
Boston Cookie Theft picture description task in patients with early
dementia of the Alzheimer's type"; Ash et al., 2006, "Trying to tell a tale:
discourse impairments in progressive aphasia and frontotemporal dementia"):
Alzheimer's speech tends to carry proportionally more closed-class words,
pronouns and indefinite/deictic terms per word (semantically "empty"
relative to its length), and to draw on a smaller working vocabulary
(reduced lexical diversity).

These are general discourse measures, not specific to the Cookie Theft
picture's content, so — unlike the IU/graph measures — they generalise to
any connected-speech transcript.
"""

import nltk

from dementia_speech.tasks.picture_description.scoring import tokenize_words

# Penn Treebank tags counted as "content" word classes for lexical density
# (Johansson-style formula: (nouns + lexical verbs + adjectives + adverbs) /
# total words). Verb tags alone don't distinguish lexical verbs ("walks")
# from auxiliaries/copulas ("is", "has been") — nltk's tagger assigns both
# the same VB* tag — so auxiliaries are excluded by lexical lookup below,
# per the standard treatment of lexical density (Halliday, 1985; Ure, 1971).
_CONTENT_TAGS = {
    "NN", "NNS", "NNP", "NNPS",
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    "JJ", "JJR", "JJS",
    "RB", "RBR", "RBS",
}
_PRONOUN_TAGS = {"PRP", "PRP$", "WP", "WP$"}
_AUXILIARY_VERBS = {
    "be", "is", "are", "was", "were", "been", "being", "am",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
}
# Deliberately excludes demonstratives ("this", "that", "it") — those are
# counted under pronoun_ratio instead, so the two measures don't overlap.
_INDEFINITE_TERMS = {
    "thing", "things", "something", "somebody", "someone",
    "stuff", "whatsit", "whatchamacallit",
}


def compute_lexical_density(transcript: str) -> dict:
    """Content-word ratio: (nouns + lexical verbs + adjectives + adverbs) / total words.

    Lower lexical density means proportionally more function words
    (determiners, prepositions, pronouns, auxiliaries) relative to content —
    "emptier" speech, associated with AD (Boschi et al., 2017).
    """
    words = tokenize_words(transcript)
    if not words:
        return {"lexical_density": None, "pronoun_ratio": None}

    tagged = nltk.pos_tag(words)
    content = sum(
        1 for word, tag in tagged
        if tag in _CONTENT_TAGS and not (tag.startswith("VB") and word in _AUXILIARY_VERBS)
    )
    pronouns = sum(1 for _, tag in tagged if tag in _PRONOUN_TAGS)

    return {
        "lexical_density": round(content / len(words), 3),
        "pronoun_ratio": round(pronouns / len(words), 3),
    }


def compute_indefinite_term_rate(transcript: str) -> dict:
    """Rate of indefinite/empty nouns ("thing", "stuff", "something", ...) per 100 words.

    A higher rate indicates word-finding difficulty being masked by vague
    placeholder terms rather than the target content word (Ash et al., 2006).
    """
    words = tokenize_words(transcript)
    if not words:
        return {"indefinite_term_rate": None}
    hits = sum(1 for w in words if w in _INDEFINITE_TERMS)
    return {"indefinite_term_rate": round(100 * hits / len(words), 2)}


def compute_mattr(transcript: str, window: int = 25) -> dict:
    """Moving-Average Type-Token Ratio (Covington & McFall, 2010).

    Lexical diversity (distinct words / total words) falls with reduced
    vocabulary access in AD, but the raw type-token ratio is confounded by
    sample length — longer samples mechanically produce lower TTR, which
    would bias comparisons across participants who simply talk for
    different lengths of time. MATTR averages the TTR over a sliding window
    of `window` words, removing that length confound. Transcripts shorter
    than `window` fall back to the plain TTR over the whole transcript
    (flagged via `mattr_window` so this can be filtered downstream).
    """
    words = tokenize_words(transcript)
    n = len(words)
    if n == 0:
        return {"mattr": None, "mattr_window": None}
    if n < window:
        return {"mattr": round(len(set(words)) / n, 3), "mattr_window": n}

    ratios = [
        len(set(words[i:i + window])) / window
        for i in range(n - window + 1)
    ]
    return {"mattr": round(sum(ratios) / len(ratios), 3), "mattr_window": window}


def compute_lexical_metrics(transcript: str, mattr_window: int = 25) -> dict:
    """Combine all lexical-semantic measures for one transcript."""
    metrics = {}
    metrics.update(compute_lexical_density(transcript))
    metrics.update(compute_indefinite_term_rate(transcript))
    metrics.update(compute_mattr(transcript, window=mattr_window))
    return metrics
