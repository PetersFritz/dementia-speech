"""Rule-based classification helpers for semantic (category) fluency scoring.

Unlike PFT_new, there's no English-dictionary / nonword check here -- the
task itself ("name as many animals as you can") already supplies the one
dictionary that matters. Anything not in it is scored as an intrusion
directly; see SFT_new/scoring.py.
"""
from pathlib import Path

from dementia_speech.paths import DATA_DIR
from dementia_speech.lexical_norms import get_aoa  # noqa: F401 -- re-exported for SFT_new.scoring

_ANIMAL_SCHEME_PATH = DATA_DIR / "raw/reference/animals_snafu_scheme.csv"


def normalize_word(word: str) -> str:
    """Lowercase and strip everything but letters/digits.

    Used both to build the animal-name lookup and to match candidates
    against it, so e.g. "black mamba", "black-mamba" and "blackmamba" all
    resolve to the same key -- the scheme file itself is inconsistent about
    which compound names keep a hyphen (e.g. "horse-fly") and which don't
    (e.g. "blackmamba").
    """
    return "".join(ch for ch in word.lower().strip() if ch.isalnum())


def _load_animal_categories(path: Path = _ANIMAL_SCHEME_PATH) -> dict[str, set[str]]:
    """{normalized_animal_name: {categories}} from the animal-category
    scheme (plain `category,word` CSV, no header, '#' comment lines).

    The file is many-to-one -- the same animal legitimately appears under
    several categories (e.g. a spider under both "African" and "Arachnid")
    -- so values accumulate into a set rather than being overwritten.
    Used as-is, not hand-curated, despite a handful of odd entries (e.g.
    "Farm,spider", "Primates,human") -- see the CognoSpeak issues doc
    discussion for why we're not hand-editing a published reference list.

    Citation: Zemla, J. C., Cao, K., Mueller, K. D., & Austerweil, J. L.
    (2020). SNAFU: The semantic network and fluency utility. Behavior
    Research Methods, 52, 1681-1699. Scheme originally from Troyer,
    Moscovitch, & Winocur (1997), extended by Hills, Jones, & Todd (2012).
    """
    if not path.exists():
        return {}

    categories: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        category, _, word = line.partition(",")
        if not word:
            continue
        key = normalize_word(word)
        if not key:
            continue
        categories.setdefault(key, set()).add(category.strip())
    return categories


ANIMAL_CATEGORIES = _load_animal_categories()

# The scheme only lists singular forms ("cat", "wolf", "mouse"), but
# participants overwhelmingly say plurals ("cats", "dogs", "lions") when
# naming a type of animal rather than one individual -- checked against the
# actual transcripts: plural forms accounted for the single largest source
# of false intrusions before this was added. These are tried in order after
# the word itself; irregular forms are listed explicitly since a bare
# "strip trailing s" rule mishandles them (e.g. "mice" -> "mouse", not
# "mouse" being stripped to "mic").
_IRREGULAR_PLURALS = {
    "mice": "mouse", "geese": "goose", "oxen": "ox", "lice": "louse",
    "men": "man", "children": "child",
}


def _singular_candidates(word: str):
    """Yield normalized match candidates for `word`: itself, then plausible
    singular forms, most specific rule first."""
    norm = normalize_word(word)
    yield norm

    lowered = word.lower().strip()
    if lowered in _IRREGULAR_PLURALS:
        yield normalize_word(_IRREGULAR_PLURALS[lowered])
    if lowered.endswith("ies") and len(lowered) > 3:
        yield normalize_word(lowered[:-3] + "y")
    if lowered.endswith("ves") and len(lowered) > 3:
        yield normalize_word(lowered[:-3] + "f")
        yield normalize_word(lowered[:-3] + "fe")
    if lowered.endswith("es") and len(lowered) > 3:
        yield normalize_word(lowered[:-2])
    if lowered.endswith("s") and len(lowered) > 3:
        yield normalize_word(lowered[:-1])


def is_animal(candidate: str) -> bool:
    """Check if a (possibly multi-word, possibly plural) candidate matches a
    known animal name."""
    return canonical_key(candidate) is not None


def canonical_key(candidate: str) -> str | None:
    """The dictionary key `candidate` resolves to (singular, normalized), or
    None if it doesn't match any known animal.

    Use this (not normalize_word()) when tracking which animals have
    already been said -- "cat" and "cats" must map to the same key for
    repetition detection to work, and normalize_word() alone wouldn't
    collapse a plural onto its singular form.
    """
    for key in _singular_candidates(candidate):
        if key in ANIMAL_CATEGORIES:
            return key
    return None


def get_categories(candidate: str) -> set[str]:
    """Categories a known animal name belongs to (empty set if not found).

    Not used yet -- kept for the semantic clustering analysis to be added
    later (see Tier 1 dashboard TODO: fluency discrepancy score first).
    """
    key = canonical_key(candidate)
    return ANIMAL_CATEGORIES.get(key, set())
