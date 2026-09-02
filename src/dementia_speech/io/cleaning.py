"""Shared token-cleaning used by every transcript loader (TextGrid, plain text, ...)."""
import re


def clean_token(token: str) -> str:
    """Strip punctuation and lowercase a transcript token.

    Keeps internal hyphens so real compound words (e.g. "part-time",
    "ping-pong") survive intact rather than being merged into a nonword
    ("parttime"). Leading/trailing hyphens (truncation markers, stray
    punctuation) are stripped. Returns "" if nothing is left.
    """
    return re.sub(r'[^\w\s-]', '', token).strip().strip('-').lower()
