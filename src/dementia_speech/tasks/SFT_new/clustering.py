"""Semantic cluster/switch analysis for semantic (category) fluency (Troyer
et al., 1997; category scheme from Zemla, Cao, Mueller & Austerweil, 2020,
"SNAFU: The semantic network and fluency utility", Behavior Research
Methods 52).

Groups consecutive correct animal names into clusters sharing at least one
SNAFU category (e.g. "lion" and "tiger" both "Big cats"), then reports
clusters, switches, and cluster sizes -- the category-fluency analog of
PFT_new/clustering.py's phonemic version, same structure and field
definitions; see that module's docstring for the general Troyer method this
mirrors. Unlike PFT_new's phonemic match, category membership is
many-to-one (the same animal can legitimately belong to several categories,
see SFT_new/rules.py), so adjacency here is "shares AT LEAST ONE category
with the immediately preceding correct word," not an exact match.
"""
import numpy as np

from dementia_speech.tasks.SFT_new.rules import get_categories


def _shares_category(word_a: str, word_b: str) -> bool:
    """True if word_a and word_b share at least one SNAFU category -- the
    category-fluency analog of PFT's phonemic-match rule. A word with no
    recognised category (shouldn't happen for an already-scored correct
    response, but handled defensively) never clusters with anything."""
    cats_a = get_categories(word_a)
    cats_b = get_categories(word_b)
    return bool(cats_a and cats_b and (cats_a & cats_b))


def build_runs(correct_words: list[str]) -> list[list[str]]:
    """Group an ordered list of correct animal names into maximal adjacency
    runs -- same definition as PFT_new.clustering.build_runs: a word starts
    a new run unless it shares a category with the immediately preceding
    correct word (judged on adjacency, not against the whole trial
    history). A run of a single word is NOT a cluster (Troyer requires 2+
    related words).
    """
    if not correct_words:
        return []

    runs = [[correct_words[0]]]
    for word in correct_words[1:]:
        if _shares_category(word, runs[-1][-1]):
            runs[-1].append(word)
        else:
            runs.append([word])
    return runs


def compute_cluster_metrics(correct_words: list[str]) -> dict:
    """Summarize cluster/switch metrics for a trial's correct-animal
    sequence -- same return shape as PFT_new.clustering.compute_cluster_metrics;
    see its docstring for field definitions (n_clusters, n_switches,
    mean_cluster_size, largest_cluster_size).
    """
    runs = build_runs(correct_words)
    if not runs:
        return {
            "n_clusters": 0, "n_switches": 0,
            "mean_cluster_size": None, "largest_cluster_size": 0,
        }

    cluster_sizes = [len(r) for r in runs if len(r) > 1]
    return {
        "n_clusters": len(cluster_sizes),
        "n_switches": len(runs) - 1,
        "mean_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else None,
        "largest_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
    }
