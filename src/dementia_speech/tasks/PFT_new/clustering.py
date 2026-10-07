"""Phonemic cluster/switch analysis for phonemic fluency (Troyer et al., 1997).

Groups consecutive correct responses into clusters of phonemically related
words, then reports the number of clusters, switches between clusters, and
cluster sizes -- established markers of retrieval strategy in verbal fluency
(reduced switching is linked to executive dysfunction, reduced clustering to
degraded phonological/lexical organisation).
"""
import numpy as np

from dementia_speech.tasks.PFT_new.rules import is_phonemic_cluster_match


def build_runs(correct_words: list[str]) -> list[list[str]]:
    """Group an ordered list of correct words into maximal adjacency runs.

    A word starts a new run unless it's phonemically related
    (`is_phonemic_cluster_match`) to the immediately preceding correct word
    -- membership is judged on adjacency, not against the whole trial
    history, per Troyer et al.'s definition. A run of a single word is NOT a
    cluster (Troyer requires 2+ related words) -- it's a word that switched
    away from whatever came before it and didn't get joined by anything
    after. Use `compute_cluster_metrics`/`assign_cluster_ids` for
    cluster-aware results; this is the shared low-level grouping.
    """
    if not correct_words:
        return []

    runs = [[correct_words[0]]]
    for word in correct_words[1:]:
        if is_phonemic_cluster_match(word, runs[-1][-1]):
            runs[-1].append(word)
        else:
            runs.append([word])
    return runs


def assign_cluster_ids(correct_words: list[str]) -> list[int | None]:
    """0-indexed cluster id for each word in an ordered correct-word sequence.

    Same length/order as `correct_words`. Words in the same true cluster
    (2+ phonemically related consecutive words) get the same id; a word
    that's a singleton run -- not related to its neighbours -- gets None,
    since a single word isn't a cluster. For mapping cluster membership onto
    a display (e.g. the transcript); use `compute_cluster_metrics` for
    summary metrics.
    """
    runs = build_runs(correct_words)
    ids: list[int | None] = []
    cluster_id = 0
    for run in runs:
        if len(run) > 1:
            ids.extend([cluster_id] * len(run))
            cluster_id += 1
        else:
            ids.append(None)
    return ids


def compute_cluster_metrics(correct_words: list[str]) -> dict:
    """Summarize cluster/switch metrics for a trial's correct-word sequence.

    A cluster requires 2+ phonemically related consecutive words -- a
    singleton run doesn't count as a cluster, so `n_clusters`,
    `mean_cluster_size` and `largest_cluster_size` only consider runs of
    size 2+. `n_switches` counts every transition between runs regardless of
    size, since moving into or out of a singleton is still a change in
    retrieval strategy.

    Returns
    -------
    dict
        n_clusters -- number of true clusters (runs of 2+ related words)
        n_switches -- transitions between runs of any size (n_runs - 1)
        mean_cluster_size -- average size of true clusters (None if none)
        largest_cluster_size -- size of the biggest true cluster (0 if none)
    """
    runs = build_runs(correct_words)
    if not runs:
        return {
            'n_clusters': 0, 'n_switches': 0,
            'mean_cluster_size': None, 'largest_cluster_size': 0,
        }

    cluster_sizes = [len(r) for r in runs if len(r) > 1]
    return {
        'n_clusters': len(cluster_sizes),
        'n_switches': len(runs) - 1,
        'mean_cluster_size': float(np.mean(cluster_sizes)) if cluster_sizes else None,
        'largest_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
    }
