"""Tests for graph.total_path_per_node -- path distance per mention
(every occurrence), distinct from the existing total_path_per_unique_nodes
(distinct IUs only). The two should disagree whenever a participant
revisits the same IUs rather than visiting as many different ones.
"""
from dementia_speech.tasks.picture_description.graph import (
    total_path_per_node,
    total_path_per_unique_nodes,
)

_COORDS = {"A": (0, 0), "B": (10, 0), "C": (0, 10)}


def test_matches_unique_node_version_when_no_repeats():
    # No IU is mentioned twice, so dividing by total mentions and dividing
    # by unique mentions is the same thing.
    seq = ["A", "B", "C"]
    assert total_path_per_node(seq, _COORDS) == total_path_per_unique_nodes(seq, _COORDS)


def test_diverges_from_unique_node_version_with_repeats():
    # Same two locations, visited three times each, vs. three locations
    # visited twice each: total_path_per_node (divides by every mention)
    # and total_path_per_unique_nodes (divides by distinct IUs only) must
    # disagree on which sequence is "more efficient" -- repeat_heavy has
    # fewer distinct locations but more mentions to spread its distance
    # over, so it should look relatively better per-node and relatively
    # worse per-unique-node than spread.
    repeat_heavy = ["A", "B", "A", "B", "A", "B"]
    spread = ["A", "B", "C", "A", "B", "C"]

    per_node_ratio = total_path_per_node(repeat_heavy, _COORDS) / total_path_per_node(spread, _COORDS)
    per_unique_ratio = (
        total_path_per_unique_nodes(repeat_heavy, _COORDS)
        / total_path_per_unique_nodes(spread, _COORDS)
    )
    assert per_node_ratio != per_unique_ratio


def test_empty_sequence_returns_zero():
    assert total_path_per_node([], _COORDS) == 0.0
