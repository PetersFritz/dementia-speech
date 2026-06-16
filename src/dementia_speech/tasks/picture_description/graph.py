"""Spatio-semantic graph construction for Cookie Theft picture description.

Adapted from existing project code. Builds a directed graph from a sequence
of mentioned information units (CIUs), using their spatial coordinates on
the image, and computes graph-theoretic metrics describing visual scanning
and narrative structure.
"""
import ast
from collections import Counter

import numpy as np
import networkx as nx
import pandas as pd


# ── Image and coordinate config ────────────────────────────────────────────
# Coordinates are in standard image space (origin top-left, y increases downward)
# after flipping from the original annotation convention (origin bottom-left).

IMAGE_PATH = "/mnt/xdrive/spandh_bessemer/SPandhAndHealthcare/Fritz/Interspeech2026/data/pictures/CookieTheft.png"
IMAGE_HEIGHT = 592  # used to flip y-coordinates; adjust if image dimensions change

_CIU_COORDINATES_FLIPPED = {
    "Boy": (295, 196),
    "Girl": (134, 354),
    "Woman": (501, 279),
    "Kitchen": (630, 300),
    "Cookie": (200, 70),
    "Jar": (200, 100),
    "Stool": (226, 438),
    "Sink": (610, 344),
    "Plate": (564, 228),
    "Dishcloth": (539, 284),
    "Water": (589, 461),
    "Window": (678, 190),
    "Cupboard": (204, 124),
    "Dishes": (765, 386),
    "Curtains": (679, 125),
    "Exterior": (694, 245),
    "Steal": (233, 105),
    "Fall": (263, 369),
    "Wash": (550, 255),
    "Overflow": (604, 376),
}

CIU_COORDINATES = {
    ciu: (x, IMAGE_HEIGHT - y)
    for ciu, (x, y) in _CIU_COORDINATES_FLIPPED.items()
}


# ── Graph construction ───────────────────────────────────────────────────────

def euclidean_distance(coord1, coord2) -> float:
    return float(np.linalg.norm(np.array(coord1) - np.array(coord2)))


def build_ciu_graph(
    ciu_sequence: list[str],
    ciu_coordinates: dict = CIU_COORDINATES,
) -> nx.DiGraph:
    """Build a directed graph from an ordered CIU sequence.

    Nodes are CIUs with their spatial position. Edges connect consecutively
    mentioned CIUs, weighted by Euclidean distance between their positions.

    Parameters
    ----------
    ciu_sequence : list[str]
        Ordered list of CIU names as mentioned by the participant
        (output of get_iu_sequence, may contain repeats)
    ciu_coordinates : dict
        Mapping of CIU name to (x, y) coordinate on the image

    Returns
    -------
    nx.DiGraph
        Directed graph with 'pos' node attribute and 'weight' edge attribute
    """
    G = nx.DiGraph()

    for ciu in ciu_sequence:
        if ciu in ciu_coordinates:
            G.add_node(ciu, pos=ciu_coordinates[ciu])
        else:
            print(f"Warning: CIU '{ciu}' not found in coordinates")

    for i in range(len(ciu_sequence) - 1):
        ciu1, ciu2 = ciu_sequence[i], ciu_sequence[i + 1]
        if ciu1 in ciu_coordinates and ciu2 in ciu_coordinates:
            dist = euclidean_distance(ciu_coordinates[ciu1], ciu_coordinates[ciu2])
            G.add_edge(ciu1, ciu2, weight=dist)
        else:
            print(f"Warning: one of '{ciu1}' or '{ciu2}' missing in coordinates")

    return G


# ── Quadrant helpers ──────────────────────────────────────────────────────────

def get_quadrant(ciu_coords, img_width: float, img_height: float):
    """Return quadrant index 0-3 (Top-Left, Top-Right, Bottom-Left, Bottom-Right)."""
    x, y = ciu_coords
    if x < img_width / 2 and y >= img_height / 2:
        return 0
    elif x >= img_width / 2 and y >= img_height / 2:
        return 1
    elif x < img_width / 2 and y < img_height / 2:
        return 2
    elif x >= img_width / 2 and y < img_height / 2:
        return 3
    return None


# ── Sequence-level metrics ───────────────────────────────────────────────────

def avg_x(seq, coords):
    return float(np.mean([coords[c][0] for c in seq]))


def stdev_x(seq, coords):
    return float(np.std([coords[c][0] for c in seq]))


def avg_y(seq, coords):
    return float(np.mean([coords[c][1] for c in seq]))


def stdev_y(seq, coords):
    return float(np.std([coords[c][1] for c in seq]))


def total_path_distance(seq, coords):
    return float(sum(
        euclidean_distance(coords[seq[i]], coords[seq[i + 1]])
        for i in range(len(seq) - 1)
    ))


def total_path_per_unique_nodes(seq, coords):
    unique = set(seq)
    if len(unique) == 0:
        return 0.0
    return total_path_distance(seq, coords) / len(unique)


def self_cycles(seq):
    return sum(1 for i in range(1, len(seq)) if seq[i] == seq[i - 1])


def cycles(seq):
    counts = Counter(seq)
    return sum(c - 1 for c in counts.values() if c > 1)


def total_nodes(seq):
    return len(seq)


def unique_nodes(seq):
    return len(set(seq))


def self_cycles_in_quadrants(seq, coords, img_width, img_height):
    count = 0
    for i in range(1, len(seq)):
        q1 = get_quadrant(coords[seq[i - 1]], img_width, img_height)
        q2 = get_quadrant(coords[seq[i]], img_width, img_height)
        if q1 == q2:
            count += 1
    return count


def cross_ratio_quadrants(seq, coords, img_width, img_height):
    same, diff = 0, 0
    for i in range(1, len(seq)):
        q1 = get_quadrant(coords[seq[i - 1]], img_width, img_height)
        q2 = get_quadrant(coords[seq[i]], img_width, img_height)
        if q1 == q2:
            same += 1
        else:
            diff += 1
    if same == 0:
        return float(diff)
    return diff / same


def calculate_metrics(
    ciu_sequence: list[str],
    ciu_coordinates: dict,
    image_width: float,
    image_height: float,
) -> dict:
    """Compute the full set of spatio-semantic graph metrics for one sequence.

    Parameters
    ----------
    ciu_sequence : list[str]
        Ordered list of mentioned CIUs (may contain repeats — use the raw
        mention order, not the deduplicated 'first appearance' list, if you
        want cycle/repetition metrics to be meaningful)
    ciu_coordinates : dict
        CIU name -> (x, y) coordinate
    image_width, image_height : float
        Dimensions of the reference image

    Returns
    -------
    dict
        Metric name -> value
    """
    if len(ciu_sequence) == 0:
        return {}

    return {
        "avg_x": avg_x(ciu_sequence, ciu_coordinates),
        "stdev_x": stdev_x(ciu_sequence, ciu_coordinates),
        "avg_y": avg_y(ciu_sequence, ciu_coordinates),
        "stdev_y": stdev_y(ciu_sequence, ciu_coordinates),
        "total_path_distance": total_path_distance(ciu_sequence, ciu_coordinates),
        "total_path_per_unique_nodes": total_path_per_unique_nodes(ciu_sequence, ciu_coordinates),
        "self_cycles": self_cycles(ciu_sequence),
        "cycles": cycles(ciu_sequence),
        "nodes": total_nodes(ciu_sequence),
        "unique_nodes": unique_nodes(ciu_sequence),
        "self_cycles_quadrants": self_cycles_in_quadrants(
            ciu_sequence, ciu_coordinates, image_width, image_height
        ),
        "cross_ratio_quadrants": cross_ratio_quadrants(
            ciu_sequence, ciu_coordinates, image_width, image_height
        ),
    }