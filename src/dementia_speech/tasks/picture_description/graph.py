"""Spatio-semantic graph construction for Cookie Theft picture description.

Builds a directed graph from a sequence of mentioned information units (CIUs),
using their spatial coordinates on the image, computes graph-theoretic metrics,
and renders an interactive Plotly overlay.

Coordinate conventions
----------------------
- CIU_COORDINATES_RAW: origin top-left, y increases downward (image pixel space).
  Use for the Plotly overlay (Plotly handles orientation via a reversed y-axis).
- CIU_COORDINATES: y-flipped (origin bottom-left). Use for the metric functions,
  which were written for this convention. Numerically the metrics are identical
  either way for distances/cycles, but quadrant assignment depends on it, so we
  keep the original flipped convention for metric computation.
"""
import ast
import base64
import os
from collections import Counter
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

from dementia_speech.tasks.picture_description.registry import COORDINATES_RAW as _REGISTRY_COORDINATES_RAW


# ── Image and coordinate config ────────────────────────────────────────────
# The shared drive mounts at a different absolute path per OS (macOS:
# /Volumes/shared/..., Linux: /mnt/xdrive/...) even though the subpath below
# the mount root is identical. Rather than hardcode one, try each known mount
# root and use whichever exists on the current machine. Override with the
# SPANDH_BESSEMER_ROOT env var if the drive is mounted somewhere else.
_SHARED_SUBPATH = "spandh_bessemer/SPandhAndHealthcare/Fritz/Interspeech2026/data/pictures/CookieTheft.png"
_CANDIDATE_ROOTS = [
    os.environ.get("SPANDH_BESSEMER_ROOT"),
    "/mnt/xdrive",
    "/Volumes/shared",
]


def _resolve_image_path() -> str:
    tried = []
    for root in _CANDIDATE_ROOTS:
        if not root:
            continue
        candidate = Path(root) / _SHARED_SUBPATH
        if candidate.exists():
            return str(candidate)
        tried.append(str(candidate))
    raise FileNotFoundError(
        "Could not find CookieTheft.png on the shared drive. Tried:\n  "
        + "\n  ".join(tried)
        + "\nSet SPANDH_BESSEMER_ROOT to the drive's mount point on this machine."
    )


IMAGE_PATH = _resolve_image_path()
IMAGE_HEIGHT = 592  # used for the metric-convention y-flip

# Raw annotation coordinates (origin top-left, y downward) — canonical
# definitions live in registry.py, shared with the IU checklist in scoring.py.
CIU_COORDINATES_RAW = _REGISTRY_COORDINATES_RAW

# Flipped coordinates for metric computation (origin bottom-left)
CIU_COORDINATES = {
    ciu: (x, IMAGE_HEIGHT - y)
    for ciu, (x, y) in CIU_COORDINATES_RAW.items()
}


# ── Graph construction ───────────────────────────────────────────────────────

def euclidean_distance(coord1, coord2) -> float:
    return float(np.linalg.norm(np.array(coord1) - np.array(coord2)))


def build_ciu_graph(ciu_sequence, ciu_coordinates=CIU_COORDINATES) -> nx.DiGraph:
    """Build a directed graph from an ordered CIU sequence."""
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
    return G


# ── Quadrant helpers ──────────────────────────────────────────────────────────

def get_quadrant(ciu_coords, img_width, img_height):
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
    if not unique:
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


def calculate_metrics(ciu_sequence, ciu_coordinates, image_width, image_height) -> dict:
    """Compute the full set of spatio-semantic graph metrics for one sequence.

    Metric provenance (Ambadi et al., 2021, Frontiers in Neurology,
    "Spatio-Semantic Graphs From Picture Description: Applications to
    Detection of Cognitive Impairment", n=1,349 WRAP/DementiaBank
    participants): six of these metrics were reported as significantly
    different (p<0.01) between cognitively impaired and unimpaired speakers
    — `avg_x`, `stdev_x`, `total_path_distance`, `total_path_per_unique_nodes`,
    `cycles`, `cross_ratio_quadrants`. Impaired speakers showed a
    left-shifted average position (`avg_x` lower — spatial neglect, despite
    the right side of the Cookie Theft picture carrying more content),
    longer/less efficient paths, more repeated mentions, and more
    quadrant-jumping (attentional fragmentation). The remaining metrics here
    (`avg_y`, `stdev_y`, `self_cycles`, `nodes`, `unique_nodes`,
    `self_cycles_quadrants`) are descriptive/exploratory — that paper did
    not report them as diagnostic, so treat them as context rather than
    evidence when reading a single case.
    """
    if len(ciu_sequence) == 0:
        return {}
    a_x = avg_x(ciu_sequence, ciu_coordinates)
    return {
        "avg_x": a_x,
        "stdev_x": stdev_x(ciu_sequence, ciu_coordinates),
        "avg_y": avg_y(ciu_sequence, ciu_coordinates),
        "stdev_y": stdev_y(ciu_sequence, ciu_coordinates),
        # Positive = narrative centred left of the picture's midline (the
        # "spatial neglect" direction reported by Ambadi et al., 2021); a
        # normalized, sign-oriented re-expression of avg_x for readability.
        "spatial_left_bias": round((image_width / 2 - a_x) / (image_width / 2), 3),
        "total_path_distance": total_path_distance(ciu_sequence, ciu_coordinates),
        "total_path_per_unique_nodes": total_path_per_unique_nodes(ciu_sequence, ciu_coordinates),
        "self_cycles": self_cycles(ciu_sequence),
        "cycles": cycles(ciu_sequence),
        "nodes": total_nodes(ciu_sequence),
        "unique_nodes": unique_nodes(ciu_sequence),
        "self_cycles_quadrants": self_cycles_in_quadrants(ciu_sequence, ciu_coordinates, image_width, image_height),
        "cross_ratio_quadrants": cross_ratio_quadrants(ciu_sequence, ciu_coordinates, image_width, image_height),
    }


# ── Interactive Plotly overlay ────────────────────────────────────────────────

def render_graph_plotly(
    ciu_sequence_full,
    iu_counts,
    coordinates_raw=CIU_COORDINATES_RAW,
    image_path=IMAGE_PATH,
    node_colour="#e74c3c",
    edge_colour="#2c3e50",
):
    """Render the CIU graph as an interactive Plotly figure over the image."""
    with Image.open(image_path) as im:
        img_width, img_height = im.size

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    img_uri = f"data:image/png;base64,{encoded}"

    G = build_ciu_graph(ciu_sequence_full, coordinates_raw)

    fig = go.Figure()

    fig.add_layout_image(dict(
        source=img_uri,
        xref="x", yref="y",
        x=0, y=0,
        sizex=img_width, sizey=img_height,
        sizing="stretch",
        opacity=0.55,
        layer="below",
    ))

    # Edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = coordinates_raw[u]
        x1, y1 = coordinates_raw[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(color=edge_colour, width=2),
            hoverinfo="skip", showlegend=False,
        ))

    # Nodes
    node_x, node_y, node_text, node_hover, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = coordinates_raw[node]
        count = iu_counts.get(node, 1)
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_hover.append(f"{node}<br>mentions: {count}")
        node_size.append(18 + 6 * (count - 1))

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(color=node_colour, size=node_size, opacity=0.65,
                    line=dict(color="white", width=1.5)),
        text=node_text, textposition="top center",
        textfont=dict(size=10, color="#222"),
        hovertext=node_hover, hoverinfo="text", showlegend=False,
    ))

    fig.update_xaxes(range=[0, img_width], showgrid=False, zeroline=False,
                     visible=False, constrain="domain")
    fig.update_yaxes(range=[img_height, 0], showgrid=False, zeroline=False,
                     visible=False, scaleanchor="x", scaleratio=1)

    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=int(520 * img_height / img_width) if img_width else 520,
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig