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
from collections import Counter

import numpy as np
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from PIL import Image


# ── Image and coordinate config ────────────────────────────────────────────
IMAGE_PATH = "/Volumes/shared/spandh_bessemer/SPandhAndHealthcare/Fritz/Interspeech2026/data/pictures/CookieTheft.png"
#IMAGE_PATH = "/mnt/xdrive/spandh_bessemer/SPandhAndHealthcare/Fritz/Interspeech2026/data/pictures/CookieTheft.png"
IMAGE_HEIGHT = 592  # used for the metric-convention y-flip

# Raw annotation coordinates (origin top-left, y downward)
CIU_COORDINATES_RAW = {
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
    """Compute the full set of spatio-semantic graph metrics for one sequence."""
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