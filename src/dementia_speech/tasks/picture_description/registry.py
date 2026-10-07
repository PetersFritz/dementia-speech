"""Canonical registry of Cookie Theft information/content units (IUs/CIUs).

Single source of truth for the IU list used across IU-coverage scoring
(scoring.py), spatio-semantic graph metrics (graph.py), and the
efficiency/lexical measures — previously the synonym list (scoring.py) and
the coordinate list (graph.py) were two separately hardcoded dicts that had
to be kept in sync by hand.

This project uses a 20-item CIU list (16 person/object units + 4 action
units), matching the 16 object/subject units in Ambadi et al. (2021,
Frontiers in Neurology, "Spatio-Semantic Graphs From Picture Description:
Applications to Detection of Cognitive Impairment") but not their full
23-item list. Ambadi et al. add three further action units — the girl's
action, the woman being unconcerned by the overflowing sink, and the woman
being indifferent to the children's danger — which are propositional/
pragmatic rather than nominal (Cummings, 2019, identifies exactly this kind
of inference failure as a marker of Alzheimer's dementia). They are
deliberately excluded here: they can't be reliably detected by keyword/
synonym matching, since a transcript can state the underlying fact ("the
sink is overflowing") without using any hedge word a rule could key on, so
the units would show near-universal absence rather than genuine signal
across all diagnostic groups. Capturing them properly would need semantic/
LLM-based extraction rather than a synonym list — a possible future
direction, not implemented here.

Coordinates are pixel positions on CookieTheft.png, origin top-left, y
increasing downward (the image's native convention) — graph.py derives its
y-flipped metric-computation convention from these.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class IU:
    name: str
    synonyms: list[str]
    category: str  # "person" | "object" | "action"
    coord_raw: tuple[float, float]  # (x, y), origin top-left, y downward


IUS: list[IU] = [
    IU("Boy", ["boy", "son", "brother", "male child"], "person", (295, 196)),
    IU("Girl", ["girl", "daughter", "sister", "female child"], "person", (134, 354)),
    IU("Woman", ["woman", "mom", "mother", "lady", "parent", "female", "adult", "grownup", "mum"], "person", (501, 279)),
    IU("Kitchen", ["kitchen", "room"], "object", (630, 300)),
    IU("Cookie", ["cookie", "biscuit", "cake", "treat"], "object", (200, 70)),
    IU("Jar", ["jar", "container", "crock", "pot"], "object", (200, 100)),
    IU("Stool", ["stool", "seat", "chair", "ladder"], "object", (226, 438)),
    IU("Sink", ["sink", "basin", "washbasin", "washbowl", "washstand", "tap"], "object", (610, 344)),
    IU("Plate", ["plate"], "object", (564, 228)),
    IU("Dishcloth", ["dishcloth", "dishrag", "rag", "cloth", "napkin", "towel"], "object", (539, 284)),
    IU("Water", ["water", "dishwater", "liquid"], "object", (589, 461)),
    IU("Window", ["window", "frame", "glass"], "object", (678, 190)),
    IU("Cupboard", ["cupboard", "closet", "shelf"], "object", (204, 124)),
    IU("Dishes", ["dish", "dishes", "cup", "cups", "counter"], "object", (765, 386)),
    IU("Curtains", ["curtain", "curtains", "drape", "drapes", "drapery", "blind", "blinds", "screen", "screens"], "object", (679, 125)),
    IU("Exterior", ["exterior", "outside", "garden", "yard", "outdoors", "backyard", "driveway", "path", "tree", "bush"], "object", (694, 245)),
    IU("Steal", ["take", "steal", "taking", "stealing"], "action", (233, 105)),
    IU("Fall", ["fall", "falling", "slip", "slipping"], "action", (263, 369)),
    IU("Wash", ["wash", "dry", "clean", "washing", "drying", "cleaning"], "action", (550, 255)),
    IU("Overflow", ["overflow", "spill", "overflowing", "spilling"], "action", (604, 376)),
]

SYNONYMS: dict[str, list[str]] = {iu.name: iu.synonyms for iu in IUS}
CATEGORY: dict[str, str] = {iu.name: iu.category for iu in IUS}
COORDINATES_RAW: dict[str, tuple[float, float]] = {iu.name: iu.coord_raw for iu in IUS}
