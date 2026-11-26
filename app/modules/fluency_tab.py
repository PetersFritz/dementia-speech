"""Shared Fluency tab markup -- the combined PFT + SFT page, used by both
app.py (the free-browse internal dashboard) and study_app.py (the blinded
rater study).

Factored out here (2026-11-19) because study_app.py had drifted badly: it
was still importing a single `phonemic_fluency_ui` function that no longer
exists (phonemic_fluency.py split into `_cards_ui`/`_detail_ui` a while
back), so this tab had silently been broken in the study app rather than
just stale. Only the UI lives here -- each caller still wires up
phonemic_fluency_server()/semantic_fluency_server() itself, with its own
participant_id reactive, same as every other tab.
"""
from shiny import ui

from modules.phonemic_fluency import phonemic_fluency_cards_ui, phonemic_fluency_detail_ui
from modules.semantic_fluency import semantic_fluency_cards_ui, semantic_fluency_detail_ui


_FLUENCY_GRID_CSS = ui.tags.style("""
.fluency-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    column-gap: 1rem; row-gap: 0;
}
.fluency-col {
    display: grid; grid-template-rows: subgrid; row-gap: 0; min-width: 0;
}
/* Cards: make each side's card wrapper fill its (shared) row, so cards
   are the same height on both sides. */
.fluency-cards-grid .fluency-col .shiny-html-output { display: flex; }
.fluency-cards-grid .fluency-col .shiny-html-output > div { flex: 1 1 auto; }
""")


def fluency_tab_ui(pft_id="pft", sft_id="sft"):
    return ui.div(
        # PAGE HEADER -- same chrome as the Cookie Theft tab
        # (cookie_theft.py's cookie_theft_ui) for visual consistency
        # across tabs.
        ui.div(
            ui.div(
                ui.h3(
                    "Fluency",
                    style="font-weight:700;color:#243447;margin:0;"
                ),
                ui.p(
                    "Phonemic and semantic fluency assessment",
                    style="color:#6c757d;margin:4px 0 0 0;"
                ),
            ),
            ui.div(
                ui.span(
                    "CLINICAL ASSESSMENT",
                    style=(
                        "font-size:11px;font-weight:700;letter-spacing:1px;"
                        "color:#426b83;background:#eaf2f6;"
                        "padding:7px 12px;border-radius:20px;"
                    )
                ),
            ),
            style=(
                "display:flex;justify-content:space-between;"
                "align-items:center;flex-wrap:wrap;"
                "padding:22px 26px;margin-bottom:22px;"
                "background:white;border:1px solid #e5eaf0;"
                "border-radius:14px;"
            ),
        ),

        # SHARED CLINICAL SUMMARY -- one heading for both tasks'
        # cards, rather than each task repeating its own "Clinical
        # summary" section.
        ui.div(
            ui.h5(
                "Clinical summary",
                style="font-weight:700;color:#243447;margin-bottom:4px;"
            ),
            ui.p(
                "Each card shows a value, an age-adjusted percentile and "
                "z-score, and a group-comparison graph — no "
                "Typical/Borderline/Atypical label, just where this score "
                "falls for a clinician to weigh themselves. On each graph "
                "the violins show every group's raw values (not "
                "age-adjusted) and the shaded band is the healthy-control "
                "range for this participant's age.",
                style="color:#6c757d;font-size:13px;margin-bottom:16px;"
            ),
            # Two-column grid whose rows are SHARED by both columns (each
            # column is a subgrid spanning the same rows), so the heading and
            # cards line up across tasks even when one side's content is
            # taller. The cards' wrapper divs are stretched so equal-row
            # cards are the same height on both sides.
            ui.div(
                ui.div(
                    ui.h6(
                        "Phonemic Fluency",
                        style="font-weight:600;color:#34495e;margin-bottom:6px;"
                    ),
                    phonemic_fluency_cards_ui(pft_id),
                    class_="fluency-col", style="grid-row:span 2;",
                ),
                ui.div(
                    ui.h6(
                        "Semantic Fluency",
                        style="font-weight:600;color:#34495e;margin-bottom:6px;"
                    ),
                    semantic_fluency_cards_ui(sft_id),
                    class_="fluency-col", style="grid-row:span 2;",
                ),
                class_="fluency-grid fluency-cards-grid",
            ),
            style="margin-bottom:26px;",
        ),

        # Detail sections: heading, transcript, middle row (bin plot /
        # discrepancy score) and additional analysis -- 4 rows shared by both
        # columns, so each block starts at the same height on both sides.
        # (Each *_detail_ui's outer div is display:contents so its sections
        # become the subgrid's direct rows.)
        ui.div(
            ui.div(
                ui.h4(
                    "Phonemic Fluency",
                    style="font-weight:700;color:#243447;margin:0 0 14px 0;"
                ),
                phonemic_fluency_detail_ui(pft_id),
                class_="fluency-col", style="grid-row:span 4;",
            ),
            ui.div(
                ui.h4(
                    "Semantic Fluency",
                    style="font-weight:700;color:#243447;margin:0 0 14px 0;"
                ),
                semantic_fluency_detail_ui(sft_id),
                class_="fluency-col", style="grid-row:span 4;",
            ),
            class_="fluency-grid",
        ),

        _FLUENCY_GRID_CSS,

        style="background:#f5f7fa;padding:22px;border-radius:14px;",
    )
