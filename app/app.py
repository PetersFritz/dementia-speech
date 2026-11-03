"""Dementia Speech Dashboard — multi-tab entry point."""
from shiny import App, ui, reactive, render

from shared import PARTICIPANTS, FLUENCY_DISCREPANCY_BY_PARTICIPANT
from modules.overview import overview_ui, overview_server
from modules.phonemic_fluency import phonemic_fluency_ui, phonemic_fluency_server
from modules.semantic_fluency import semantic_fluency_ui, semantic_fluency_server
from modules.cookie_theft import cookie_theft_ui, cookie_theft_server

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Dementia Speech"),
        ui.input_select(
            "participant",
            "Select Participant",
            choices={str(p): f"Participant {p}" for p in PARTICIPANTS},
        ),
        width=240,
    ),
    ui.navset_tab(
        ui.nav_panel("Overview", overview_ui("overview")),
        ui.nav_panel(
            "Fluency",
            ui.output_ui("discrepancy_card"),
            ui.layout_columns(
                ui.div(
                    ui.h3("Phonemic Fluency", style="margin-top:0;"),
                    phonemic_fluency_ui("pft"),
                ),
                ui.div(
                    ui.h3("Semantic Fluency", style="margin-top:0;"),
                    semantic_fluency_ui("sft"),
                ),
                col_widths=[6, 6],
            ),
        ),
        ui.nav_panel("Cookie Theft", cookie_theft_ui("ct")),
    ),
    title="Dementia Speech Dashboard",
)


def server(input, output, session):
    @reactive.calc
    def selected_participant():
        return input.participant()

    overview_server("overview", participant_id=selected_participant)
    phonemic_fluency_server("pft", participant_id=selected_participant)
    semantic_fluency_server("sft", participant_id=selected_participant)
    cookie_theft_server("ct", participant_id=selected_participant)

    # Letter-vs-category discrepancy score (Wright et al. 2022) -- a
    # comparison between the two fluency tasks, so it sits above both
    # rather than belonging to either module. Experimental: single trial
    # per task, and doesn't show the group separation the paper reports on
    # this dataset -- see pipelines/score_fluency_discrepancy.py.
    @render.ui
    def discrepancy_card():
        row = FLUENCY_DISCREPANCY_BY_PARTICIPANT.get(str(selected_participant()))
        if not row:
            return ui.div()
        value = row.get("discrepancy_score")
        value_str = f"{value:+.2f}" if value is not None and value == value else "—"
        return ui.div(
            ui.div(
                ui.span("Experimental", style=(
                    "background:#9e9e9e;color:white;border-radius:12px;"
                    "padding:2px 10px;font-size:0.8em;font-weight:600;"
                )),
                style="margin-bottom:6px;",
            ),
            ui.div("Fluency discrepancy score", style="font-weight:600;font-size:1.05em;"),
            ui.div(value_str, style="font-size:1.6em;font-weight:700;margin:2px 0;"),
            ui.div(
                "Letter fluency z-score minus category fluency z-score "
                "(Wright et al. 2022), against this project's own TRAIN-"
                "split HC group. Positive = category fluency fell further "
                "below normal than letter fluency did. On this dataset it "
                "doesn't show the group separation the paper reports — "
                "shown for reference, not as a clinical flag.",
                style="color:#666;font-size:0.82em;max-width:40em;",
            ),
            style=(
                "background:white;border:1px solid #e0e0e0;border-left:4px solid #9e9e9e;"
                "border-radius:8px;padding:12px 14px;max-width:40em;margin-bottom:14px;"
            ),
        )


app = App(app_ui, server)

# Run with:
# shiny run app/app.py --reload