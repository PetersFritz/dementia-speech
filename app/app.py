"""Dementia Speech Dashboard — multi-tab entry point."""
from shiny import App, ui, reactive

from shared import PARTICIPANTS
from modules.summary import summary_ui, summary_server
from modules.overview import overview_ui, overview_server
from modules.phonemic_fluency import phonemic_fluency_server
from modules.semantic_fluency import semantic_fluency_server
from modules.fluency_tab import fluency_tab_ui
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
        ui.nav_panel("Summary", summary_ui("summary")),
        ui.nav_panel("Memory & Daily Life", overview_ui("overview")),
        ui.nav_panel("Fluency", fluency_tab_ui("pft", "sft")),
        ui.nav_panel("Cookie Theft", cookie_theft_ui("ct")),
    ),
    title="Dementia Speech Dashboard",
)


def server(input, output, session):
    @reactive.calc
    def selected_participant():
        return input.participant()

    summary_server("summary", participant_id=selected_participant)
    overview_server("overview", participant_id=selected_participant)
    phonemic_fluency_server("pft", participant_id=selected_participant)
    semantic_fluency_server("sft", participant_id=selected_participant)
    cookie_theft_server("ct", participant_id=selected_participant)


app = App(app_ui, server)

# Run with:
# shiny run app/app.py --reload