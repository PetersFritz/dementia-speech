"""Dementia Speech Dashboard — multi-tab entry point."""
from shiny import App, ui, reactive

from shared import PARTICIPANTS
from modules.overview import overview_ui, overview_server
from modules.phonemic_fluency import phonemic_fluency_ui, phonemic_fluency_server
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
        ui.nav_panel("Phonemic Fluency", phonemic_fluency_ui("pft")),
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
    cookie_theft_server("ct", participant_id=selected_participant)


app = App(app_ui, server)

# Run with:
# shiny run app/app.py --reload