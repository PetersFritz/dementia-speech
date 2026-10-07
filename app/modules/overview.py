"""Overview tab module — participant demographics."""
import pandas as pd
from shiny import module, ui, render

from shared import DEMOGRAPHICS


@module.ui
def overview_ui():
    return ui.div(
        ui.h5("Demographics"),
        ui.output_ui("demographics_card"),
    )


@module.server
def overview_server(input, output, session, participant_id):
    """participant_id is a reactive returning the selected participant string."""

    @render.ui
    def demographics_card():
        demo = DEMOGRAPHICS.get(participant_id(), {})
        age = demo.get("age")
        gender = demo.get("gender")
        diagnosis = demo.get("diagnosis")

        def field(label, value):
            return ui.div(
                ui.div(label, style="color:#666;font-size:0.82em;"),
                ui.div(
                    str(value) if pd.notna(value) else "—",
                    style="font-size:1.4em;font-weight:700;",
                ),
            )

        return ui.div(
            field("Age", age),
            field("Gender", gender),
            field("Diagnosis", diagnosis),
            style=(
                "background:white;border:1px solid #e0e0e0;border-radius:8px;"
                "padding:12px 14px;display:flex;gap:24px;max-width:24em;"
            ),
        )
