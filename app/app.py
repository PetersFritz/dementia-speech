"""Dementia Speech Dashboard — multi-tab entry point."""
from shiny import App, ui, reactive, render
import plotly.graph_objects as go
from shinywidgets import output_widget, render_widget

from shared import (
    PARTICIPANTS, FLUENCY_DISCREPANCY_BY_PARTICIPANT, FLUENCY_DISCREPANCY_DF,
    GROUP_COLOURS,
)
from modules.overview import overview_ui, overview_server
from modules.phonemic_fluency import (
    phonemic_fluency_cards_ui, phonemic_fluency_detail_ui, phonemic_fluency_server,
)
from modules.semantic_fluency import (
    semantic_fluency_cards_ui, semantic_fluency_detail_ui, semantic_fluency_server,
)
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
            ui.div(
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
                        "Each card shows a value and an age-adjusted percentile "
                        "— no Typical/Borderline/Atypical label, just where this "
                        "score falls for a clinician to weigh themselves.",
                        style="color:#6c757d;font-size:13px;margin-bottom:16px;"
                    ),
                    ui.layout_columns(
                        ui.div(
                            ui.h6(
                                "Phonemic Fluency",
                                style="font-weight:600;color:#34495e;margin-bottom:6px;"
                            ),
                            phonemic_fluency_cards_ui("pft"),
                        ),
                        ui.div(
                            ui.h6(
                                "Semantic Fluency",
                                style="font-weight:600;color:#34495e;margin-bottom:6px;"
                            ),
                            semantic_fluency_cards_ui("sft"),
                        ),
                        col_widths=[6, 6],
                    ),
                    style="margin-bottom:26px;",
                ),

                # Experimental cross-task score -- sits right after the
                # Clinical summary cards (user request, 2026-11-17): it's a
                # direct comparison of the two `C` values just shown above,
                # so it reads as a continuation of that summary rather than
                # an afterthought at the bottom of the page, below both
                # tasks' full detail.
                ui.output_ui("discrepancy_card"),

                ui.layout_columns(
                    ui.div(
                        ui.h4(
                            "Phonemic Fluency",
                            style="font-weight:700;color:#243447;margin:0 0 14px 0;"
                        ),
                        phonemic_fluency_detail_ui("pft"),
                    ),
                    ui.div(
                        ui.h4(
                            "Semantic Fluency",
                            style="font-weight:700;color:#243447;margin:0 0 14px 0;"
                        ),
                        semantic_fluency_detail_ui("sft"),
                    ),
                    col_widths=[6, 6],
                ),

                style="background:#f5f7fa;padding:22px;border-radius:14px;",
            ),
        ),
        ui.nav_panel("Cookie Theft", cookie_theft_ui("ct")),
    ),
    title="Dementia Speech Dashboard",
)


_DISCREPANCY_GROUPS = ["HC", "MCI", "Dementia"]


def _discrepancy_distribution_figure(participant_value):
    """Per-group violin of discrepancy_score (z_letter - z_category) plus
    this participant's own value as a dashed line -- same visual language
    as every other measure card's embedded graph (see
    phonemic_fluency.py::_card_distribution_figure), but with no shaded
    "typical" band: that convention bands against an age-adjusted norm
    fit for the one measure being shown, which hasn't been done for this
    derived, explicitly experimental score."""
    fig = go.Figure()
    for group in _DISCREPANCY_GROUPS:
        vals = FLUENCY_DISCREPANCY_DF.loc[
            FLUENCY_DISCREPANCY_DF["diagnosis"] == group, "discrepancy_score"
        ].dropna().values
        if len(vals) < 2 or vals.std() == 0:
            continue
        fig.add_trace(go.Violin(
            x=vals, y=[group] * len(vals), orientation="h", name=group,
            line_color=GROUP_COLOURS[group], fillcolor=GROUP_COLOURS[group],
            opacity=0.35, meanline_visible=True,
            meanline=dict(color="#111", width=2),
            points=False, spanmode="hard", showlegend=False,
        ))
    fig.update_yaxes(categoryorder="array", categoryarray=["Dementia", "MCI", "HC"])

    if participant_value is not None and participant_value == participant_value:
        fig.add_vline(x=participant_value, line=dict(color="#1565C0", width=2, dash="dash"))

    fig.update_layout(height=160, margin=dict(t=6, b=24, l=6, r=6), plot_bgcolor="white")
    fig.update_xaxes(gridcolor="#eee", tickfont=dict(size=10),
                     title=dict(text="discrepancy score (z_letter − z_category)", font=dict(size=10)))
    fig.update_yaxes(gridcolor="#eee", tickfont=dict(size=10))
    return fig


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
                style="color:#666;font-size:0.82em;max-width:40em;margin-bottom:10px;",
            ),
            output_widget("discrepancy_graph"),
            style=(
                "background:white;border:1px solid #e1e8ef;border-left:4px solid #9e9e9e;"
                "border-radius:10px;padding:16px 18px;max-width:40em;margin-bottom:26px;"
                "box-shadow:0 2px 8px rgba(35,55,75,0.05);"
            ),
        )

    @render_widget
    def discrepancy_graph():
        row = FLUENCY_DISCREPANCY_BY_PARTICIPANT.get(str(selected_participant()))
        value = row.get("discrepancy_score") if row else None
        return _discrepancy_distribution_figure(value)


app = App(app_ui, server)

# Run with:
# shiny run app/app.py --reload