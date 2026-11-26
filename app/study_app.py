"""Blinded human-evaluation study — sequential 20-case diagnostic-decision flow.

Separate from app.py (the free-browse internal dashboard), but shares all of
its tab modules (Summary, Overview, Fluency, Cookie Theft) rather than a
stale, separately-maintained subset -- see the Tier 2 note below for why
that's worth calling out. Each rater visits with a personal link emailed
after consent, e.g.:

    https://<public-host>/?token=<token>

The token determines (deterministically, see study/state.py) which 20 TEST-split
cases the rater sees and in what order. Progress is stored server-side keyed by
token, so re-opening the same link resumes where the rater left off. The true
diagnosis is never rendered -- summary_server() is called with
show_diagnosis=False here specifically because its Demographics panel would
otherwise show it (see modules/summary.py); every other tab only ever shows
GROUP-level (HC/MCI/Dementia) comparison distributions, never this
participant's own label, so nothing else needed the same treatment.

2026-11-19: this file had drifted to the point of being broken -- it
imported a single `phonemic_fluency_ui` function that no longer exists
(phonemic_fluency.py split into `_cards_ui`/`_detail_ui` a while back), and
only wired up 2 of what are now 5 tabs. Rebuilt against the current module
API; the Fluency tab's markup now lives in modules/fluency_tab.py precisely
so app.py and this file can't silently diverge like that again.

After the 20th case, a qualitative questionnaire is shown in-app (mock
questions below — swap QUESTIONNAIRE_QUESTIONS for the real instrument later,
no other code needs to change). Answers land in the same per-token session
file as the decisions.

Run with:
    shiny run app/study_app.py --host 0.0.0.0 --port <port>
"""
import os
from urllib.parse import parse_qs

from shiny import App, reactive, render, ui

from modules.cookie_theft import cookie_theft_server, cookie_theft_ui
from modules.fluency_tab import fluency_tab_ui
from modules.overview import overview_server, overview_ui
from modules.phonemic_fluency import phonemic_fluency_server
from modules.semantic_fluency import semantic_fluency_server
from modules.summary import summary_server, summary_ui
from study.state import (
    DIAGNOSES,
    get_progress,
    get_questionnaire,
    is_valid_token,
    record_decision,
    record_questionnaire,
)

# DEV ONLY -- TURN OFF BEFORE DEPLOYING: show the participant ID in the
# decision panel so the underlying raw data can be looked up. Raters must
# never see it (it would let them look up the true diagnosis). Currently
# always ON for local development; set STUDY_SHOW_PARTICIPANT_ID=0 to hide it,
# and flip the default to "0" before the app goes on the VM.
SHOW_PARTICIPANT_ID = os.environ.get("STUDY_SHOW_PARTICIPANT_ID", "1") == "1"
print(f"[study_app] SHOW_PARTICIPANT_ID = {SHOW_PARTICIPANT_ID} (loaded from {__file__})")

DIAGNOSIS_CHOICES = {"": "— Select a diagnosis —", **{d: d for d in DIAGNOSES}}

# Mock questionnaire — replace with the real instrument later. Each entry is
# (input_id, kind, label); "likert" renders a 1-5 slider, "text" a text area.
QUESTIONNAIRE_QUESTIONS = [
    ("q_clarity", "likert", "How easy was the dashboard to interpret?"),
    ("q_confidence_overall", "likert", "Overall, how confident did you feel making decisions using this dashboard?"),
    ("q_useful", "text", "What did you find most useful about the dashboard?"),
    ("q_confusing", "text", "What did you find confusing or unhelpful?"),
    ("q_comments", "text", "Any other comments?"),
]

# Same panel/heading styles the dashboard tabs use (duplicated per-module by
# project convention, see summary.py's _PANEL_STYLE).
_PANEL_STYLE = (
    "background:white;border:1px solid #e1e8ef;border-top:4px solid #527d96;"
    "border-radius:12px;padding:22px;min-width:0;"
    "box-shadow:0 2px 8px rgba(35,55,75,0.05);"
)
_SECTION_HEADER_STYLE = "font-weight:700;color:#243447;margin-bottom:4px;"
_SECTION_CAPTION_STYLE = "color:#6c757d;font-size:13px;margin-bottom:16px;"
_SUBMIT_BUTTON_STYLE = (
    "background:#527d96;border:1px solid #527d96;color:white;font-weight:600;"
    "padding:8px 22px;border-radius:8px;"
)

# Tab selector styled to match the dashboard's palette: a white rounded bar
# (same border/radius as each tab's page header), slate-blue text, and the
# active tab filled with the same light blue as the "CLINICAL ASSESSMENT" badge.
_TAB_CSS = ui.tags.style("""
.nav-tabs {
    display: flex; gap: 6px; flex-wrap: wrap;
    background: white; border: 1px solid #e5eaf0 !important;
    border-radius: 16px; padding: 10px; margin-bottom: 20px;
}
.nav-tabs .nav-item { margin: 0; }
.nav-tabs .nav-link {
    border: none !important; border-radius: 10px;
    padding: 13px 30px; font-weight: 600; font-size: 17px;
    color: #6c757d; background: transparent;
    transition: background 0.15s, color 0.15s;
}
.nav-tabs .nav-link:hover { background: #f1f5f8; color: #243447; }
.nav-tabs .nav-link.active {
    background: #eaf2f6; color: #243447;
    box-shadow: inset 0 -3px 0 #527d96;
}
""")

app_ui = ui.page_fluid(
    _TAB_CSS,
    ui.output_ui("page"),
    title="Dementia Speech Dashboard — Study",
)


def server(input, output, session):

    decision_trigger = reactive.Value(0)
    questionnaire_trigger = reactive.Value(0)

    @reactive.calc
    def token():
        # session.http_conn is the websocket connection, whose URL doesn't carry
        # the page's original query string — clientdata.url_search() is the
        # value Shiny's JS client reports back for window.location.search.
        search = session.clientdata.url_search()
        return parse_qs(search.lstrip("?")).get("token", [""])[0]

    @reactive.calc
    def progress():
        decision_trigger()  # re-run after each recorded decision
        t = token()
        if not is_valid_token(t):
            return None
        return get_progress(t)  # (current_index, assigned_cases)

    @reactive.calc
    def questionnaire_submitted():
        questionnaire_trigger()  # re-run after questionnaire submit
        t = token()
        if not is_valid_token(t):
            return False
        return get_questionnaire(t) is not None

    @reactive.calc
    def current_participant_id():
        p = progress()
        if p is None:
            return None
        idx, cases = p
        if idx >= len(cases):
            return None
        return str(cases[idx])

    summary_server("summary", participant_id=current_participant_id, show_diagnosis=False)
    overview_server("overview", participant_id=current_participant_id)
    phonemic_fluency_server("pft", participant_id=current_participant_id)
    semantic_fluency_server("sft", participant_id=current_participant_id)
    cookie_theft_server("ct", participant_id=current_participant_id)

    @reactive.effect
    @reactive.event(input.submit_decision)
    def _on_submit_decision():
        p = progress()
        if p is None:
            return
        idx, cases = p
        if idx >= len(cases):
            return

        guess = input.diagnosis_guess()
        if not guess:
            ui.notification_show(
                "Please select a diagnosis before continuing.", type="warning", duration=4
            )
            return

        record_decision(token(), idx, cases[idx], guess, int(input.confidence()))
        decision_trigger.set(decision_trigger() + 1)

    @reactive.effect
    @reactive.event(input.submit_questionnaire)
    def _on_submit_questionnaire():
        p = progress()
        if p is None:
            return
        idx, cases = p
        if idx < len(cases):
            return  # cases not finished yet — shouldn't normally happen

        answers = {qid: input[qid]() for qid, _kind, _label in QUESTIONNAIRE_QUESTIONS}
        record_questionnaire(token(), answers)
        questionnaire_trigger.set(questionnaire_trigger() + 1)

    def _questionnaire_inputs():
        inputs = []
        for qid, kind, label in QUESTIONNAIRE_QUESTIONS:
            if kind == "likert":
                inputs.append(ui.input_slider(qid, label, min=1, max=5, value=3))
            else:
                inputs.append(ui.input_text_area(qid, label, width="100%", rows=3))
        return inputs

    @render.ui
    def page():
        p = progress()

        if p is None:
            return ui.div(
                ui.h3("Link not recognised"),
                ui.p("Please use the personal link from your study invitation email."),
                style="max-width:600px;margin:4rem auto;text-align:center;",
            )

        idx, cases = p

        if idx >= len(cases):
            if not questionnaire_submitted():
                return ui.div(
                    ui.h3("Almost done"),
                    ui.p("You have completed all 20 cases — please answer a few final questions:"),
                    ui.card(*_questionnaire_inputs()),
                    ui.input_action_button(
                        "submit_questionnaire", "Submit", class_="btn-primary"
                    ),
                    style="max-width:700px;margin:2rem auto;",
                )

            return ui.div(
                ui.h3("Thank you"),
                ui.p("Your responses have been recorded. You may now close this window."),
                style="max-width:600px;margin:4rem auto;text-align:center;",
            )

        return ui.div(
            # Dashboard tabs on the left, decision panel pinned on the right
            # (sticky, so it stays in view while scrolling a long tab). Wraps
            # to a stacked layout on narrow windows.
            ui.div(
                ui.div(
                    ui.navset_tab(
                        ui.nav_panel("Summary", summary_ui("summary")),
                        ui.nav_panel("Memory & Daily Life", overview_ui("overview")),
                        ui.nav_panel("Fluency", fluency_tab_ui("pft", "sft")),
                        ui.nav_panel("Cookie Theft", cookie_theft_ui("ct")),
                    ),
                    style="flex:1 1 640px;min-width:0;",
                ),
                ui.div(
                    ui.div(
                        ui.h5(
                            f"Case {idx + 1} of {len(cases)}",
                            style="font-weight:700;color:#243447;margin-bottom:8px;",
                        ),
                        # Progress = cases already submitted (idx), so it reads
                        # 0% on the first case and 100% only once all are done.
                        ui.div(
                            ui.div(
                                style=(
                                    f"width:{idx / len(cases) * 100:.0f}%;height:100%;"
                                    "background:#527d96;border-radius:4px;"
                                ),
                            ),
                            style=(
                                "height:8px;background:#e1e8ef;border-radius:4px;"
                                "margin-bottom:18px;"
                            ),
                            role="progressbar",
                            **{
                                "aria-valuemin": "0",
                                "aria-valuemax": str(len(cases)),
                                "aria-valuenow": str(idx),
                            },
                        ),
                        *(
                            [ui.div(
                                f"Participant ID: {cases[idx]} (dev only)",
                                style=(
                                    "font-size:12px;font-weight:600;color:#9a5b00;"
                                    "background:#fff4e0;border-radius:6px;"
                                    "padding:4px 6px;margin:-6px 0 14px 0;"
                                ),
                            )]
                            if SHOW_PARTICIPANT_ID else []
                        ),
                        ui.hr(style="margin:0 0 16px 0;"),
                        ui.h5("Your clinical decision", style=_SECTION_HEADER_STYLE),
                        ui.p(
                            "Based on the dashboard, select the diagnosis you think "
                            "is most likely and how confident you are in that choice.",
                            style=_SECTION_CAPTION_STYLE,
                        ),
                        ui.input_radio_buttons(
                            "diagnosis_guess", "Diagnosis", choices=DIAGNOSIS_CHOICES,
                        ),
                        ui.input_slider(
                            "confidence", "Confidence (1 = low, 5 = high)",
                            min=1, max=5, value=3, width="100%",
                        ),
                        ui.input_action_button(
                            "submit_decision", "Submit & continue",
                            style=_SUBMIT_BUTTON_STYLE + "width:100%;padding:8px 2px;font-size:12px;",
                        ),
                        style=_PANEL_STYLE + "padding:8px;",
                    ),
                    style="flex:0 0 100px;position:sticky;top:1rem;align-self:flex-start;",
                ),
                style="display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap;",
            ),
            style="padding:1rem;",
        )


app = App(app_ui, server)
