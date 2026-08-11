"""Blinded human-evaluation study — sequential 20-case diagnostic-decision flow.

Separate from app.py (the free-browse internal dashboard). Each rater visits
with a personal link emailed after consent, e.g.:

    https://<public-host>/?token=<token>

The token determines (deterministically, see study/state.py) which 20 TEST-split
cases the rater sees and in what order. Progress is stored server-side keyed by
token, so re-opening the same link resumes where the rater left off. The true
diagnosis is never rendered — see app/modules/ for confirmation.

After the 20th case, a qualitative questionnaire is shown in-app (mock
questions below — swap QUESTIONNAIRE_QUESTIONS for the real instrument later,
no other code needs to change). Answers land in the same per-token session
file as the decisions.

Run with:
    shiny run app/study_app.py --host 0.0.0.0 --port <port>
"""
from urllib.parse import parse_qs

from shiny import App, reactive, render, ui

from modules.cookie_theft import cookie_theft_server, cookie_theft_ui
from modules.phonemic_fluency import phonemic_fluency_server, phonemic_fluency_ui
from study.state import (
    DIAGNOSES,
    get_progress,
    get_questionnaire,
    is_valid_token,
    record_decision,
    record_questionnaire,
)

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

app_ui = ui.page_fluid(
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

    phonemic_fluency_server("pft", participant_id=current_participant_id)
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
            ui.h4(f"Case {idx + 1} of {len(cases)}"),
            ui.navset_tab(
                ui.nav_panel("Phonemic Fluency", phonemic_fluency_ui("pft")),
                ui.nav_panel("Cookie Theft", cookie_theft_ui("ct")),
            ),
            ui.hr(),
            ui.card(
                ui.h5("Your clinical decision"),
                ui.input_radio_buttons(
                    "diagnosis_guess", "Diagnosis", choices=DIAGNOSIS_CHOICES, inline=True,
                ),
                ui.input_slider(
                    "confidence", "Confidence (1 = low, 5 = high)", min=1, max=5, value=3,
                ),
                ui.input_action_button("submit_decision", "Submit & continue", class_="btn-primary"),
            ),
            style="padding:1rem;",
        )


app = App(app_ui, server)
