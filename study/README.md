# Human evaluation study — setup

Two pieces, set up separately: the **consent Google Form** (Google's own UI —
nothing here is deployed automatically) and the **study server**
(`app/study_app.py` in this repo). See `docs/study_protocol.md` for the study
design and `docs/study_hosting.md` for exposing the server publicly.

## 1. Consent Google Form

1. Create a Google Form with your information sheet and consent questions.
2. **Settings > Responses > Collect email addresses** — turn this on. The
   Apps Script below relies on Google's verified respondent email rather than
   a free-text field.
3. **Responses tab > green Sheets icon** — link the form to a new Google
   Sheet, if not already linked.
4. **Extensions > Apps Script** (from the Form editor). Create a new script
   file and paste in the contents of `study/consent_form/onSubmit.gs`.
5. Edit the constants at the top of the script:
   - `STUDY_BASE_URL` — the public URL of the study server (see
     `docs/study_hosting.md`; you'll need this set up first, or come back and
     fill it in once you have it).
   - `SENDER_NAME` — signs the invitation email.
6. **Set up the trigger** — this must be an *installable* trigger, not a
   simple one, because sending email requires authorization that simple
   triggers don't have:
   - In the Apps Script editor, open **Triggers** (clock icon in the left
     sidebar).
   - **Add Trigger** → function: `onFormSubmitHandler`, event source:
     `From form`, event type: `On form submit`. Save.
   - Google will prompt you to authorize the script (send email, access the
     spreadsheet) — accept as the form owner.
7. Submit a test response to the form yourself and confirm you receive the
   study-link email and that a `token` column appears in the Sheet.

## 2. Qualitative questionnaire (post-study)

No separate Google Form — the questionnaire is built into `app/study_app.py`
and shown automatically once a rater completes all 20 cases. Answers are
stored in the same per-token session file as their decisions
(`data/study/sessions/<token>.json`) and picked up by
`pipelines/aggregate_study_results.py` into `data/processed/study_questionnaire.csv`.

The current questions are a placeholder — edit `QUESTIONNAIRE_QUESTIONS` near
the top of `app/study_app.py` to swap in the real instrument. Each entry is
`(input_id, kind, label)` where `kind` is `"likert"` (1–5 slider) or `"text"`
(free-text box); add, remove, or reorder entries freely, no other code needs
to change.

## 3. Study server

See the repo root `README.md` for running `app/study_app.py`, and
`docs/study_hosting.md` for making it reachable from the public internet.
