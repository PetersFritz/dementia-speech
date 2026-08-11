# Study protocol — decisions log

Follows the same log style as `docs/decisions.md`, kept separate since this
covers the human evaluation study rather than scoring/dashboard decisions.

## 2026-08-11: Study design

- **Goal:** validate the dashboard as a clinical decision-support tool via an
  online human evaluation study.
- **Decision task per case:** rater picks a diagnosis guess (HC / MCI /
  Dementia) plus a confidence rating (1–5). No free-text justification for
  now.
- **Case set:** 20 cases per rater, sampled from `data/raw/meta-info.csv`
  rows where `Split == "TEST"` (80 cases: 40 HC / 30 MCI / 10 Dementia),
  stratified proportionally to that split (10 HC / 8 MCI / 2 Dementia at
  n=20, via largest-remainder allocation — see `app/study/state.py`).
- **Case order & selection are deterministic per rater token** — seeded from
  `sha256(token)` — rather than pre-computed and shared with the consent
  form. This means the study server never needs to talk to the Google Form
  or Sheet; the token alone is sufficient. Trade-off: changing the sampling
  logic changes assignments for any token not yet used, so avoid changing
  `derive_case_list` after the study has started recruiting.
- **Blinding:** true diagnosis is never rendered in the dashboard modules
  (confirmed by code review of `app/modules/phonemic_fluency.py` and
  `cookie_theft.py` — only anonymised HC/MCI/Dementia group-mean reference
  lines are shown, never the current participant's own label), so the
  existing modules are reused unmodified in the study flow.
- ~~**Known data gap:** PFT scoring currently only covers 202 participants,
  all in the `TRAIN` split~~ — resolved 2026-08-11: all raw PFT TextGrids
  were added and `pipelines/score_pft.py` rerun; all 80 TEST-split
  participants now have both PFT and Cookie Theft data.
- **Consent & recruitment:** Google Form (info sheet + consent + verified
  email) → Apps Script (`study/consent_form/onSubmit.gs`) emails a personal
  study link automatically on submit. No participant PII is stored by the
  study server — only an opaque token. See `study/README.md`.
- **Post-study questionnaire:** originally planned as a second Google Form
  linked back by token via a pre-filled URL parameter. Changed 2026-08-11 to
  an in-app form (`QUESTIONNAIRE_QUESTIONS` in `app/study_app.py`, shown once
  `get_progress(token)` reports all 20 cases complete) — simpler than
  wiring up Forms prefill matching, and keeps questionnaire answers in the
  same per-token session file as the decisions so `aggregate_study_results.py`
  can join them without a second data source. Current questions are a
  placeholder mock-up (2 Likert + 3 free-text) pending the real instrument.
- **Persistence:** originally planned as a single shared SQLite database.
  Switched to one JSON file per token (`data/study/sessions/<token>.json`,
  atomic write-temp-then-rename) after confirming the dev checkout's CIFS
  network mount (`/mnt/xdrive/...`) does not support the file locking SQLite
  requires — every SQLite write failed with "database is locked" there, while
  local disk worked fine. Per-token files avoid single-file lock contention
  and tolerate both local disk and this network mount; final deployment
  target (local disk vs. an IT-provisioned VM) was undecided at time of
  writing, so this was chosen for portability rather than performance.
- **Hosting:** must be reachable from the public internet (raters are
  off-site). Considered: own machine + Cloudflare Tunnel, a commercial cloud
  VM, shinyapps.io, and Sheffield IT Services' self-requestable virtual
  server. Ruled out shinyapps.io — it's US-hosted (AWS us-east-1) with no
  EU/UK region, which would add a cross-border data transfer question to
  ethics/governance beyond what hosting on Sheffield-controlled
  infrastructure needs. Current lean: Sheffield IT Services VM (5 working
  day turnaround, self-requestable, Linux + SSH — see `docs/study_hosting.md`
  Option B for the concrete setup and caveats, e.g. Puppet reverting
  unmanaged config changes, automatic monthly reboots, public-internet
  reachability isn't covered by the standard request and needs asking about
  explicitly).

### Open questions / follow-ups
- [ ] Confirm the Sheffield VM will actually be public-internet reachable
      (not just on-campus/VPN) — ask explicitly when submitting the request.
- [ ] Confirm ethics/governance approval before recruiting real participants.
- [ ] Replace the placeholder `QUESTIONNAIRE_QUESTIONS` in `app/study_app.py`
      with the real qualitative instrument.
