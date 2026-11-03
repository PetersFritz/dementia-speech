# Dashboard TODO: changes before study release

Working file for Claude Code. Source: clinician focus group feedback (coded, one transcript so far).
Code IDs in brackets refer to the reworked codebook (e.g. D7 = written summary).

## Working agreement (read first)

- Plan before editing. For each task: say which files and functions it touches, then wait for my OK.
- One task per branch or commit. Tick the box here when a task is done and tested.
- No participant or patient data in the repo, in prompts, or in logs. Use synthetic data only.
- Use only data already collected. If a measure needs data we don't have, flag it and skip it.
- Keep scoring logic in plain functions, separate from the Shiny UI, and unit test them.
- Do not write clinical wording without showing it to me first. Summary text is reviewed by clinicians.
- Mark unvalidated measures as experimental in the UI.

## Decisions (provisional, to be confirmed by clinicians)

- [ ] **No diagnostic label** on the dashboard (option A). Describe, don't diagnose. [L1, L2]
- [ ] **Bands per measure** from likelihood shares under each group (HC, MCI, dementia), equal priors:
  - normal: HC share >= 0.6
  - abnormal: MCI + dementia share >= 0.6
  - slightly abnormal: otherwise (overlap zone)
- [ ] **"Closest group"** is named only if its share >= 0.5, otherwise "no clear group". Chart caption only, never the summary headline.
- [ ] **Measure direction**: record per measure whether lower is worse, higher is worse, or two-sided (e.g. discrepancy score).
- [ ] **"Normal overall" rule**: define so that chance flags across ~20 measures don't make almost everyone abnormal.
- [ ] **Confounders to show**: years of education confirmed; age and others to decide. [B2, B3] (education isn't currently in `data/raw/meta-info.csv` — needs sourcing, or this decision needs revisiting)
- [ ] **Measures gated out**: measures with weak separation (AUC below a threshold, placeholder 0.65) are shown as "not informative alone" and excluded from the overall rule.

## Tier 0: audit (do first)

- [ ] Write a Python script that prints, per measure: n per group (HC, MCI, dementia), missing values, distribution shape.
- [ ] Same script checks which of these exist: age, education, recording date, finance item, daily-living item, mood and anxiety, news answer, life-story answer, words and duration per answer.
- [ ] Output a short `docs/data_audit.md` (counts only, no participant-level data).
- [ ] Note which "Add" items below are blocked by missing data.
- [ ] Known gap (checked ahead of the audit): `data/raw/meta-info.csv` currently only has `IDs, diagnosis, age, gender, MMSE, Split` — no education column, no recording date. Confirm whether these exist elsewhere before the audit script can surface anything different.

## Tier 1: core scoring and summary

- [ ] Reference distributions per measure and group (kernel density for skewed measures; log-transform where needed).
- [ ] Equal-prior shares per measure; bands and closest group from the shares (see Decisions).
- [ ] Cross-validated accuracy and AUC per measure. Never score a person with distributions that include them.
- [ ] Show or hide curves based on group size (placeholder: n < 30 per group, hide).
- [ ] Unit tests: all normal, one extreme value, missing measures, small group, measure with two-sided direction.
- [ ] Dynamic written summary, template-based, three branches:
  - all measures normal
  - mostly normal with some deviations (list them, largest first, grouped by task)
  - many deviations
  - plus a line for missing measures so absence of data is never read as normal [D7]
- [ ] "Closest to ..." line only for measures that clear the 0.5 rule.

## Tier 1: dashboard changes (in order)

- [ ] 1. Bullet-point written summary on the overview. [D7]
- [ ] 2. Distribution chart per measure: three group curves, shaded HC range, the person's marker. Replace harsh cut-off display. [R4, R5, R6, D8]
- [ ] 3. Fluency discrepancy score in the top section. [F1, F2]
- [ ] 4. Semantic fluency (SFT) analysis section. [F9]
- [ ] 5. Confounder warning on the overview, plus years of education. [B2, B3]
- [ ] 6. Make the scales at the bottom more readable. [D9]
- [ ] 7. Collapsible additional measures; information icons for explanations on demand; one-line plain-language description per measure. [D6, U2, U3]
- [ ] 8. Demote or remove questioned measures: pronoun ratio, path distance, measures with similar group means, repetitions. [C5, C7]
- [ ] 9. Per-group comparison view (HC vs MCI vs dementia on each measure). [R7]
- [ ] 10. Note on the overview that patient concern is self-reported and limited. [X6]

## Tier 2: additions (only if the audit shows the data exists)

- [x] Finance question — already built and live (`pipelines/score_finance.py`, Overview tab, all 400 participants). [X1]
- [ ] Daily-living items. [X2, X3] (no such data found in the repo — stays gated)
- [ ] Mood and anxiety scores, possibly a separate tab. [X4]
- [ ] Engagement: words produced and time taken, compared with group means. Caution: may disadvantage the dementia group. [N1, N2, N3, N4]
- [ ] Self-correction measure. [I1, I2]
- [ ] "Who is concerned about memory" item. [X5]

## Tier 2: recent-news question (separate pipeline)

Depends on recording dates being present. If not, show words and duration for this question instead. (checked ahead of time: no recording-date field found in `data/raw/meta-info.csv` or `Process_demographics.csv` — expect to need the fallback for everyone unless a date source turns up elsewhere)

- [ ] Write `docs/news_scoring_rules.md`: "recent" window (placeholder 42 days), categories (matches in-window event, real but outdated, vague, none or wrong, unverifiable), 0-3 rubric.
- [ ] Build a cached event list for the recording dates only: Wikipedia current-events pages plus Guardian UK/world/politics headlines. Keep the source URL per event.
- [ ] Local Qwen matches each answer to candidate events (JSON output, temperature 0). Date logic and final score are computed in code, not by the model.
- [ ] Unsure or inconsistent model output becomes missing, never zero. Log raw output, model name, prompt version.
- [ ] Review the "unverifiable" answers, add frequent missing events, re-run.
- [ ] Hand-rate 50 to 100 answers and compare with the model before showing the measure. Until then, flag it as experimental.
- [ ] Optional second line: detail per 100 words (named entities or LLM extraction, counted in code).
- [ ] Life-story question: only compute detail per 100 words; check group differences against HC first. Include only if groups separate. Label it "richness of life story account", not accuracy.

## Release preparation

- [ ] Clinician review of: cut-offs, summary wording, and whether "closest to the MCI range" reads as a diagnostic label.
- [ ] "About this tool" note: what it is, what data it was built on, that it does not diagnose, which measures are experimental.
- [ ] Version stamp in the UI.
- [ ] Tests: unit tests for scoring and summary; app tests (shinytest2) or screenshots on synthetic data.
- [ ] Manual check on two or three real profiles locally; read summaries for wrong numbers, overclaiming, awkward wording.
- [ ] Prepare the feedback questions to ask participants.

## Parked (not this release)

- Longitudinal features: measure of change, z-scores plus change, mapping to cognitive domains. [LG1, LG2, LG3]
- Memory questionnaire (needs new data), collateral history. [X11, X7]
- Information about the training dataset (belongs in documentation). [T4]
- Qwen rewording of the summary (template is enough for the study).
- Multivariate "overall pattern" model (only if cross-validated performance is good enough to defend).
- Explicit certainty scores or an optional label view.

## Open questions

- What cut-offs and "normal overall" rule do clinicians prefer?
- Which measures are two-sided?
- Which confounders beyond education should be shown, and where?
- How to show engagement so it doesn't penalise people with dementia?
- Discrepancy score wording: the transcript says it is sensitive "early" and useful "in MCI" (F3, F4). Check against the recording before any stage-specific text goes in the UI.
