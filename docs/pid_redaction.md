# PID redaction in Qwen-extracted text (who_concerned, finance, memory_complaints, news)

User request, 2026-11-11: names and specific places must never reach a
dashboard-facing CSV. This covers the four tasks that run a local Qwen model
over a transcript and write free text (`summary`, `evidence_quote`, and
news's `details`) into a CSV the Shiny app reads.

## How it works

The model is told to redact, not a separate NER/gazetteer pass run
afterwards. Each task's `prompt.py` embeds an instruction telling the model
to substitute `[NAME]` for any person's name and `[PLACE]` for any specific
place, directly into `evidence_quote`/`details`/`summary` as it writes
them -- see `dementia_speech.tasks._shared.redaction.REDACTION_INSTRUCTIONS`
(embedded verbatim in `who_concerned`, `finance`, `memory_complaints`; see
below for why `news` has its own variant).

The harder part is grounding: every one of these tasks already verifies
that `evidence_quote` is an exact substring of the transcript before
trusting it (`verify.py`'s `is_grounded`, dropping the record/finding/claim
otherwise -- see each task's own scoring docs). Once the quote can contain
`[NAME]`/`[PLACE]`, it's no longer a literal substring by design, so a plain
substring check would now reject every correctly-redacted quote.
`dementia_speech.tasks._shared.redaction.is_grounded_allowing_redaction`
fixes this: it treats each placeholder as a wildcard (matching the real
name/place text, up to a 3-word span) at that exact position, while every
other word in the quote still has to match the transcript exactly. All four
tasks' `verify.py` now import this as `is_grounded`, so nothing else about
the verification logic changed -- a quote with a fabricated surrounding
sentence still fails, same as before.

## The `news` task is different: personal vs. public

`who_concerned`/`finance`/`memory_complaints` are about the participant's
own life, so any name/place mentioned is almost certainly personal.
`news`'s `details` field is deliberately designed to capture things like a
storm's name or the place an event happened, as evidence of real
substance (see `docs/news_scoring_rules.md`) -- blanket-redacting every
name/place there would destroy that measure, and the event-matching logic
itself depends on being able to recognise public figures/places. So
`news/prompt.py` has its own `NEWS_REDACTION_INSTRUCTIONS`: redact a name
or place from the participant's OWN life (a family member, their own home
town), never a name/place that's part of the news event being described
(a storm's name, a politician, a country, a city something happened in).
`is_grounded_allowing_redaction` itself doesn't care which kind of
redaction produced a placeholder -- it's the same function either way.

## Running it

Nothing new to launch -- this changed each task's prompt and its
`verify.py`, not the pipeline scripts or the sbatch files. Re-run whichever
of these you want redacted output from, same as always:

```
sbatch pipelines/slurm/score_who_concerned.sbatch
sbatch pipelines/slurm/score_finance.sbatch
sbatch pipelines/slurm/score_memory_complaints.sbatch
sbatch pipelines/slurm/score_news.sbatch
```

Each of these overwrites its own output CSV (`who_concerned.csv`,
`finance.csv`, `memory_complaints.csv`, `news_scores.csv`) under
`data/processed/`. `news`'s `PROMPT_VERSION` was bumped to `v4` (the other
three tasks don't currently track a prompt version the way `news` does) --
if you add that same audit-log convention to them later, this change is
the point old/new rows should split at.

## What this is -- and isn't

This is an instruction to the model, not a guaranteed filter. It will miss
a name or place the model fails to recognise as such -- same caveat as
every other LLM-extracted field in this project (and genuinely no
different from asking a human transcriber to redact by hand and
occasionally missing one). **Spot-check a sample of each CSV's
`summary`/`evidence_quote`/`details` columns after re-running these
pipelines**, specifically looking for a name or place that slipped through
unredacted, before treating this as sufficient on its own for anything
sensitive leaving this environment. It reduces risk; it doesn't eliminate
it.

Not covered by this change at all: the raw audit log
(`data/processed/news_scores_audit.jsonl`) keeps the model's true,
unredacted raw completion on purpose, for QA against the real transcript
during the hand-rating pass -- it must stay access-restricted the same way
the rest of `data/` already is, never surfaced in the dashboard. Also not
covered: the Cookie Theft/PFT/SFT tabs, which show the participant's raw
transcript directly rather than LLM-extracted text -- a different, much
larger redaction problem that wasn't part of this request.
