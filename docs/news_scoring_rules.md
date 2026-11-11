# Recent-news (STN) scoring rules

**Experimental.** Per the Tier 2 TODO (`docs/to_do/TODO_dashboard.md`), this
measure needs a 50-100 answer hand-rating pass against
`pipelines/score_news.py`'s output before it's shown in the dashboard as
anything more than exploratory.

## The question

GPCOG-style "what has been in the news recently?" (STN). Exact wording
varies slightly by system -- `src/dementia_speech/tasks/news/prompt.py`'s
`QUESTION_TEXT` is a reasonable generic phrasing, not yet checked against
every system's exact per-question wording in `data/raw/matching_prompts.csv`.

## Why a "recent" window of 7 days

GPCOG's own recent-news item only asks about the preceding week, so that's
the window `pipelines/build_news_event_cache.py` caches against and
`src/dementia_speech/tasks/news/scoring.py`'s `WINDOW_DAYS` checks against --
not the 42-day placeholder an earlier version of the TODO used.

## Pipeline shape: retrieval, then a model, then code -- not one LLM call

1. **Retrieval (code, no model)** -- `news/retrieval.py` narrows the full
   cached event list down to a short candidate list for one participant: a
   plain keyword-overlap score between their answer and each cached event's
   headline+summary, over a 60-day window (wider than the 7-day "recent
   enough" cutoff, so a genuinely real but *outdated* answer can still be
   found and distinguished from one that's simply made up or unverifiable).
   This step decides what the model gets to see, nothing about whether the
   answer is right.
2. **Matching (local Qwen, temperature 0)** -- `news/extract.py` shows the
   model the transcript and that numbered candidate list, and asks for:
   - `answer_quality`: `specific` / `vague` / `none` -- is this even a
     checkable claim?
   - `claims`: a *list*, one entry per distinct event the answer describes
     -- an answer can name more than one ("there was that storm, and
     something about the Prime Minister too"), and the old single-match
     design silently dropped every event after the first. Each claim has
     its own `matched_candidate_number` (which exact numbered candidate, if
     any) and `evidence_quote`. The model is never asked to reproduce a
     date or headline itself -- only to point at a number -- specifically
     so `news/verify.py` can catch a hallucinated or out-of-range number
     rather than trusting free-text back from the model.
   - `summary`, one sentence for the whole answer.
   - `details`: each claim also carries its own list of concrete facts the
     participant gave about that event -- a name, place, number, cause, or
     consequence, beyond merely naming/recognising the event. This is
     what catches "fluent but empty" recall, a pattern worth flagging on
     its own especially for MCI/dementia: a participant can correctly
     identify a real recent event (so `category`/`score` above are fine)
     while giving no actual substance about it, just confident filler
     around the bare topic. See "Detail/substance metrics" below.
   - Every `evidence_quote` and every `details` entry must be copied
     verbatim from the transcript, same grounding convention as
     `who_concerned`/`finance`, checked by `verify.is_grounded`. A claim
     that fails grounding, or whose `matched_candidate_number` is out of
     range, is dropped individually by `verify.sanitize_claims` -- one bad
     claim doesn't throw away other valid claims from the same answer, and
     one bad `details` entry doesn't throw away the claim it belongs to.
3. **Date logic and final score (code, no model)** -- `news/scoring.py`
   compares each matched claim's event date against the participant's
   `assessment_date`, and the answer is scored on its *best* claim: if any
   claim lands within the 7-day window, the whole answer is
   `in_window_match`, even if it also mentions something vague or
   outdated. All matched events are still kept (for the audit log), not
   just the best one. The model is deliberately never asked "was this
   recent enough" -- that's a plain date subtraction, not a judgement call
   worth spending model reliability on.

## Categories and scores

| Category | Meaning | Score |
|---|---|---|
| `in_window_match` | At least one claim in the answer matches a cached event within the 7 days before the assessment date. | **3** |
| `outdated` | At least one claim matches a cached event, but none are within 7 days. | **2** |
| `vague` | Answer is too generic to check against anything. | **1** |
| `none_or_wrong` | No real answer at all (don't know / refused / off-topic). | **0** |
| `unverifiable` | Specific-sounding answer, but no claim matched a cached candidate. | **missing** |
| `no_date` | No `assessment_date` for this participant, so no candidate search was possible. | **missing** |

An answer with multiple claims is scored on its best one -- recalling one
genuinely recent real event is good evidence of intact recall even if the
same answer also includes something vague, outdated, or unverifiable.
`pipelines/score_news.py`'s `news_scores.csv` row also records `n_claims`
and `n_matched_claims`, and the audit log keeps every claim and every
matched event, not just the one that decided the final category.

`unverifiable` and `no_date` are deliberately **not** 0 -- per the Tier 2
TODO ("unsure or inconsistent model output becomes missing, never zero"),
not being able to verify an answer is a gap in this measure, not evidence
the participant got it wrong. A specific answer that's actually correct but
about a real event the cache happens not to cover (Guardian's free-tier
backfill is still in progress as of 2026-11-09, and there's no local South
Yorkshire source at all) would also land here -- see the TODO for the
possible later fix.

## Detail/substance metrics (exploratory, separate from category/score)

`news/scoring.py::compute_detail_metrics` adds three columns to
`news_scores.csv`, independent of the accuracy category above:

| Column | Meaning |
|---|---|
| `detail_count` | Total concrete details across every claim in the answer (whether or not that claim matched a real cached event -- elaboration is informative even when the topic can't be verified). |
| `best_claim_detail_count` | Details specifically on the claim that decided `category`/`score` -- the direct answer to "they got the event right, but how much did they actually say about it?" `None` if no claim did (vague/none/unverifiable). |
| `detail_rate_per_100_words` | `detail_count` normalised by the participant's total word count, so a long, fluent, low-substance answer doesn't look fine just by rambling -- a high word count with a low `detail_rate_per_100_words` is exactly the "fluent but empty" pattern this is meant to catch. |

These are plain counts over whatever `details` the model extracted for each
claim (grounded, per-entry, same as everything else here) -- no separate
model judgement of "how detailed is this." Like every exploratory measure
in this project (e.g. PFT/SFT's mean-AoA, clustering), **not yet checked
for a group difference** -- per the Tier 2 TODO's own instruction for the
life-story question's equivalent idea ("check group differences against HC
first... include only if groups separate"), this needs that same check,
and the wording should describe "richness of detail," not "accuracy,"
if and when it's shown.

### A deliberate scope limit: no model fact-checking

The model only ever matches against the candidate list it's shown -- it is
never asked to judge, from its own world knowledge, whether an answer
describes something real. A specific-but-false claim ("there was a huge
earthquake in London last week") gets the same `unverifiable` treatment as
a specific-but-real claim the cache doesn't happen to cover; this pipeline
can't currently tell those apart. The Tier 2 TODO's own category list names
a `wrong` category distinct from `unverifiable` that this doesn't yet
implement, as doing so safely would mean trusting the model's open-world
fact-checking rather than grounding everything in the cached event list --
exactly the kind of hallucination risk the rest of this design avoids.
Revisit only if hand-rating shows `unverifiable` answers are disproportionately
real-but-uncached rather than fabricated.

## Logging

Every scored row's raw model completion, the model name/path, the prompt
version (`news/prompt.py`'s `PROMPT_VERSION`), the exact candidate list
shown, and the parsed result are appended to
`data/processed/news_scores_audit.jsonl` (one JSON line per participant) --
separate from the tidy analysis table in `data/processed/news_scores.csv`,
per the Tier 2 TODO's audit requirement.

## Not yet done

- Hand-rate 50-100 answers against this pipeline's output; until then, treat
  the measure as experimental wherever it's shown.
- Finish the Guardian cache backfill (free-tier 500 calls/day cap).
- Check `detail_count`/`detail_rate_per_100_words` for an actual HC-vs-MCI/
  Dementia group difference before showing it as anything beyond raw data.
- Dashboard presentation (card, wording, experimental flag) -- not yet
  wired into `app/modules/`.
