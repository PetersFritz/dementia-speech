# SFT scoring (SFT_new): what is scored and how

This describes the semantic/category fluency (SFT) scoring pipeline
(`pipelines/score_sft.py`, `src/dementia_speech/tasks/SFT_new/`) and the
clinician-facing presentation built on top of it (`app/modules/semantic_fluency.py`).
Mirrors the structure of `docs/pft_new_scoring.md` for the equivalent phonemic
fluency pipeline, with the differences called out explicitly.

## Input

Unlike PFT, there's no word-level `TextGrid` for SFT — no forced-aligned
timing exists for this task. Instead, scoring reads the manually
transcribed free-text transcripts in `data/raw/process/SFT/` (NOT the
Whisper transcripts in `data/raw/text/SFT/`, which lack the speaker tags
and pause/event annotations this scoring relies on). Each transcript may
contain:
- Speaker tags at the start of a line (`Pat:`, `Oth:`/`Oth2:`) — a file with
  no tag at all is treated as one continuous `Pat` turn.
- Parenthetical/angle-bracket annotations: pause durations (`(3 seconds)`),
  the task buzzer (`(buzzer sounds)`), unclear speech (`(?)`), non-speech
  sounds (`(sighs)`, `(laughs)`, `(tuts)`, ...), `<laugh>`/`<lugh>` — all
  stripped as noise before tokenizing.

Only `Pat` turns are scored. `Oth`/`Oth2` turns (a bystander or assistant
speaking) are excluded entirely from scoring, but counted as
`oth_turn_count` metadata, since a clinician reading the score should know
whether the patient had help during the trial.

## Output

- `data/processed/sft_words.csv` — one row per scored item (`word`,
  `classification`), plus `participant_id`/`diagnosis`/`source_file`.
- `data/processed/sft_metrics.csv` — one row per participant: `total_items`,
  `C`, `R`, `OTH`, `F`, their rates, and `oth_turn_count`.

## Classification codes

A deliberately smaller set than PFT_new's Leslie codes — no nonword,
proper-noun, or similarity codes. The animal-name reference list (Zemla et
al. 2020's `animals_snafu_scheme.csv`, extending Troyer, Moscovitch &
Winocur 1997 and Hills, Jones & Todd 2012) is itself the only dictionary
that matters here, so there's no separate "is this a real word" check:

| Code | Meaning |
|---|---|
| `C` | Correct — first mention of a known animal |
| `R` | Repetition of an earlier `C` animal |
| `OTH` | Intrusion — not a known animal |
| `F` | Filled pause (um, er, eh, oh, mm, ...) |

Plural forms are matched to their singular dictionary entry (`cats` →
`cat`), including common irregulars (`mice` → `mouse`) — the scheme only
lists singulars, and participants overwhelmingly say plurals. Multi-word
names (`guinea pig`, `polar bear`) are matched via a greedy longest-match
scan, needed because a naive "whole segment or split on every word"
approach breaks when two items appear with no comma between them and one
of them is itself multi-word (e.g. "mouse guinea pig").

## Trial-level metrics

`compute_metrics()` (`SFT_new/scoring.py`) returns `total_items`, the four
code counts, their per-100-item rates, and `oth_turn_count`.

## Clinician-facing presentation: headline, context, and error analysis

Checked every SFT error/code rate for (a) a monotonic HC → MCI → Dementia
trend and (b) whether it still separates groups after residualizing
against `C` (correct count) — the same method and bar `PFT_new` applied to
its own rates (`pipelines/check_sft_error_rates.py`, mirroring
`docs/pft_new_scoring.md`'s "Error analysis").

- **Headline (banded):** `C` — the standard category-fluency outcome
  measure.
- **Context (unbanded):** `total_items` — all scored items including
  errors; helps distinguish low output overall from plenty of output that
  mostly didn't land on new animals.
- **Headline (banded), error section:** `R_rate` — see below.
- **Exploratory only:** `OTH_rate`, `F_rate`.

## Error analysis

| Rate | Monotonic (raw)? | `d` (HC vs. Dementia, raw) | Monotonic after residualizing `C`? | `d` after residualizing `C` | Verdict |
|---|---|---|---|---|---|
| `R_rate` (repetition) | ✅ (2.95 → 4.67 → 6.02) | large (0.70) | ✅ | 0.48 | **Headline** — genuinely independent signal |
| `OTH_rate` (intrusion) | ✅ | small–moderate (0.38) | ❌ | -0.04 | Exploratory — raw effect fully explained by `C` |
| `F_rate` (filled pause) | ❌ (16.34 → 18.70 → 17.68) | ~none (0.11) | ❌ | -0.07 | Exploratory — no effect either way |

This is the mirror image of PFT, where `OTH_rate` survived and `R_rate`
didn't. Not an inconsistency — a genuine task difference. Increased
repetition (perseveration) with greater severity is a well-documented
marker of executive/retrieval dysfunction in verbal fluency tasks, so
`R_rate` earning headline treatment here rather than `OTH_rate` is a
sensible finding, not a surprising one.

## Not yet implemented

- **Semantic clustering/switching** (grouping consecutive animals sharing a
  Zemla category) — deferred; the reference data (category membership, not
  just a flat animal list) is already loaded and available for when this
  is built.
- **Temporal features** — no word-level timing exists for this task, so
  there's no equivalent of PFT's 15-second-bin analysis.

## Reference group for banding

Unlike PFT and Cookie Theft (which currently band against *all* HC
participants, train and test combined — a known leakage issue, see the
Tier 1 dashboard TODO), SFT's banding uses the **TRAIN-split HC group
only**, so no participant is ever scored against a reference distribution
that includes them or a held-out test case.
