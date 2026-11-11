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
  `C`, `R`, `OTH`, `N`, `F`, their rates, and `oth_turn_count`.

## Classification codes

A smaller set than PFT_new's Leslie codes — still no proper-noun or
similarity codes (no clean animal-naming analog for "morphological/synonym
variant of an earlier correct word," and proper-noun detection was
deliberately left out of the 2026-11-17 extension below, by request). The
animal-name reference list (Zemla et al. 2020's `animals_snafu_scheme.csv`,
extending Troyer, Moscovitch & Winocur 1997 and Hills, Jones & Todd 2012)
is still the dictionary for *correct* responses, but a nonword code was
added to distinguish two different kinds of non-animal item:

| Code | Meaning |
|---|---|
| `C` | Correct — first mention of a known animal |
| `R` | Repetition of an earlier `C` animal |
| `N` | Nonword — not a known animal, and not a real English word either (fragment/false start, e.g. "alpac", "de-de", "ka") |
| `OTH` | Intrusion — not a known animal, but a real word or narration aside (e.g. "car", "I can't think") |
| `F` | Filled pause (um, er, eh, oh, mm, ...) |

`N` vs. `OTH` reuses `PFT_new.rules.is_real_word()` (WordNet + nltk's
`words` corpus) on any single non-matching word; a multi-word non-matching
segment (a narration aside like "and through Africa we went") is always
`OTH`, never `N` — it's narration, not one failed word-finding attempt.
`is_real_word()`'s dictionaries don't cover contractions or possessives at
all ("can't", "it's", "cetti's" all fail it, same as a genuine fragment
would) — rare enough in PFT's rapid-fire word lists to go unnoticed there,
but ~45% of this task's narration-heavy `N` items before a fix
(`SFT_new/scoring.py::_is_real_word_or_contraction`, which accepts a word
ending `'s`/`'t`/`'re`/`'ve`/`'ll`/`'d`/`'m` as real regardless of its stem).

Plural forms are matched to their singular dictionary entry (`cats` →
`cat`), including common irregulars (`mice` → `mouse`) — the scheme only
lists singulars, and participants overwhelmingly say plurals. Multi-word
names (`guinea pig`, `polar bear`) are matched via a greedy longest-match
scan, needed because a naive "whole segment or split on every word"
approach breaks when two items appear with no comma between them and one
of them is itself multi-word (e.g. "mouse guinea pig").

## Trial-level metrics

`compute_metrics()` (`SFT_new/scoring.py`) returns `total_items`, the four
code counts, their per-100-item rates, `oth_turn_count`, and (added
2026-11-17, mirroring `PFT_new`'s own `errors`/`error_rate`) `errors` /
`error_rate` — `R + OTH + N` (everything except correct responses and
filled pauses; SFT has no `PN`/`SIM` codes) and that count as a percentage
of `total_items`.

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
- **Exploratory only:** `OTH_rate`, `N_rate`, `F_rate`.

## Error analysis

| Rate | Monotonic (raw)? | `d` (HC vs. Dementia, raw) | Monotonic after residualizing `C`? | `d` after residualizing `C` | Verdict |
|---|---|---|---|---|---|
| `R_rate` (repetition) | ✅ (2.95 → 4.67 → 6.02) | large (0.70) | ✅ | 0.48 | **Headline** — genuinely independent signal |
| `OTH_rate` (intrusion) | ✅ | small–moderate (0.34) | ❌ | -0.07 | Exploratory — raw effect fully explained by `C` |
| `N_rate` (nonword) | ✅ (1.53 → 1.77 → 2.40) | small (0.29) | ❌ | 0.14 | Exploratory — same verdict as PFT's own `N_rate` |
| `F_rate` (filled pause) | ❌ (16.34 → 18.70 → 17.68) | ~none (0.11) | ❌ | -0.07 | Exploratory — no effect either way |

(`OTH_rate`'s numbers moved slightly from an earlier version of this table
— `N_rate` didn't exist yet, so fragments/false starts were still counted
inside `OTH_rate`.)

This is the mirror image of PFT, where `OTH_rate` survived and `R_rate`
didn't. Not an inconsistency — a genuine task difference. Increased
repetition (perseveration) with greater severity is a well-documented
marker of executive/retrieval dysfunction in verbal fluency tasks, so
`R_rate` earning headline treatment here rather than `OTH_rate` is a
sensible finding, not a surprising one.

## Semantic clustering/switching

Implemented 2026-11-17 (`SFT_new/clustering.py`), mirroring
`PFT_new/clustering.py`'s structure exactly but grouping consecutive
correct animal names by shared Zemla et al. (2020, SNAFU) category instead
of phonemic similarity — the category scheme was already loaded
(`SFT_new/rules.py::get_categories`) but unused until now. Checked the same
way as the error rates above:

| Measure | Monotonic (raw)? | `d` (HC vs. Dementia, raw) | Corr. with `C` | Monotonic after residualizing `C`? | `d` after residualizing `C` | Verdict |
|---|---|---|---|---|---|---|
| `n_clusters` | ✅ (5.70 → 4.51 → 3.94) | large (-1.01) | 0.80 | ✅ | -0.16 | Exploratory — mostly explained by `C`, same pattern as PFT's own phonemic clustering |
| `n_switches` | ✅ | moderate (-0.67) | 0.69 | ❌ (reversed) | 0.08 | Exploratory — fully explained by `C`, same pattern as PFT's `switch_rate` test |
| `mean_cluster_size` | ✅ | moderate (-0.43) | 0.34 | ✅ | -0.11 | Exploratory — small residual effect, comparable to SFT's own `F_rate` "no real effect" |
| `largest_cluster_size` | ✅ | moderate (-0.66) | 0.53 | ❌ (reversed) | -0.13 | Exploratory — fully explained by `C` |

None clear this project's bar for an independent signal. This is a
genuinely informative result, not just a replication of PFT's: category-
fluency *switching* specifically is the measure the wider literature (e.g.
Ramanan et al., 2015) reports as most robust for discriminating AD from
MCI, so it was a real candidate for headline status here — it just doesn't
survive controlling for output in this cohort. Kept computed and saved in
`sft_metrics.csv` (available for research use) but not shown as a dashboard
measure, same treatment PFT_new gives its own clustering/switching.

## Not yet implemented

- **Temporal features** — no word-level timing exists for this task, so
  there's no equivalent of PFT's 15-second-bin analysis or response-latency
  measures.

## Reference group for banding

Unlike PFT and Cookie Theft (which currently band against *all* HC
participants, train and test combined — a known leakage issue, see the
Tier 1 dashboard TODO), SFT's banding uses the **TRAIN-split HC group
only**, so no participant is ever scored against a reference distribution
that includes them or a held-out test case.
