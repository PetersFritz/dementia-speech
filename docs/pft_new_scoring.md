# PFT scoring (PFT_new): what is scored and how

This describes the rebuilt phonemic fluency task (PFT) scoring pipeline in
`src/dementia_speech/tasks/PFT_new/` (entry point: `pipelines/score_pft_new.py`).
It scores letter-fluency trials (default target letter: **P**) against the
Leslie scoring criteria.

## Input

A trial can be loaded from either of two sources:

- **A Praat TextGrid** with a `transcript/word` tier, one interval per
  spoken word/token. `load_word_tier` (`src/dementia_speech/io/textgrid.py`)
  turns this into one row per word with columns `word, start, end, duration,
  iwi` (`iwi` = inter-word interval, the gap since the previous word's start).
- **A plain-text transcript** containing only the participant's spoken
  words, whitespace-separated, with no speaker labels or timestamps.
  `load_words_from_txt` (`src/dementia_speech/io/text.py`) turns this into
  one row per word with just a `word` column — no timing.

Both loaders clean tokens the same way (`clean_token` in
`src/dementia_speech/io/cleaning.py`):
- Punctuation is stripped, text is lowercased; internal hyphens are kept
  (e.g. `part-time` survives as one word) but leading/trailing hyphens are
  removed.
- Empty tokens are dropped.

`load_word_tier` additionally drops empty/whitespace-only intervals and
bracketed annotations (`[laugh]`, `[sigh]`, etc.) — these don't arise in
plain text since it never carries them in the first place.

Filled pauses and discourse markers (`um`, `well`, `so`, ...) are **not**
dropped at either loading stage — they flow into scoring and are
classified there.

`score_trial` only ever reads the `word` column, so scoring is identical
regardless of which loader produced the input — a text-sourced trial just
has empty `start`/`end`/`duration`/`iwi` columns in the output.
`pipelines/score_pft_new.py` (this project's pipeline) only reads
TextGrids; `load_words_from_txt` is available for anyone working from
plain-text transcripts to call directly, or wire into their own pipeline.

## Output

`score_trial()` (`src/dementia_speech/tasks/PFT_new/scoring.py`) adds a set
of boolean feature columns plus one final `classification` column per word.
`pipelines/score_pft_new.py` runs this over every TextGrid in
`data/raw/textgrids/PFT`, attaches `participant_id`, `diagnosis` (looked up
from `data/raw/meta-info.csv`) and `source_file`, and writes one long
word-level table to `data/processed/pft_words_new.csv`.

`compute_metrics()` collapses a scored trial into per-code counts
(`total_words`, `C`, `OTH`, `N`, `PN`, `R`, `SIM`, `F`, and `errors`).

## Classification codes (Leslie code system)

Each word gets exactly one code, decided by checking rules **in this order**
per word (first match wins):

| Order | Code | Meaning | Rule |
|---|---|---|---|
| 1 | `F` | Filled pause | Word is in the filled-pause list (`um`, `er`, `uh`, `ah`, `hmm`, ...) |
| 2 | `OTH` | Other / non-attempt speech | Doesn't start with the target letter — **regardless** of whether it's a real word, a closed-class function word, or gibberish |
| 3 | `PN` | Proper noun | Starts with the target letter and is a recognised person name or country name |
| 4 | `R` | Repetition | Starts with the target letter, is a real word, and exactly repeats a word already scored `C` earlier in the same trial |
| 5 | `SIM` | Similar form/meaning | Starts with the target letter, is a real word, not an exact repeat, but shares a word stem or a WordNet synset with an earlier `C` word |
| 6 | `C` | Correct | Starts with the target letter, is a real word, not a repeat or near-repeat of an earlier correct word |
| 7 | `N` | Nonword | Starts with the target letter but isn't a recognised real word (a failed attempt at the target letter, not off-target speech) |

`N` is now reserved for words that **do** start with the target letter but
aren't real words. Any word that doesn't start with the target letter — a
real word, a function word/discourse marker, or pure gibberish — is scored
`OTH` and never `N`.

Only words classified `C` are added to the trial's running "correct words"
history — `R` and `SIM` are checked **only against that history**, never
against earlier errors, matching the Leslie definition of R as "repetition
of a previous *correct* word."

### Not yet implemented

The pipeline can currently only assign `C, OTH, N, PN, R, SIM, F`. Leslie
codes `SC_CORR, LP, FRAG, SC_ERR, SC_META, META, TG` are not implemented.
Any wrong-letter word (Leslie's `L`, whether a real word, function word, or
nonword) is deliberately folded into `OTH` rather than kept as its own
code. `BGN` markers and unclear speech (`?`) are dropped upstream during
TextGrid loading and never reach the scorer.

## How each rule decides

- **Starts with target letter** (`starts_with_target_letter`): first
  alphabetic character of the lowercased word, compared orthographically
  (not phonetically) to the target letter.

- **Real word** (`is_real_word`): a word counts as real if it has WordNet
  synsets *or* appears in the NLTK words corpus (~237k entries). Single
  letters are always rejected first (avoids spurious hits like `p` matching
  the element symbol phosphorus). No frequency filtering, so some
  obscure/archaic entries are accepted.

- **Filled pause** (`is_filler`): exact match against a fixed list (`er`,
  `erm`, `um`, `umm`, `uh`, `oh`, `ah`, `mm`, `hmm`, `phew`, `pooh`, ...).
  Distinct from discourse markers (`well`, `so`, `right`, `yeah`, `no`),
  which are deliberately left out of this list and scored through the
  normal hierarchy instead.

- **Person name** (`is_person_name`): word must appear in a name set built
  from ONS baby name registrations 1996–2025 (`data/raw/reference/babynames1996to2025.xlsx`,
  boys + girls sheets), AND have no "ordinary" WordNet sense (a sense that
  isn't an instance/proper-noun hypernym) — this avoids flagging names that
  are also common words (e.g. someone named "Grace" isn't necessarily a
  name-attempt). A small manual override list (`paddy`, `pauline`,
  `penelope`, `peter`, `phoebe`) forces these to always count as names,
  since automatic checking got them wrong.

- **Country name** (`is_country_name` / `find_country_phrase_indices`):
  matched against `pycountry`'s country list. Single-word countries
  (`portugal`) are checked per-word; multi-word countries (`papua new
  guinea`) are matched as contiguous phrases across the word sequence and
  every word in the matched span is flagged.

- **Repetition** (`is_repetition`): exact (case/whitespace-normalized)
  string match against words already classified `C` earlier in the trial.

- **Similar form or meaning** (`is_similar_form_or_meaning`): compares the
  current word to every earlier `C` word two ways —
  1. same Porter stem (catches inflectional/derivational variants, e.g.
     `paint` / `painting`), or
  2. shares a WordNet synset (catches exact synonyms, e.g. `pretty` /
     `pleasant`, only when they co-occur in at least one sense's synonym
     set).
  This is intentionally conservative — it does not catch broader semantic
  relatedness such as hypernyms/hyponyms, only literal shared stems or
  shared synsets. An exact match to the earlier word itself is excluded
  here (that's `R`, checked first).

## Trial-level metrics

`compute_metrics()` summarizes a scored trial as counts of each code plus:

- `total_words` — all scored words in the trial
- `errors` — `total_words` minus `C`, `F`, and `OTH` (i.e. `N + PN + R +
  SIM`). Filled pauses and other/non-attempt speech are treated as neither
  correct responses nor scoring errors.

## Worked example

For the target letter **P**, a trial containing `um`, `paint`, `paint`,
`painting`, `peter`, `banana`, `xyzzy` would score as:

| Word | Code | Why |
|---|---|---|
| `um` | `F` | Filled pause |
| `paint` | `C` | Starts with P, real word, first occurrence |
| `paint` | `R` | Exact repeat of an earlier `C` word |
| `painting` | `SIM` | Shares Porter stem `paint` with the earlier `C` word `paint` |
| `peter` | `PN` | Starts with P, in the manual name-override list |
| `banana` | `OTH` | Doesn't start with P (real word, but off-target) |
| `xyzzy` | `OTH` | Doesn't start with P (nonword, but off-target) |
