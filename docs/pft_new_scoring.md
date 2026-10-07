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
- `mean_aoa_correct` — mean age-of-acquisition rating (years, Kuperman et al.
  2012) across words classified `C` that have a known rating; `None` if none
  of the trial's correct words are in the norms table. `aoa_n` reports how
  many `C` words that mean is based on, since a low `aoa_n` makes it
  unreliable. **Direction (corrected):** a *lower* mean is the impaired
  direction — this project's data shows HC producing correct words with a
  higher mean AoA (7.6 years) than Dementia (7.2), matching the standard
  "vocabulary regression" account: as access to later-acquired vocabulary
  narrows, a speaker's surviving correct output skews toward earlier-
  acquired, more entrenched words. (An earlier version of this doc stated
  the opposite direction — corrected after checking against this project's
  own data; see "Clinician-facing presentation" below.)
- `n_clusters`, `n_switches`, `mean_cluster_size`, `largest_cluster_size` —
  phonemic cluster/switch analysis (Troyer et al., 1997) over the ordered
  sequence of `C` words; see "Phonemic cluster/switch analysis" below.
  Reduced switching is associated with executive dysfunction, reduced
  clustering with degraded phonological/lexical organisation — though see
  "Clinician-facing presentation" for why these don't hold up as
  *independent* markers in this project's data.
- `OTH_rate`, `N_rate`, `PN_rate`, `R_rate`, `SIM_rate`, `F_rate` — each
  code's count expressed as a rate per 100 words, so individuals/groups can
  be compared irrespective of how much they said overall. See "Error
  analysis" below for which of these actually carry signal.

## Phonemic cluster/switch analysis

`compute_cluster_metrics()` (`src/dementia_speech/tasks/PFT_new/clustering.py`)
groups consecutive correct words into clusters of phonemically related words,
then reports:

- `n_clusters` — number of true clusters: maximal runs of **2+** phonemically
  related consecutive words. An isolated word with no relation to either
  neighbour is a singleton, not a cluster, and isn't counted here.
- `n_switches` — transitions between runs of *any* size, including
  singletons (`n_runs - 1`) — moving into or out of a singleton word is
  still a change in retrieval strategy, even though the singleton itself
  isn't a cluster.
- `mean_cluster_size` — average size of the true (2+) clusters; `None` if
  the trial has none.
- `largest_cluster_size` — size of the biggest true cluster; `0` if none.

A word is only added to a run if it's phonemically related to the
*immediately preceding* correct word (`is_phonemic_cluster_match` in
`rules.py`) — grouping is judged on adjacency, not against the whole trial
history, per Troyer et al.'s definition. Two words are related if they:

- share their first two IPA phones (e.g. "penny"/"pencil"), or
- rhyme — share a rime (last vowel phone through the end of the word, e.g.
  "paint"/"faint" would rhyme despite different onsets), or
- are exact homonyms (identical full phone sequence, e.g. "pair"/"pear")

**Rhyme exception:** a rime that's just a bare, coda-less schwa (`/ə/`) never
counts as a rhyme match. This phone set doesn't mark stress, so we can't
distinguish a genuinely rhyming stressed vowel from an incidental unstressed
one -- and a coda-less schwa is specifically the *reduced* vowel. It's also
extremely common: `english_uk_mfa` is non-rhotic, so any "-er"/"-or"/"-ar"
ending ("doctor", "water", "sugar", "camera", "banana", "paper", ...)
collapses to a trailing bare schwa. Without this guard, the rhyme rule
spuriously matched unrelated words like "pizza"/"perpendicular" on nothing
but that shared reduced ending.

This is a conservative subset of Troyer et al.'s four clustering criteria —
it does not implement "differ by exactly one phoneme". A pair with either
word missing from the pronouncing dictionary is never merged into a cluster
(no guessing), which is why cluster counts should be read alongside how much
of the trial's vocabulary actually has a pronunciation (currently ~92% of
`C` words across the corpus).

### IPA pronouncing dictionary

Phone lookups come from `data/raw/reference/english_uk_mfa.dict` (Montreal
Forced Aligner's `english_uk_mfa` model — a UK-English lexicon, chosen
because all participants are UK speakers; this project's TextGrids were
aligned manually, not with MFA, so there was no alignment model to stay
consistent with). Loaded once at import time by `_load_phone_dict()` in
`rules.py`. Some words have multiple weighted pronunciation variants (e.g.
"paint" → /pʰ ej n t/ at 0.99, /pʰ ej n/ at 0.86); only the
highest-probability variant per word is kept. If the file isn't present,
`PHONE_DICT` is empty and every word pair is treated as unrelated (each
correct word becomes its own singleton cluster) rather than erroring.

## Age-of-acquisition norms

Each word also gets an `aoa` column (looked up regardless of its
classification, for inspection) from `get_aoa()` in
`src/dementia_speech/tasks/PFT_new/rules.py`, which reads
`data/raw/reference/AoA_ratings_Kuperman_et_al_BRM.xlsx` (`Word` +
`Rating.Mean` columns; case/whitespace-insensitive column matching) once at
import time. If that file isn't present, `AOA_NORMS` is empty and every
lookup returns `None` — scoring still runs, just without the AoA metric.

## Clinician-facing presentation: headline, context, and error analysis

`PFT_new/interpretation.py` splits the computed measures for the dashboard
(`app/modules/phonemic_fluency.py`), using the same method as the Cookie
Theft task (docs/cookie_theft_scoring.md): check every measure for a
monotonic HC → MCI → Dementia trend, and check whether it still separates
groups once the more obvious explanatory variables are accounted for. Doing
that here turned up something stronger than in Cookie Theft: `C` (correct
words) is such a dominant signal (large effect, d=1.09 HC vs. Dementia,
monotonic, matching decades of letter-fluency literature — e.g. Tombaugh et
al., 1999, COWAT/FAS normative data) that most of the other measures which
looked informative *on their own* turned out to be fully explained by it.
Residualizing `n_clusters`, `largest_cluster_size`, and `mean_aoa_correct`
against `C` (linear regression, then compare group means of the residuals)
collapses every one of their effects to near zero and non-monotonic —
meaning once you know how many words someone got right, these don't tell
you anything more. Presenting them as separate headline flags would
double-count one underlying effect (reduced word retrieval) as if it were
several independent findings.

**Headline** (banded against the HC reference group): `C`, relabelled
"Words produced correctly."

**Context, not banded**: `total_words` (all words spoken, correct or not).
Unlike Cookie Theft, participants tend to keep talking for the full trial
regardless of ability (d=0.04 HC vs. Dementia — essentially flat), so it
doesn't separate groups and isn't banded. It's still shown because it helps
read the headline measure: a low `C` alongside a normal `total_words` means
plenty of attempts that mostly missed the target, which is a different
clinical picture from a low `C` alongside low `total_words` (reduced
output/initiation).

**Exploratory** (raw numbers, not banded, explicitly captioned as
non-independent in this dataset): `n_clusters`, `n_switches`,
`largest_cluster_size`, `mean_aoa_correct`. Real, monotonic, moderate
effects on their own (d=0.31–0.51) — but see the residualization result
above for why they aren't shown as separate clinical flags. Still used for
the coloured phonemic-cluster underlining in the transcript view, which is
a legitimate qualitative/research use distinct from a between-group
clinical flag. `switch_rate` (`n_switches / (C - 1)`, i.e. switches as a
share of all adjacent transitions) was tested as a way to normalize
`n_switches` by output and remove its near-total collinearity with `C`
(r=0.96) — but the correctly-normalized rate turned out not to separate
groups at all (d=-0.15, non-monotonic), so it isn't included: the raw
switch *count* only looked diagnostic because it was standing in for `C`,
and the switching *pattern*, once you control for output, doesn't actually
differ between groups in this cohort.

## Error analysis

The per-code rates (`OTH_rate`, `N_rate`, `PN_rate`, `R_rate`, `SIM_rate`,
`F_rate`) were checked the same way, including whether each one survives
controlling for `C` (since a participant with fewer correct words
necessarily has more "room" in their output for other codes — the question
is which code that slack actually goes into, and whether that's genuine
signal or just arithmetic).

| Rate | Monotonic? | Effect (HC vs. Dementia) | Survives controlling for `C`? | Verdict |
|---|---|---|---|---|
| `OTH_rate` (off-target, doesn't attempt the letter) | ✅ | large (d=0.71) | ✅ (residual d=-0.30, still monotonic) | **Headline** — genuinely independent signal |
| `N_rate` (nonword, attempts the letter) | ✅ | small (d=0.27) | not checked | Exploratory — 53% of participants have zero, moderate floor effect |
| `PN_rate` (proper noun) | ✅ (thin) | small (d=0.32) | not checked | Exploratory — 90% of participants use zero; almost entirely a floor effect, unreliable at individual level |
| `R_rate` (repetition) | ❌ | small, reversed (d=-0.21) | — | Exploratory — repeating a word requires having already said an earlier correct word, so the opportunity to repeat is structurally lower for anyone producing fewer correct words in the first place |
| `SIM_rate` (phonemic/semantic similarity) | ❌ | moderate, reversed (d=-0.35) | — | Exploratory — same structural confound as `R_rate` |
| `F_rate` (filled pause) | ❌ | ~none (d=0.09) | — | Exploratory — doesn't separate groups here |

Only `OTH_rate` earns headline treatment: increasingly severe participants
don't just produce fewer correct words, the words they do produce
disproportionately drift into off-target, non-attempt speech rather than
nonwords, fillers, or repeats — and that pattern holds even after removing
the part of `OTH_rate` explained by `C` alone.

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
