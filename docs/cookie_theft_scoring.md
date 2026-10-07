# Cookie Theft scoring: what is scored and how

This describes the Cookie Theft picture-description scoring pipeline in
`src/dementia_speech/tasks/picture_description/` (entry point:
`pipelines/score_cookie_theft.py`). It scores a plain-text transcript of a
participant describing the Boston Diagnostic Aphasia Examination "Cookie
Theft" picture against four measure domains: **IU coverage**, **efficiency**,
**lexical-semantic content**, and a **spatio-semantic graph** built from the
order in which information units (IUs) are mentioned.

The measures are deliberately kept as a multi-domain profile rather than
collapsed into a single composite score — a clinician should see *which*
domain is abnormal (sparse content? verbose but low-yield? empty/vague
vocabulary? disorganised visual scanning?) rather than one opaque number,
since each domain points to a different underlying deficit and the
literature behind them was validated domain-by-domain, not as a combined
index.

## Input

A transcript is plain text (no timestamps, no speaker labels) — one
participant's full response to the picture, loaded and lowercased by
`load_transcript()`. Unlike the PFT pipeline, no TextGrid/timing source
exists for Cookie Theft in this project, so nothing here depends on
duration, speech rate, or pause timing.

## The IU registry

`registry.py` is the single source of truth for the 20 information units
(IUs) scored: each one's synonym list (for text matching), its category
(`person`/`object`/`action`), and its pixel coordinates on
`CookieTheft.png`. `scoring.py`'s IU-matching and `graph.py`'s coordinate
lookups both import from here — previously these were two separately
hardcoded dicts that had to be kept in sync by hand.

**Why 20 IUs, not more.** Ambadi et al. (2021, *Frontiers in Neurology*,
"Spatio-Semantic Graphs From Picture Description: Applications to Detection
of Cognitive Impairment") — the paper the graph metrics below are drawn from
— use a 23-item CIU list: the same 16 person/object units used here, plus 4
action units where this project only defines the concrete 4 (`Steal`,
`Fall`, `Wash`, `Overflow`). Their remaining 3 are propositional/pragmatic:
the girl's action, the woman being unconcerned by the overflowing sink, and
the woman being indifferent to the children's danger. Missing exactly this
kind of pragmatic inference is identified by Cummings (2019, *Pragmatics &
Society*, "Describing the Cookie Theft picture: Sources of breakdown in
Alzheimer's dementia") as a specific marker of AD — so these units are
clinically meaningful — but they are not implemented here because keyword/
synonym matching can't detect them reliably: a transcript can state the
underlying fact ("the sink is overflowing") without any hedge word a rule
could key on, so the unit would show near-universal absence across every
diagnostic group rather than genuine signal. Capturing them properly would
need semantic/LLM-based extraction rather than a synonym list. This project
also deliberately keeps the 20-item list for continuity with prior analyses
that used it; switching to 23 (or any different IU list) would break
comparability with earlier results and isn't done lightly.

`extract_information_units()` finds every occurrence of every synonym (not
just the first), giving both a coverage view (`get_iu_sequence_unique`, one
entry per IU ordered by first mention) and a repeat-aware sequence
(`get_iu_sequence_full`, every mention in transcript order) — the latter is
what the graph and cycle metrics are built from.

## Domain 1: IU coverage

`compute_iu_metrics()` — unchanged from before this round of work:
`total_ius`, `mentioned_ius`, `iu_score` (mentioned/total), `total_mentions`.
This is a lightweight version of the classic Correct Information Unit
approach (Nicholas & Brookshire, 1993, "A system for quantifying the
informativeness and efficiency of connected speech"): coverage answers "did
they get to this content at all," irrespective of how much talking it took.

## Domain 2: Efficiency

`compute_efficiency_metrics()` (`scoring.py`) answers the question coverage
can't: how much verbal output did it take to reach that content? Two
participants can reach identical `iu_score` while one needed twice as many
words — verbose, low-yield speech is itself a discourse marker (Nicholas &
Brookshire, 1993).

- `total_words` — word count via `tokenize_words()` (letters/apostrophes
  only), shared with the lexical measures below so word counts are
  consistent across every measure that needs one.
- `iu_mentions_per_100_words` — total IU mentions per 100 words spoken.
  Nicholas & Brookshire's original efficiency measure is CIUs-per-*minute*,
  using recording duration; that's unavailable here since these transcripts
  carry no timing, so this uses CIUs-per-*word* instead. Lower values mean
  more of the participant's speech went to circumlocution, repetition, or
  off-topic content relative to on-target content.

In this project's data (n=400: 200 HC / 150 MCI / 50 Dementia),
`total_words` declines with severity (HC 170 → MCI 150 → Dementia 126 mean
words) — reduced verbal output is itself a well-documented marker, distinct
from the informativeness-per-word question `iu_mentions_per_100_words`
addresses.

## Domain 3: Lexical-semantic content

`lexical.py`, independent of the IU checklist — these are general
connected-speech measures (Boschi et al., 2017, *Frontiers in Psychology*,
"Connected Speech in Neurodegenerative Language Disorders: A Review") that
would apply to any transcript, not just Cookie Theft descriptions.

- `lexical_density` — (nouns + lexical verbs + adjectives + adverbs) / total
  words, via nltk POS tagging (Johansson-style formula; Halliday, 1985; Ure,
  1971). Auxiliary/copula verbs (`is`, `has been`, `would`, ...) are
  excluded from the verb count by lexical lookup, since POS tags alone
  don't distinguish them from lexical verbs. Lower density means
  proportionally more function words — "emptier" speech associated with AD.
- `pronoun_ratio` — pronouns (`PRP`, `PRP$`, `WP`, `WP$`) / total words.
- `indefinite_term_rate` — rate per 100 words of vague placeholder nouns
  (`thing`, `stuff`, `something`, `somebody`, ...), deliberately excluding
  demonstratives (`it`, `this`, `that`) since those are already counted
  under `pronoun_ratio`. A higher rate suggests word-finding difficulty
  being masked by a placeholder rather than the target content word (Ash et
  al., 2006, "Trying to tell a tale: discourse impairments in progressive
  aphasia and frontotemporal dementia"; Giles et al., 1996, empty speech in
  early AD).
- `mattr` (+ `mattr_window`) — Moving-Average Type-Token Ratio (Covington &
  McFall, 2010), a sliding-window lexical-diversity measure. Plain
  type-token ratio is confounded by sample length (longer samples
  mechanically produce lower TTR), which would bias any comparison across
  participants who simply talk for different lengths of time — MATTR
  removes that confound by averaging TTR over a fixed window (default 25
  words) rather than the whole transcript. Transcripts shorter than the
  window fall back to plain TTR over the whole transcript, flagged via
  `mattr_window` (equal to the transcript length rather than 25) so these
  can be filtered out of comparisons if needed.

In this project's data, `mattr` declines with severity as expected (HC
0.827 → MCI 0.820 → Dementia 0.814) and `indefinite_term_rate` rises as
expected (HC 0.25 → MCI 0.31 → Dementia 0.37 per 100 words). `lexical_density`
and `pronoun_ratio` did not show the monotonic pattern the literature
predicts in this cohort — plausibly because shorter, more telegraphic
Dementia-group transcripts drop function words disproportionately (raising
relative content-word density) rather than adding more of them; this is
worth treating as an open question for this cohort specifically rather than
assumed to replicate the (different-population) literature by default.

## Domain 4: Spatio-semantic graph

`graph.py` builds a directed graph from the repeat-aware IU mention
sequence, using each IU's picture coordinates, and computes graph-theoretic
metrics. This is a direct implementation of Ambadi et al. (2021) — see
above for the citation. Their study (n=1,349, WRAP/DementiaBank) found
**six of the twelve metrics computed here were statistically significant**
(p<0.01) between cognitively impaired and unimpaired speakers:

| Metric | Direction in impairment | Interpretation |
|---|---|---|
| `avg_x` | lower (more left) | spatial neglect — despite the picture's right side carrying more content |
| `stdev_x` | higher | more scattered horizontal attention |
| `total_path_distance` | higher | more visual/semantic scanning overall |
| `total_path_per_unique_nodes` | higher | less efficient path per unit of content covered |
| `cycles` | higher | more repeated (non-consecutive) mentions — memory lapses |
| `cross_ratio_quadrants` | higher | more jumping between quadrants — attentional fragmentation |

The remaining metrics (`avg_y`, `stdev_y`, `self_cycles`, `nodes`,
`unique_nodes`, `self_cycles_quadrants`) are **descriptive/exploratory** —
Ambadi et al. did not report them as diagnostic, so read them as context
rather than evidence for a single case.

`spatial_left_bias` is an added, purely derived convenience metric — a
sign-oriented, normalized re-expression of `avg_x`
(`(image_width/2 - avg_x) / (image_width/2)`, positive = left of the
picture's midline) — for readability; it carries no information beyond
`avg_x` itself.

**Replication note.** In this project's data, `cycles` runs in the
*opposite* direction to Ambadi et al. (HC mean 9.2 > MCI 7.2 > Dementia
5.6), and `spatial_left_bias` doesn't cleanly separate the three groups
either. `cycles` here is a raw count, not normalized by verbal output —
and in this cohort the HC group talks *more* overall (mean 170 words vs.
126 for Dementia), so raw cycle counts are confounded with `total_words` in
the same way raw type-token ratio is confounded with sample length (see
`mattr` above). A length-normalized cycle rate (e.g. cycles per 100 words,
analogous to `iu_mentions_per_100_words`) would be a natural next step if
this metric is going to be used for between-group comparison in this
cohort specifically — not added here without checking first, since
`cycles` is already an established metric from the literature and changing
its definition should be a deliberate choice, not a silent patch.

## Clinician-facing presentation: headline vs. exploratory

`interpretation.py` splits the computed measures into two tiers for the
dashboard (`app/modules/cookie_theft.py`), based on which ones actually
behave sensibly *in this project's own data* — checking each measure for
(a) a monotonic HC → MCI → Dementia trend and (b) a direction matching the
literature it's drawn from turned up a real problem: several of the raw
graph counts correlate 0.6–0.87 with `total_words` in this cohort, so a
participant who simply talks less automatically looks "better" on them
regardless of cognitive content (see "Replication note" above). Presenting
those as confident Typical/Atypical clinical flags would overstate what
they support.

**Headline measures** (banded against the HC reference group, shown as the
"Clinical summary" cards): `iu_score` (content coverage), `total_words`
(verbal output), `path_distance_per_100_words` (visual scanning effort —
the word-length-normalized version of `total_path_distance`, not the raw
count), `mattr` (vocabulary variety), `indefinite_term_rate` (vague word
use). Each is banded as **Typical** (within 1 SD of the HC mean),
**Borderline** (1–2 SD, in the impaired direction), or **Atypical** (beyond
2 SD) — standard neuropsychological test-interpretation convention (cf.
Lezak, Howieson & Loring, *Neuropsychological Assessment*), applied
one-directionally so scoring unusually *well* still reads as Typical.

**Exploratory measures** (raw numbers only, in a collapsed "Additional
measures" section, explicitly captioned as not clinically diagnostic in
this dataset): `iu_mentions_per_100_words`, `lexical_density`,
`pronoun_ratio`, `cycles_per_100_words`, `total_path_per_unique_nodes`,
`cross_ratio_quadrants`, `spatial_left_bias`. Raw `cycles`,
`total_path_distance`, `self_cycles`, and `unique_nodes` are no longer
shown in the dashboard at all (the length-confound above applies to them
directly) — they remain in the CSV for research use only.

**Known limitation of `indefinite_term_rate` specifically:** 71% of
participants in this cohort use *zero* indefinite/vague terms in their
description (the wordlist is deliberately narrow — see "Domain 3" above),
so most individual scores land exactly at the HC-relative z implied by a
value of 0, which reads as "Typical" for the large majority of every
diagnostic group. The population-level trend is real (monotonic, matches
the literature) but this floor effect means the measure will rarely flag
Borderline/Atypical for any individual — it behaves as a low-sensitivity,
high-specificity marker (informative when triggered, silent for most
people) rather than one that discriminates every case. This is presented
honestly (a 0 really is typical here) rather than adjusted, but is worth
knowing before reading too much into a "Typical" band on this measure
specifically.

## Worked interpretation

A participant profile is read across domains, not as a single score:

- **High IU coverage, high word count, low efficiency, low lexical
  density** → talks fluently and at length but the content is diffuse
  relative to its length — circumlocution rather than word-finding failure.
- **Low IU coverage, low word count, but high efficiency** → sparse but
  on-target — reduced verbal output/initiation rather than confused
  content.
- **High `indefinite_term_rate`, low `mattr`, otherwise-typical coverage**
  → vocabulary-access difficulty being masked by vague substitutions,
  worth flagging even where the checklist coverage looks fine.
- **Low `avg_x` / high `spatial_left_bias`, high `cross_ratio_quadrants`,
  high `total_path_per_unique_nodes`** → visuospatial scanning/organisation
  deficit, independent of how much content was named — this is the profile
  Ambadi et al. specifically link to cognitive impairment, distinct from
  the content-based domains above.

## Known limitations

- No timing data for Cookie Theft transcripts in this project (unlike PFT),
  so no speech-rate or pause-based measures are possible without new data
  collection (e.g. TextGrids or ASR timestamps).
- The 3 pragmatic/inferential CIUs from Ambadi et al.'s full 23-item list
  are not implemented (see "The IU registry" above) — a genuine coverage
  gap versus the literature, not an oversight.
- `cycles` and `spatial_left_bias` did not replicate Ambadi et al.'s
  direction of effect in this cohort (see "Replication note" above);
  treat these two with more caution than the others in this dataset until
  investigated further.
- Lexical measures use nltk's POS tagger, which is not specifically tuned
  for disfluent/spontaneous speech transcripts — tagging errors on
  fragments, false starts, or ungrammatical output will introduce some
  noise into `lexical_density` and `pronoun_ratio`.
