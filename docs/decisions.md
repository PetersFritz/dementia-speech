# Decisions Log

## 2026-05-17: TextGrid structure for phonemic fluency

- Tier name: `transcript/word`
- Each interval contains one produced word
- Trial duration: 60 seconds
- Letter: P
- Notes: How to go about compound nouns such as post man (sometimes transcribed as post man and sometimes as postman)? 

## 2026-05-17: Phonemic fluency scoring decisions

- Target letter: P
- Variants: excluded (morphological variants of same root do not count)
- Proper nouns: counted as correct for now — revisit after clinician consultation
- Scoring hierarchy: repetition → variant → intrusion → correct (checked in this order)
- Stemming: NLTK PorterStemmer for variant detection, simple suffix fallback if unavailable
- TextGrid tier name: `transcript/word`
- Output: pft_words.csv (word-level), pft_metrics.csv (participant-level)
- ISSUES 
- [ ] Decide what to do with self talk or questions 
- [ ] compound nouns such as polka dot 
- [ ] count other words besides nouns? 
- [ ] get a UK english dict to check words exist 
- [ ] filler words? 
- [ ] work with timing information as well as clustering (phonetic, semantic, time)

## 2026-05-17: Shiny app — first version

- Framework: Shiny for Python
- App reads pre-computed CSVs from data/processed/ (no computation in app)
- Shows: colour-coded word transcript + 5 summary metric boxes per participant
- Colours: correct=green, intrusion=red, repetition=orange, variant=grey
- Deployment: runs on remote server, accessed via SSH port forwarding locally
- Next: add temporal plot, IWI stats, then move to semantic fluency task

## 2026-05-17: Word cleaning update
- Filter empty intervals AFTER cleaning, not before
- Catches space-only labels that slip through praatio's includeEmptyIntervals=False

## 2026-05-18: End of day log

### Completed today
- [x] Set up repo structure and git/GitHub with SSH
- [x] Built TextGrid parser (load_word_tier) with cleaning:
      - Strip punctuation before scoring
      - Filter empty/space-only intervals after cleaning
- [x] Built phonemic fluency scoring pipeline:
      - Rules: repetition → variant → intrusion → nonword → correct
      - Letter P, variants excluded, proper nouns counted for now
      - WordNet (en) for non-word detection
      - NLTK PorterStemmer for variant detection
- [x] Built pipeline script (score_pft.py) to score all PFT TextGrids
      - Outputs pft_words.csv (word-level) and pft_metrics.csv (participant-level)
      - Metadata joined via str.contains() on 3-digit participant ID
- [x] Built minimal Shiny app (app.py):
      - Colour-coded word transcript per participant
      - Summary metric boxes (total, correct, intrusions, repetitions, variants, nonwords)
      - 15-second temporal bin plot with group mean comparison lines (HC/MCI/Dementia)

### Decisions made
- Proper nouns: counted as correct — revisit after clinician consultation
- Variants: excluded from correct count
- Non-words: detected via WordNet, shown in purple
- Temporal bins: 4 x 15s bins over 60s trial, correct words only
- Metadata ID matching: str.contains() on 3-digit numeric ID

### TODO — next session
- [ ] Validate diagnosis matching (check no Unknowns remain)
- [ ] Validate app runs via SSH port forwarding
- [ ] Add clustering and switching (start with first-n-letters, then slope difference)
- [ ] Revisit proper noun exclusion after clinician input
- [ ] Add IWI statistics to metrics and app
- [ ] Write pytest tests for scoring rules
- [ ] Move to semantic fluency task pipeline

## 2026-05-20: End of day log

### Completed today

#### TextGrid parsing
- Added exclusion of bracketed annotations e.g. [laugh], [sigh] at read-in
  via regex `^\[.*\]$`
- Added exclusion of filled pauses: er, erm, um, umm, uhm, uh, oh, ah, mm, hmm, hm
- Both exclusions happen at read-in in `load_word_tier` before any scoring

#### Shiny app — major updates
- Fixed indentation errors (mixed tabs/spaces) by rewriting app.py cleanly
- Fixed Plotly fillcolor error — 8-digit hex not supported, switched to rgba()
- Added `GROUP_COLOURS` dict at top of file, reused across all plots
- Moved numpy and scipy imports to top of file

#### Summary plot (performance vs group means)
- Replaced line plot with small multiples — one panel per metric
- Each panel shows HC/MCI/Dementia group means with 95% CI error bars
- Participant shown as dashed horizontal line + diamond marker
- Tighter horizontal spacing (0.06) between panels
- Participant marker changed to thick solid line + diamond for visibility

#### Bin plot (words per 15s)
- Kept as line plot (preferred over grouped bars)
- Group mean lines in HC/MCI/Dementia colours
- Participant shown as thicker dark line

#### Radar chart — added new
- Shows all five metrics: correct, intrusions, repetitions, variants, non-words
- Normalisation rules:
  - Axis ceiling = highest group mean for that metric
  - If participant exceeds group ceiling, axis expands to participant value
- Intrusions and repetitions inverted (1 - normalised value) so larger
  shape always means better performance
- Inverted axes labelled with "(inverted)" so clinician is not misled
- Group means shown as dashed lines with light fill
- Participant shown as solid dark line with diamond markers
- Hover tooltips show raw values not normalised values

#### Non-word detection
- WordNet approach confirmed and implemented
- Punctuation stripping fixed — filter applied after cleaning not before
  to catch space-only intervals

### Decisions made
- Filled pauses excluded at read-in, not treated as errors
- Bracketed annotations excluded at read-in
- Radar chart: lower-is-better metrics (intrusions, repetitions) inverted
- Radar normalisation: per-metric scale based on max group mean,
  expanding to participant value if they exceed it
- Line plot kept for temporal bins (preferred over grouped bars)
- Small multiples preferred over overlapping dot plot for metric comparison

### TODO — next session
- [ ] Validate filled pause exclusion list against annotator conventions
- [ ] Run notebook check: non-P words appearing before scoring
- [ ] Come back to clustering (first-n-letters then slope difference)
- [ ] Add IWI statistics to metrics and app
- [ ] Write pytest tests for scoring rules
- [ ] Revisit proper noun exclusion after clinician consultation
- [ ] Move to semantic fluency task pipeline
- [ ] Commit and push all changes from today

## 2026-06-16
markdown## Cookie Theft graph metrics
- Graph metrics computed on full repeat-aware sequence (get_iu_sequence_full)
- IU coverage uses unique first-appearance sequence (get_iu_sequence_unique)
- Limitation: synonym matching only catches lexical repeats, not pronoun
  references (e.g. "he" for boy) — cycle metrics undercount revisits vs human annotator
- Image dimensions read automatically via PIL