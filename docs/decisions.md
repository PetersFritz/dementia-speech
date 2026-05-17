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