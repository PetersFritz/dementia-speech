"""Compute the full list of 27 candidate extra measures for the Cookie Theft
tab (see docs/to_do/TODO_dashboard.md for the list with suggested dashboard
labels/priority), for inspection before deciding which to actually add.

This is additive and read-only with respect to the existing pipeline --
score_cookie_theft.py and its outputs are untouched. Writes one wide CSV to
data/processed/ with everything computed per participant, plus NaN/flag
columns where a measure's underlying data is missing (e.g. no pause
annotations in that transcript) so "missing" isn't silently read as zero.

Not computed (see candidate_measures.py docstring for why): semantic errors,
referential clarity -- both need manual annotation or an LLM-based judgement
against the picture's ground truth, a scoping decision, not a parameter.
"""
import os
from pathlib import Path

import pandas as pd
from PIL import Image

project_root = Path(__file__).parent.parent
os.chdir(project_root)

from dementia_speech.tasks.picture_description.scoring import (
    load_transcript,
    extract_information_units,
    compute_iu_metrics,
    compute_efficiency_metrics,
)
from dementia_speech.tasks.picture_description.registry import COORDINATES_RAW
from dementia_speech.tasks.picture_description.graph import IMAGE_PATH
from dementia_speech.tasks.picture_description.candidate_measures import (
    compute_syllable_metrics,
    compute_utterance_metrics,
    compute_region_metrics,
    compute_grammar_complexity_metrics,
    compute_repetition_metrics,
    compute_text_pause_metrics,
    compute_filled_pause_metrics,
    compute_noun_verb_ratio,
    compute_audio_duration_metrics,
)
from dementia_speech.tasks._shared.manual_transcript import parse_speaker_turns
from dementia_speech.paths import DATA_DIR

TRANSCRIPT_DIR = DATA_DIR / "raw/process/CTD"
WAV_DIR        = DATA_DIR / "raw/process/wav/CTD"
METADATA_PATH  = DATA_DIR / "raw/meta-info.csv"
OUTPUT_PATH    = DATA_DIR / "processed/cookie_theft_candidate_measures.csv"


def extract_participant_id(filename) -> str:
    """Same convention as score_cookie_theft.py's extract_participant_id."""
    parts = filename.stem.split("__")
    return parts[1] if len(parts) >= 2 else filename.stem


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(METADATA_PATH)

    with Image.open(IMAGE_PATH) as img:
        img_width, img_height = img.size

    txt_files = sorted(TRANSCRIPT_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} transcripts")

    rows = []
    for txt_path in txt_files:
        participant_id = extract_participant_id(txt_path)
        raw_text = txt_path.read_text(encoding="utf-8", errors="replace")
        transcript = load_transcript(txt_path)

        iu_results = extract_information_units(transcript)
        iu_metrics = compute_iu_metrics(iu_results)
        eff_metrics = compute_efficiency_metrics(transcript, iu_results)
        total_words = eff_metrics["total_words"]

        row = {"participant_id": participant_id}

        match = metadata[metadata["IDs"].astype(str).str.contains(participant_id)]
        row["diagnosis"] = match.iloc[0]["diagnosis"] if len(match) > 0 else "Unknown"

        # 1, 4 (duplicate of 1 under current keyword-match scoring -- see
        # candidate_measures module docstring / report), 6
        row["total_content_units"] = iu_metrics["mentioned_ius"]
        row["correct_information_units"] = iu_metrics["mentioned_ius"]  # alias, see report
        row["total_words"] = total_words

        # 3, 13 -- CIU-per-word already exists as iu_mentions_per_100_words;
        # surfaced again here under the candidate-list name for comparison
        row["content_accuracy_ciu_per_100_words"] = eff_metrics["iu_mentions_per_100_words"]

        # 2, 14, 15, 16, 17
        row.update(compute_region_metrics(iu_results, COORDINATES_RAW, img_width, img_height))

        # 5, 11
        syll_metrics = compute_syllable_metrics(transcript)
        row.update(syll_metrics)
        cus = row["total_content_units"]
        row["syllables_per_content_unit"] = (
            round(syll_metrics["total_syllables"] / cus, 2) if cus else None
        )

        # 9, 10
        row.update(compute_utterance_metrics(transcript, total_words))

        # 18 is already in cookie_theft_metrics.csv (mattr) -- not duplicated here

        # 19 (word-finding) -- filled-pause rate; indefinite_term_rate already
        # exists in lexical.py, not duplicated here
        row.update(compute_filled_pause_metrics(transcript, total_words))

        # 20, 21
        row.update(compute_grammar_complexity_metrics(transcript))

        # 22
        row.update(compute_repetition_metrics(transcript, total_words))

        # 25 (text-annotation pauses -- separate from audio-derived timing below)
        row.update(compute_text_pause_metrics(raw_text))

        # 26 (pronoun_ratio already exists in lexical.py, not duplicated here) 27
        row.update(compute_noun_verb_ratio(transcript))

        # 7, 8, 12 -- needs matching wav
        wav_path = WAV_DIR / (txt_path.stem + ".wav")
        if wav_path.exists():
            audio_metrics = compute_audio_duration_metrics(wav_path)
            row.update(audio_metrics)
            dur_min = (
                audio_metrics["speech_duration_seconds"] / 60
                if audio_metrics.get("speech_duration_seconds") else None
            )
            row["words_per_minute"] = round(total_words / dur_min, 1) if dur_min else None
            row["syllables_per_minute"] = (
                round(syll_metrics["total_syllables"] / dur_min, 1) if dur_min else None
            )
            row["content_units_per_minute"] = round(cus / dur_min, 2) if dur_min else None
        else:
            row["speech_duration_seconds"] = None
            row["total_audio_duration_seconds"] = None
            row["words_per_minute"] = None
            row["syllables_per_minute"] = None
            row["content_units_per_minute"] = None
            row["wav_missing"] = True

        # Flag the small minority where audio-derived duration is at risk of
        # including non-participant speech (see candidate_measures docstring)
        other_speaker_turns = any(
            speaker != "Pat" for speaker, _ in parse_speaker_turns(raw_text)
        )
        row["has_other_speaker_in_transcript"] = other_speaker_turns

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    main()
