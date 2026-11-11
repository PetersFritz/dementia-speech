"""Score all Cookie Theft transcripts: IU coverage + spatio-semantic graph metrics.

Reads data/raw/process/CTD/ -- manually-transcribed free text, not the Whisper
ASR output in data/raw/text/CTD/ (same switch SFT already made, and for the
same reason: scoring shouldn't run on uncorrected ASR text when a manual
transcript exists). See picture_description/scoring.py::load_transcript for
the Pat:/Oth:/Int: speaker-turn and annotation stripping this relies on.
"""
import os
import json
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
    compute_semantic_relevance,
    compute_iu_mention_density,
    get_iu_sequence_unique,
    get_iu_sequence_full,
)
from dementia_speech.tasks.picture_description.graph import (
    CIU_COORDINATES,
    calculate_metrics,
    IMAGE_PATH,
)
from dementia_speech.tasks.picture_description.lexical import compute_lexical_metrics
from dementia_speech.tasks.picture_description.registry import COORDINATES_RAW
from dementia_speech.tasks.picture_description.candidate_measures import (
    compute_region_metrics,
    compute_repetition_metrics,
    compute_noun_verb_ratio,
    compute_syllable_metrics,
    compute_audio_duration_metrics,
)
from dementia_speech.paths import DATA_DIR

TRANSCRIPT_DIR = DATA_DIR / "raw/process/CTD"
WAV_DIR        = DATA_DIR / "raw/process/wav/CTD"
METADATA_PATH  = DATA_DIR / "raw/meta-info.csv"
OUTPUT_IU      = DATA_DIR / "processed/cookie_theft_iu.csv"
OUTPUT_METRICS = DATA_DIR / "processed/cookie_theft_metrics.csv"
 
 
def extract_participant_id(filename) -> str:
    """Extract 3-digit participant ID from filename — adjust to your convention."""
    parts = filename.stem.split("__")
    return parts[1] if len(parts) >= 2 else filename.stem
 
 
def main():
    OUTPUT_IU.parent.mkdir(parents=True, exist_ok=True)
 
    metadata = pd.read_csv(METADATA_PATH)
 
    # Read image dimensions once, automatically
    with Image.open(IMAGE_PATH) as img:
        img_width, img_height = img.size
    print(f"Image dimensions: {img_width} x {img_height}")
 
    all_iu_rows = []
    all_metrics = []
 
    txt_files = sorted(TRANSCRIPT_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} transcript files")
 
    for txt_path in txt_files:
        participant_id = extract_participant_id(txt_path)
        print(f"Processing participant {participant_id}...")
 
        try:
            transcript = load_transcript(txt_path)
            iu_results = extract_information_units(transcript)
 
            # Coverage, efficiency, and lexical-semantic metrics
            metrics = compute_iu_metrics(iu_results)
            metrics.update(compute_efficiency_metrics(transcript, iu_results))
            metrics["semantic_relevance"] = compute_semantic_relevance(
                metrics["mentioned_ius"], metrics["total_words"]
            )
            metrics["iu_mention_density"] = compute_iu_mention_density(
                metrics["total_mentions"], metrics["total_words"]
            )
            metrics.update(compute_lexical_metrics(transcript))
 
            # Two sequences
            seq_unique = get_iu_sequence_unique(iu_results)   # for display / coverage order
            seq_full   = get_iu_sequence_full(iu_results)     # for graph / cycle metrics
 
            # Graph metrics computed on the full (repeat-aware) sequence
            graph_metrics = calculate_metrics(
                seq_full, CIU_COORDINATES, img_width, img_height
            ) if seq_full else {}
 
            # Diagnosis lookup
            match = metadata[metadata["IDs"].astype(str).str.contains(participant_id)]
            diagnosis = match.iloc[0]["diagnosis"] if len(match) > 0 else "Unknown"
 
            # IU-level rows (one per IU)
            for iu_name, result in iu_results.items():
                all_iu_rows.append({
                    "participant_id": participant_id,
                    "diagnosis":      diagnosis,
                    "iu":             iu_name,
                    "present":        result["present"],
                    "count":          result["count"],
                    "evidence":       ", ".join(result["evidence"]),
                    "first_position": result["positions"][0] if result["positions"] else None,
                })
 
            # Combine all participant-level metrics into one row
            row = {
                "participant_id": participant_id,
                "diagnosis":      diagnosis,
            }
            row.update(metrics)
            row.update(graph_metrics)

            # Length-normalized graph metrics: raw cycles/path_distance/
            # self_cycles/unique_nodes correlate 0.6-0.87 with total_words in
            # this cohort, so a participant who simply talks less looks
            # "better" on every one of them regardless of actual cognitive
            # content. Dividing by total_words removes that confound — see
            # docs/cookie_theft_scoring.md "Replication note".
            total_words = metrics.get("total_words") or 0
            if total_words > 0 and graph_metrics:
                row["cycles_per_100_words"] = round(100 * graph_metrics["cycles"] / total_words, 2)
                row["path_distance_per_100_words"] = round(100 * graph_metrics["total_path_distance"] / total_words, 1)
            else:
                row["cycles_per_100_words"] = None
                row["path_distance_per_100_words"] = None

            row["iu_sequence_unique"] = json.dumps(seq_unique)
            row["iu_sequence_full"]   = json.dumps(seq_full)
            row["iu_sequence_readable"] = " > ".join(seq_unique)

            # Additional measures (2026-11-11 dashboard extension) -- see
            # docs/cookie_theft_scoring.md for how these were screened
            # (TRAIN/TEST direction-stability, not just full-sample AUC).
            region_metrics = compute_region_metrics(iu_results, COORDINATES_RAW, img_width, img_height)
            row["content_units_by_region"] = json.dumps(region_metrics["content_units_by_region"])
            row["left_right_cu_ratio"] = region_metrics["left_right_cu_ratio"]

            row.update(compute_repetition_metrics(transcript, total_words))
            row.update(compute_noun_verb_ratio(transcript))

            # syllables_per_content_unit only -- total_syllables itself isn't
            # shown standalone, it tracks total_words too closely to earn its
            # own dashboard slot (2026-11-17 decision).
            total_syllables = compute_syllable_metrics(transcript)["total_syllables"]
            mentioned_ius = metrics["mentioned_ius"]
            row["syllables_per_content_unit"] = (
                round(total_syllables / mentioned_ius, 2) if mentioned_ius else None
            )

            # Speech duration from the matching wav -- energy-based silence
            # trim, not forced alignment/diarization (see candidate_measures
            # module docstring for the interviewer-contamination caveat on
            # the small minority of recordings with an Oth/Int speaker).
            wav_path = WAV_DIR / (txt_path.stem + ".wav")
            if wav_path.exists():
                audio_metrics = compute_audio_duration_metrics(wav_path)
                row["speech_duration_seconds"] = audio_metrics["speech_duration_seconds"]
                dur_min = (
                    audio_metrics["speech_duration_seconds"] / 60
                    if audio_metrics.get("speech_duration_seconds") else None
                )
                row["content_units_per_minute"] = (
                    round(mentioned_ius / dur_min, 2) if dur_min else None
                )
            else:
                row["speech_duration_seconds"] = None
                row["content_units_per_minute"] = None

            all_metrics.append(row)
 
        except Exception as e:
            print(f"  ERROR on {txt_path.name}: {e}")
            continue
 
    iu_df = pd.DataFrame(all_iu_rows)
    iu_df.to_csv(OUTPUT_IU, index=False)
    print(f"\nSaved IU data: {OUTPUT_IU} ({len(iu_df)} rows)")
 
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_METRICS, index=False)
    print(f"Saved metrics: {OUTPUT_METRICS} ({len(metrics_df)} rows)")
    print("\nDone! Preview:")
    print(metrics_df.head())
 
 
if __name__ == "__main__":
    main()