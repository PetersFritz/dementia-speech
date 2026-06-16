"""Score all Cookie Theft transcripts: IU coverage + spatio-semantic graph metrics."""
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
    get_iu_sequence_unique,
    get_iu_sequence_full,
)
from dementia_speech.tasks.picture_description.graph import (
    CIU_COORDINATES,
    calculate_metrics,
    IMAGE_PATH,
)
 
TRANSCRIPT_DIR = Path("data/raw/text/CTD")  # adjust to your path
METADATA_PATH  = Path("data/raw/meta-info.csv")
OUTPUT_IU      = Path("data/processed/cookie_theft_iu.csv")
OUTPUT_METRICS = Path("data/processed/cookie_theft_metrics.csv")
 
 
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
 
            # Coverage metrics
            metrics = compute_iu_metrics(iu_results)
 
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
            row["iu_sequence_unique"] = json.dumps(seq_unique)
            row["iu_sequence_full"]   = json.dumps(seq_full)
            row["iu_sequence_readable"] = " > ".join(seq_unique)
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