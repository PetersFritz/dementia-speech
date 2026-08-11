"""Aggregate per-token study session files into analysis-ready CSVs."""
import json
import os
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
os.chdir(project_root)

SESSIONS_DIR = Path("data/study/sessions")
METADATA_PATH = Path("data/raw/meta-info.csv")
OUTPUT_DECISIONS = Path("data/processed/study_decisions.csv")
OUTPUT_SUMMARY = Path("data/processed/study_summary.csv")
OUTPUT_QUESTIONNAIRE = Path("data/processed/study_questionnaire.csv")


def load_true_diagnoses() -> dict[int, str]:
    meta = pd.read_csv(METADATA_PATH, dtype={"IDs": str})
    meta["participant_id"] = meta["IDs"].str.split("__").str[-1].astype(int)
    return dict(zip(meta["participant_id"], meta["diagnosis"]))


def main():
    OUTPUT_DECISIONS.parent.mkdir(parents=True, exist_ok=True)
    true_diagnosis = load_true_diagnoses()

    session_files = sorted(SESSIONS_DIR.glob("*.json"))
    if not session_files:
        print(f"No session files found in {SESSIONS_DIR}")
        return

    decision_rows = []
    summary_rows = []
    questionnaire_rows = []

    for path in session_files:
        with open(path, "r", encoding="utf-8") as f:
            session = json.load(f)

        token = session["token"]
        n_assigned = len(session["assigned_cases"])
        decisions = session["decisions"]
        questionnaire = session.get("questionnaire")

        if questionnaire is not None:
            questionnaire_rows.append({
                "token": token,
                "submitted_at": questionnaire["submitted_at"],
                **questionnaire["answers"],
            })

        n_correct = 0
        confidences = []
        for d in decisions:
            true_dx = true_diagnosis.get(d["participant_id"])
            correct = d["diagnosis_guess"] == true_dx
            n_correct += correct
            confidences.append(d["confidence"])
            decision_rows.append({
                "token": token,
                "position": d["position"],
                "participant_id": d["participant_id"],
                "true_diagnosis": true_dx,
                "diagnosis_guess": d["diagnosis_guess"],
                "correct": correct,
                "confidence": d["confidence"],
                "submitted_at": d["submitted_at"],
            })

        summary_rows.append({
            "token": token,
            "n_assigned": n_assigned,
            "n_completed": len(decisions),
            "complete": len(decisions) >= n_assigned,
            "accuracy": n_correct / len(decisions) if decisions else None,
            "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
            "questionnaire_complete": questionnaire is not None,
            "created_at": session["created_at"],
        })

    pd.DataFrame(decision_rows).to_csv(OUTPUT_DECISIONS, index=False)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_SUMMARY, index=False)
    pd.DataFrame(questionnaire_rows).to_csv(OUTPUT_QUESTIONNAIRE, index=False)

    print(f"Wrote {len(decision_rows)} decisions from {len(session_files)} raters to {OUTPUT_DECISIONS}")
    print(f"Wrote per-rater summary to {OUTPUT_SUMMARY}")
    print(f"Wrote {len(questionnaire_rows)} questionnaire responses to {OUTPUT_QUESTIONNAIRE}")


if __name__ == "__main__":
    main()
