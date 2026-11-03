"""Overview tab module — participant demographics and memory-complaint summary."""
import pandas as pd
from shiny import module, ui, render

from shared import (
    DEMOGRAPHICS,
    FINANCE_BY_PARTICIPANT,
    FINANCE_CHANGED_COLOURS,
    FINANCE_CHANGED_LABELS,
    FINANCE_MANAGER_COLOURS,
    FINANCE_MANAGER_LABELS,
    FINANCE_STATUS_COLOURS,
    FINANCE_STATUS_LABELS,
    MEMORY_COMPLAINTS_BY_PARTICIPANT,
    STATUS_COLOURS,
    STATUS_LABELS,
)


@module.ui
def overview_ui():
    return ui.div(
        ui.h5("Demographics"),
        ui.output_ui("demographics_card"),
        ui.h5("Memory Complaints", style="margin-top:20px;"),
        ui.output_ui("memory_complaints_card"),
        ui.h5("Finance", style="margin-top:20px;"),
        ui.output_ui("finance_card"),
    )


@module.server
def overview_server(input, output, session, participant_id):
    """participant_id is a reactive returning the selected participant string."""

    @render.ui
    def demographics_card():
        demo = DEMOGRAPHICS.get(participant_id(), {})
        age = demo.get("age")
        gender = demo.get("gender")
        diagnosis = demo.get("diagnosis")

        def field(label, value):
            return ui.div(
                ui.div(label, style="color:#666;font-size:0.82em;"),
                ui.div(
                    str(value) if pd.notna(value) else "—",
                    style="font-size:1.4em;font-weight:700;",
                ),
            )

        return ui.div(
            field("Age", age),
            field("Gender", gender),
            field("Diagnosis", diagnosis),
            style=(
                "background:white;border:1px solid #e0e0e0;border-radius:8px;"
                "padding:12px 14px;display:flex;gap:24px;max-width:24em;"
            ),
        )

    @render.ui
    def memory_complaints_card():
        findings = MEMORY_COMPLAINTS_BY_PARTICIPANT.get(participant_id())
        if not findings:
            return ui.p(
                "No memory-complaints data for this participant.",
                style="color:#666;font-style:italic;",
            )

        status = findings[0]["status"]
        badge = ui.span(
            STATUS_LABELS.get(status, status),
            style=(
                f"background:{STATUS_COLOURS.get(status, '#999')};color:white;"
                "padding:3px 10px;border-radius:12px;font-size:0.82em;font-weight:600;"
            ),
        )

        if status != "complaint":
            return ui.div(
                badge,
                style=(
                    "background:white;border:1px solid #e0e0e0;border-radius:8px;"
                    "padding:12px 14px;max-width:40em;"
                ),
            )

        finding_rows = [
            ui.div(
                ui.span(
                    f["domain"].replace("_", " ").title(),
                    style=(
                        "background:#f0f0f0;color:#444;padding:2px 8px;"
                        "border-radius:10px;font-size:0.78em;font-weight:600;"
                        "white-space:nowrap;"
                    ),
                ),
                ui.div(f["summary"], style="margin-top:4px;"),
                ui.div(
                    f'"{f["evidence_quote"]}"',
                    style="color:#666;font-style:italic;font-size:0.9em;margin-top:2px;",
                ),
                style="padding:8px 0;border-top:1px solid #eee;",
            )
            for f in findings
            if pd.notna(f["domain"])
        ]

        return ui.div(
            badge,
            *finding_rows,
            style=(
                "background:white;border:1px solid #e0e0e0;border-radius:8px;"
                "padding:12px 14px;max-width:40em;"
            ),
        )

    @render.ui
    def finance_card():
        row = FINANCE_BY_PARTICIPANT.get(participant_id())
        if not row:
            return ui.p(
                "No finance data for this participant.",
                style="color:#666;font-style:italic;",
            )

        status = row["status"]

        if status != "extracted":
            badge = ui.span(
                FINANCE_STATUS_LABELS.get(status, status),
                style=(
                    f"background:{FINANCE_STATUS_COLOURS.get(status, '#999')};color:white;"
                    "padding:3px 10px;border-radius:12px;font-size:0.82em;font-weight:600;"
                ),
            )
            return ui.div(badge, style=(
                "background:white;border:1px solid #e0e0e0;border-radius:8px;"
                "padding:12px 14px;max-width:40em;"
            ))

        # status == "extracted" is the normal case, not worth a badge of its own --
        # only the manager/changed-recently badges below carry useful information.
        badges = []
        if pd.notna(row.get("manager")):
            badges.append(ui.span(
                FINANCE_MANAGER_LABELS.get(row["manager"], row["manager"]),
                style=(
                    f"background:{FINANCE_MANAGER_COLOURS.get(row['manager'], '#999')};"
                    "color:white;padding:3px 10px;border-radius:12px;"
                    "font-size:0.82em;font-weight:600;"
                ),
            ))
        if pd.notna(row.get("changed_recently")):
            badges.append(ui.span(
                FINANCE_CHANGED_LABELS.get(row["changed_recently"], row["changed_recently"]),
                style=(
                    f"background:{FINANCE_CHANGED_COLOURS.get(row['changed_recently'], '#999')};"
                    "color:white;padding:3px 10px;border-radius:12px;"
                    "font-size:0.82em;font-weight:600;"
                ),
            ))

        detail_rows = []
        if pd.notna(row.get("summary")):
            detail_rows.append(ui.div(row["summary"], style="margin-top:8px;"))
        if pd.notna(row.get("change_reason")):
            detail_rows.append(ui.div(row["change_reason"], style="margin-top:4px;color:#666;"))
        if pd.notna(row.get("evidence_quote")):
            detail_rows.append(ui.div(
                f'"{row["evidence_quote"]}"',
                style="color:#666;font-style:italic;font-size:0.9em;margin-top:4px;",
            ))

        return ui.div(
            ui.div(*badges, style="display:flex;gap:6px;flex-wrap:wrap;"),
            *detail_rows,
            style=(
                "background:white;border:1px solid #e0e0e0;border-radius:8px;"
                "padding:12px 14px;max-width:40em;"
            ),
        )
