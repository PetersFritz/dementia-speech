"""Overview tab module — the qualitative task findings (memory awareness,
finance, recent news). Demographics moved to the Summary tab (user request,
2026-11-15).

Same page-header/panel chrome as every other tab (cookie_theft.py,
phonemic_fluency.py, summary.py) -- duplicated rather than shared, same
convention this project already uses for these small per-module style
constants.
"""
import pandas as pd
from shiny import module, ui, render
import plotly.graph_objects as go
from shinywidgets import output_widget, render_widget

from shared import (
    FINANCE_BY_PARTICIPANT,
    FINANCE_CHANGED_COLOURS,
    FINANCE_CHANGED_LABELS,
    FINANCE_MANAGER_COLOURS,
    FINANCE_MANAGER_LABELS,
    FINANCE_STATUS_COLOURS,
    FINANCE_STATUS_LABELS,
    GROUP_COLOURS,
    MEMORY_COMPLAINTS_BY_PARTICIPANT,
    NEWS_BY_PARTICIPANT,
    NEWS_CATEGORY_COLOURS,
    NEWS_CATEGORY_LABELS,
    NEWS_DF,
    STATUS_COLOURS,
    STATUS_LABELS,
    WHO_CONCERNED_BY_PARTICIPANT,
    WHO_CONCERNED_COLOURS,
    WHO_CONCERNED_LABELS,
    WHO_CONCERNED_STATUS_COLOURS,
    WHO_CONCERNED_STATUS_LABELS,
)

_GROUPS = ["HC", "MCI", "Dementia"]

_PANEL_STYLE = (
    "background:white;border:1px solid #e1e8ef;border-top:4px solid #527d96;"
    "border-radius:12px;padding:22px;min-width:0;"
    "box-shadow:0 2px 8px rgba(35,55,75,0.05);"
)
_SECTION_HEADER_STYLE = "font-weight:700;color:#243447;margin-bottom:4px;"
_SECTION_CAPTION_STYLE = "color:#6c757d;font-size:13px;margin-bottom:16px;"
_SUBPANEL_LABEL_STYLE = "font-weight:600;color:#34495e;margin-bottom:6px;"


def _empty_panel(message):
    """Same panel chrome as a populated card, so a participant with no data
    for a given task doesn't suddenly look like a rendering error -- just an
    empty-feeling version of the same box."""
    return ui.div(
        ui.p(message, style="color:#666;font-style:italic;margin:0;"),
        style=_PANEL_STYLE,
    )


def _status_only_panel(label, colour):
    """A panel showing just a status badge -- the "needs_review"/non-
    "extracted" case every one of these cards falls back to when there's
    nothing else worth displaying."""
    badge = ui.span(
        label,
        style=(
            f"background:{colour};color:white;padding:3px 10px;"
            "border-radius:12px;font-size:0.82em;font-weight:600;"
        ),
    )
    return ui.div(badge, style=_PANEL_STYLE)


def _news_group_comparison_figure(participant_score):
    """Per-group violin of the recent-news score (0-3, NEWS_DF's own `score`
    column -- unverifiable/no_date rows excluded via dropna, not counted as
    0, same "missing is never zero" convention used throughout this project),
    plus this participant's own score as a labelled dashed line. Same visual
    language as every other measure card's embedded graph (violin per group,
    mean line, labelled dashed participant line, no hover). The score only
    takes 4 discrete values, so the violins are smoothed with a wide
    bandwidth and clipped to the 0-3 range -- read them as a rough picture of
    where each group's scores sit, not exact proportions (the in-window rate
    under the graph is the exact figure).
    """
    fig = go.Figure()
    scored = NEWS_DF.dropna(subset=["score"])
    for group in _GROUPS:
        vals = scored.loc[scored["diagnosis"] == group, "score"].values
        if len(vals) < 2 or vals.std() == 0:
            continue
        fig.add_trace(go.Violin(
            x=vals, y=[group] * len(vals), orientation="h", name=group,
            line_color=GROUP_COLOURS[group], fillcolor=GROUP_COLOURS[group],
            opacity=0.35, meanline_visible=True,
            meanline=dict(color="#111", width=2),
            points=False, spanmode="hard", bandwidth=0.45, showlegend=False,
            hoverinfo="skip",
        ))
    fig.update_yaxes(categoryorder="array", categoryarray=["Dementia", "MCI", "HC"])
    if participant_score is not None and pd.notna(participant_score):
        fig.add_vline(x=participant_score, line=dict(color="#0D47A1", width=3, dash="dash"), annotation_text="Participant", annotation_position="top", annotation_font=dict(size=10, color="#0D47A1"))
    fig.update_layout(height=166, margin=dict(t=22, b=24, l=6, r=6), plot_bgcolor="white")
    fig.update_xaxes(gridcolor="#eee", range=[-0.1, 3.1], tickvals=[0, 1, 2, 3], tickfont=dict(size=10),
                     title=dict(text="recall score (0-3)", font=dict(size=10)))
    fig.update_yaxes(gridcolor="#eee", tickfont=dict(size=10))
    return fig


def _news_in_window_rate_text() -> str:
    """'% of each group whose answer matched a real, in-window event',
    counted over EVERY participant in that group (including unscoreable
    ones, unlike the mean-score bar chart above, which excludes them) --
    the plainest, most clinically readable single number this task
    produces: a straight hit rate out of everyone who attempted it."""
    rates = NEWS_DF.groupby("diagnosis")["category"].apply(
        lambda s: (s == "in_window_match").mean() * 100
    )
    parts = [f"{g} {rates[g]:.0f}%" for g in _GROUPS if g in rates.index]
    return "Correctly recalled an in-window event: " + " · ".join(parts)


@module.ui
def overview_ui():
    return ui.div(
        # PAGE HEADER -- same chrome as every other tab.
        ui.div(
            ui.div(
                ui.h3("Memory & Daily Life", style="font-weight:700;color:#243447;margin:0;"),
                ui.p(
                    "The qualitative task findings.",
                    style="color:#6c757d;margin:4px 0 0 0;",
                ),
            ),
            ui.div(
                ui.span(
                    "CLINICAL ASSESSMENT",
                    style=(
                        "font-size:11px;font-weight:700;letter-spacing:1px;"
                        "color:#426b83;background:#eaf2f6;"
                        "padding:7px 12px;border-radius:20px;"
                    ),
                ),
            ),
            style=(
                "display:flex;justify-content:space-between;"
                "align-items:center;flex-wrap:wrap;"
                "padding:22px 26px;margin-bottom:22px;"
                "background:white;border:1px solid #e5eaf0;"
                "border-radius:14px;"
            ),
        ),

        # Paired side-by-side, not stacked -- these two answer related
        # questions (does the participant report a concern themselves; who
        # does the transcript say is actually worried) and a clinician
        # scanning for agreement/disagreement between them shouldn't have
        # to scroll to compare the two.
        ui.div(
            ui.h5("Memory awareness", style=_SECTION_HEADER_STYLE),
            ui.p(
                "Self-reported complaint vs. who the transcript says is "
                "actually concerned, side by side.",
                style=_SECTION_CAPTION_STYLE,
            ),
            ui.layout_columns(
                ui.div(
                    ui.h6("Self-reported complaint", style=_SUBPANEL_LABEL_STYLE),
                    ui.output_ui("memory_complaints_card"),
                ),
                ui.div(
                    ui.h6("Who is concerned", style=_SUBPANEL_LABEL_STYLE),
                    ui.output_ui("who_concerned_card"),
                ),
                col_widths=[6, 6],
            ),
            style="margin-bottom:26px;",
        ),

        ui.div(
            ui.h5("Finance", style=_SECTION_HEADER_STYLE),
            ui.output_ui("finance_card"),
            style="margin-bottom:26px;",
        ),

        ui.div(
            ui.h5("Recent news", style=_SECTION_HEADER_STYLE),
            ui.p(
                "Exploratory: date-matched against a recent-news event "
                "cache, not yet hand-validated against this project's own "
                "HC/MCI/Dementia groups -- see docs/news_scoring_rules.md.",
                style=_SECTION_CAPTION_STYLE,
            ),
            ui.layout_columns(
                ui.div(
                    ui.h6("Recall accuracy", style=_SUBPANEL_LABEL_STYLE),
                    ui.output_ui("news_score_card"),
                ),
                ui.div(
                    ui.h6("Answer detail", style=_SUBPANEL_LABEL_STYLE),
                    ui.output_ui("news_detail_card"),
                ),
                col_widths=[5, 7],
            ),
        ),

        style="background:#f5f7fa;padding:22px;border-radius:14px;",
    )


@module.server
def overview_server(input, output, session, participant_id):
    """participant_id is a reactive returning the selected participant string."""

    @render.ui
    def memory_complaints_card():
        findings = MEMORY_COMPLAINTS_BY_PARTICIPANT.get(participant_id())
        if not findings:
            return _empty_panel("No memory-complaints data for this participant.")

        status = findings[0]["status"]
        badge = ui.span(
            STATUS_LABELS.get(status, status),
            style=(
                f"background:{STATUS_COLOURS.get(status, '#999')};color:white;"
                "padding:3px 10px;border-radius:12px;font-size:0.82em;font-weight:600;"
            ),
        )

        if status != "complaint":
            return ui.div(badge, style=_PANEL_STYLE)

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

        return ui.div(badge, *finding_rows, style=_PANEL_STYLE)

    @render.ui
    def finance_card():
        row = FINANCE_BY_PARTICIPANT.get(participant_id())
        if not row:
            return _empty_panel("No finance data for this participant.")

        status = row["status"]

        if status != "extracted":
            return _status_only_panel(
                FINANCE_STATUS_LABELS.get(status, status),
                FINANCE_STATUS_COLOURS.get(status, "#999"),
            )

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
            style=_PANEL_STYLE,
        )

    @render.ui
    def news_score_card():
        row = NEWS_BY_PARTICIPANT.get(participant_id())
        if not row:
            return _empty_panel("No recent-news data for this participant.")

        # "needs_review" (see score_news.py) isn't in the normal category
        # set -- same no-badge-of-its-own-category-detail shortcut the
        # other cards use for their own non-"extracted" status. "no_date"
        # IS a normal category (see shared.NEWS_CATEGORY_LABELS) so it goes
        # through the value-tile path below like any other category.
        if row.get("status") == "needs_review":
            return _status_only_panel("Needs review", "#BA7517")

        category = row.get("category")
        score = row.get("score")
        has_score = pd.notna(score)

        value_tile = ui.div(
            ui.div(
                NEWS_CATEGORY_LABELS.get(category, category),
                style=(
                    "font-size:15px;font-weight:700;"
                    f"color:{NEWS_CATEGORY_COLOURS.get(category, '#333')};"
                ),
            ),
            ui.div(
                f"{int(score)}/3" if has_score else "— / 3",
                style="font-size:28px;font-weight:700;color:#212529;margin:4px 0;",
            ),
            ui.div(
                "Not scoreable -- a gap in the measure, not a wrong answer."
                if not has_score else
                "3 = recent event · 2 = older event · 1 = vague · 0 = no answer",
                style="font-size:12px;color:#6c757d;",
            ),
        )

        return ui.div(
            value_tile,
            ui.div(output_widget("news_comparison_graph"), style="margin-top:10px;"),
            ui.div(_news_in_window_rate_text(), style="color:#888;font-size:0.78em;margin-top:6px;"),
            style=_PANEL_STYLE,
        )

    @render_widget
    def news_comparison_graph():
        row = NEWS_BY_PARTICIPANT.get(participant_id())
        score = row.get("score") if row else None
        return _news_group_comparison_figure(score if pd.notna(score) else None)

    @render.ui
    def news_detail_card():
        row = NEWS_BY_PARTICIPANT.get(participant_id())
        if not row:
            return _empty_panel("No recent-news data for this participant.")

        if row.get("status") == "needs_review":
            return _status_only_panel("Needs review", "#BA7517")

        detail_rows = []
        if pd.notna(row.get("assessment_date")):
            detail_rows.append(ui.div(
                f"Assessment date: {row['assessment_date']}",
                style="color:#888;font-size:0.82em;",
            ))
        if pd.notna(row.get("summary")):
            detail_rows.append(ui.div(row["summary"], style="margin-top:8px;"))
        if pd.notna(row.get("matched_event_headline")):
            detail_rows.append(ui.div(
                f'Matched event: "{row["matched_event_headline"]}" '
                f'({row.get("matched_event_date", "?")}, {row.get("matched_event_source", "?")})',
                style="color:#444;font-size:0.9em;margin-top:4px;",
            ))
        if pd.notna(row.get("evidence_quotes")):
            detail_rows.append(ui.div(
                f'"{row["evidence_quotes"]}"',
                style="color:#666;font-style:italic;font-size:0.9em;margin-top:4px;",
            ))
        # Concrete details given about the matched event (verbatim from the
        # transcript), shown as raw evidence for the reader to judge -- no
        # derived "level" or group comparison (not validated). Falls back to a count-only note if the scores file predates the
        # best_claim_details column.
        details_text = row.get("best_claim_details")
        if pd.notna(details_text) and str(details_text).strip():
            items = [d.strip() for d in str(details_text).split(" | ") if d.strip()]
            detail_rows.append(ui.div(
                ui.div("Specific details given", style=_SUBPANEL_LABEL_STYLE + "margin-top:10px;"),
                ui.tags.ul(
                    *[ui.tags.li(f'"{d}"') for d in items],
                    style="margin:0;padding-left:18px;color:#444;font-size:0.9em;",
                ),
            ))
        elif pd.notna(row.get("best_claim_detail_count")) and int(row["best_claim_detail_count"]) == 0:
            detail_rows.append(ui.div(
                "No specific details beyond naming the topic.",
                style="color:#666;font-style:italic;font-size:0.9em;margin-top:8px;",
            ))
        if not detail_rows:
            detail_rows.append(ui.p(
                "No further detail extracted for this participant.",
                style="color:#666;font-style:italic;margin:0;",
            ))

        return ui.div(*detail_rows, style=_PANEL_STYLE)

    @render.ui
    def who_concerned_card():
        row = WHO_CONCERNED_BY_PARTICIPANT.get(participant_id())
        if not row:
            return _empty_panel("No who-is-concerned data for this participant.")

        status = row["status"]

        if status != "extracted":
            return _status_only_panel(
                WHO_CONCERNED_STATUS_LABELS.get(status, status),
                WHO_CONCERNED_STATUS_COLOURS.get(status, "#999"),
            )

        # status == "extracted" is the normal case, not worth a badge of its own --
        # only the who_concerned badge below carries useful information.
        badges = []
        if pd.notna(row.get("who_concerned")):
            badges.append(ui.span(
                WHO_CONCERNED_LABELS.get(row["who_concerned"], row["who_concerned"]),
                style=(
                    f"background:{WHO_CONCERNED_COLOURS.get(row['who_concerned'], '#999')};"
                    "color:white;padding:3px 10px;border-radius:12px;"
                    "font-size:0.82em;font-weight:600;"
                ),
            ))

        detail_rows = []
        if pd.notna(row.get("summary")):
            detail_rows.append(ui.div(row["summary"], style="margin-top:8px;"))
        if pd.notna(row.get("evidence_quote")):
            detail_rows.append(ui.div(
                f'"{row["evidence_quote"]}"',
                style="color:#666;font-style:italic;font-size:0.9em;margin-top:4px;",
            ))

        return ui.div(
            ui.div(*badges, style="display:flex;gap:6px;flex-wrap:wrap;"),
            *detail_rows,
            style=_PANEL_STYLE,
        )
