"""Semantic/category fluency (SFT) tab module -- mirrors phonemic_fluency.py's
structure, adapted for what SFT actually has: no word-level timing (manually
transcribed free text, not an aligned TextGrid -- see SFT_new/scoring.py), so
no temporal-bin plot and no timestamp in the transcript tooltip; no semantic
clustering yet (deferred), so no cluster underlining or clustering panel.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shiny import module, ui, render, reactive
from shinywidgets import output_widget, render_widget

from shared import (
    sft_words_df, sft_metrics_df, SPLIT,
    CLASSIFICATION_COLOURS, CLASSIFICATION_LABELS, GROUP_COLOURS,
)
from dementia_speech.tasks.SFT_new.interpretation import (
    HEADLINE_MEASURES, CONTEXT_MEASURES, ERROR_HEADLINE_MEASURES,
    ERROR_EXPLORATORY_MEASURES, EXPLORATORY_MEASURES,
)
from dementia_speech.interpretation import BAND_COLOURS, band

# Reference stats for banding, computed against the TRAIN-split HC group
# only (see shared.SPLIT) -- Cookie Theft's HC_STATS still uses all HC for
# now (separate task). See SFT_new/interpretation.py for why `C` and
# `R_rate` are the two banded measures here, not `OTH_rate` (PFT's banded
# error measure). CONTEXT_MEASURES are included here too so their cards can
# show a z-score alongside the raw value -- still rendered with
# banded=False, so no Typical/Borderline/Atypical judgement is made on them.
_sft_split = sft_metrics_df["participant_id"].astype(str).map(SPLIT)
_HC = sft_metrics_df[(sft_metrics_df["diagnosis"] == "HC") & (_sft_split == "TRAIN")]
_ALL_STATS_MEASURES = HEADLINE_MEASURES + ERROR_HEADLINE_MEASURES + CONTEXT_MEASURES + EXPLORATORY_MEASURES
HC_STATS = {m.key: (_HC[m.key].mean(), _HC[m.key].std()) for m in _ALL_STATS_MEASURES if m.key in _HC}

_GROUPS = ["HC", "MCI", "Dementia"]


def _group_distribution_figure(metric_keys, labels, participant_row):
    """Same approach as phonemic_fluency.py's -- one row per metric, group
    box plots plus a dashed line for this participant's own value."""
    fig = make_subplots(rows=len(metric_keys), cols=1,
                        subplot_titles=labels, vertical_spacing=0.06)

    for m_idx, metric in enumerate(metric_keys):
        row = m_idx + 1
        for group in _GROUPS:
            vals = sft_metrics_df.loc[sft_metrics_df["diagnosis"] == group, metric].dropna().values
            if len(vals) < 2:
                continue
            fig.add_trace(go.Box(
                x=vals, y=[group] * len(vals), orientation="h",
                marker_color=GROUP_COLOURS[group], line=dict(width=1.5),
                boxpoints=False, showlegend=False,
            ), row=row, col=1)

        pval = participant_row.get(metric)
        if pval is not None and not (isinstance(pval, float) and pval != pval):
            fig.add_vline(x=pval, line=dict(color="#1565C0", width=2, dash="dash"),
                          row=row, col=1)

    fig.update_layout(height=110 * len(metric_keys),
                      margin=dict(t=30, b=10, l=10, r=10),
                      plot_bgcolor="white")
    fig.update_xaxes(gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee")
    return fig


def _headline_distribution_figure(metric_keys, labels, participant_row, hc_stats):
    """One row per metric: a violin per group (shape of the distribution,
    PLUS an explicit mean line -- same fix as phonemic_fluency.py's: a
    plain density curve lost the "where's the group's centre" cue a box
    plot gave for free), the HC "typical" range (mean ± 1 SD) shaded, and
    a dashed line marking this participant's own value.
    """
    fig = make_subplots(rows=len(metric_keys), cols=1,
                        subplot_titles=labels, vertical_spacing=0.28)

    for m_idx, metric in enumerate(metric_keys):
        row = m_idx + 1

        hc_mean, hc_sd = hc_stats.get(metric, (None, None))
        if hc_mean is not None and hc_sd:
            fig.add_vrect(x0=hc_mean - hc_sd, x1=hc_mean + hc_sd,
                          fillcolor=GROUP_COLOURS["HC"], opacity=0.12, line_width=0,
                          row=row, col=1)

        for group in _GROUPS:
            vals = sft_metrics_df.loc[sft_metrics_df["diagnosis"] == group, metric].dropna().values
            if len(vals) < 2 or np.std(vals) == 0:
                continue
            fig.add_trace(go.Violin(
                x=vals, y=[group] * len(vals), orientation="h", name=group,
                line_color=GROUP_COLOURS[group], fillcolor=GROUP_COLOURS[group],
                opacity=0.35, meanline_visible=True,
                meanline=dict(color="#111", width=3),
                points=False, spanmode="hard", showlegend=False,
            ), row=row, col=1)
        # HC at top, Dementia at bottom -- Plotly's default categorical
        # y-axis order is "first trace added = bottom", which is the
        # reverse of what we want here.
        fig.update_yaxes(categoryorder="array", categoryarray=["Dementia", "MCI", "HC"],
                          row=row, col=1)

        pval = participant_row.get(metric)
        if pval is not None and not (isinstance(pval, float) and pval != pval):
            fig.add_vline(x=pval, line=dict(color="#1565C0", width=2, dash="dash"),
                          row=row, col=1)

    fig.update_layout(height=180 * len(metric_keys),
                      margin=dict(t=30, b=10, l=10, r=10),
                      plot_bgcolor="white")
    fig.update_xaxes(gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee")
    return fig


def _measure_card(measure, value, hc_mean, hc_sd, banded=True):
    """Same card style as phonemic_fluency.py's."""
    has_value = value is not None and not (isinstance(value, float) and value != value)
    value_str = measure.fmt.format(value) if has_value else "—"
    z_str = f"z={(value - hc_mean) / hc_sd:+.2f}" if has_value and hc_mean is not None and hc_sd else None

    if banded:
        label, _ = band(value, hc_mean, hc_sd, measure.higher_is_worse)
        colour = BAND_COLOURS[label]
        badge = ui.div(
            ui.span(label, style=(
                f"background:{colour};color:white;border-radius:12px;"
                "padding:2px 10px;font-size:0.8em;font-weight:600;"
            )),
            style="margin-bottom:6px;",
        )
    else:
        colour = "#9e9e9e"
        badge = ui.div(
            ui.span("Context (not banded)", style=(
                f"background:{colour};color:white;border-radius:12px;"
                "padding:2px 10px;font-size:0.8em;font-weight:600;"
            )),
            style="margin-bottom:6px;",
        )

    range_str = (
        f"typical range ≈ {measure.fmt.format(max(0, hc_mean - hc_sd))}–{measure.fmt.format(hc_mean + hc_sd)}"
        if hc_mean is not None and hc_sd
        else "reference range unavailable"
    )

    return ui.div(
        badge,
        ui.div(measure.label, style="font-weight:600;font-size:1.05em;"),
        ui.div(
            value_str,
            ui.span(f" ({z_str})", style="font-size:0.6em;font-weight:400;color:#888;") if z_str else None,
            style="font-size:1.6em;font-weight:700;margin:2px 0;",
        ),
        ui.div(measure.description, style="color:#666;font-size:0.82em;"),
        ui.div(range_str, style="color:#999;font-size:0.78em;margin-top:4px;"),
        style=(
            "background:white;border:1px solid #e0e0e0;border-left:4px solid "
            f"{colour};border-radius:8px;padding:12px 14px;min-width:12em;flex:1;"
        ),
    )


@module.ui
def semantic_fluency_ui():
    return ui.div(
        ui.h5("Clinical summary"),
        ui.p(
            "Banded against a TRAIN-split healthy-control reference group "
            "(within 1 SD = Typical, 1–2 SD = Borderline, beyond 2 SD = "
            "Atypical). \"Animals produced correctly\" and the repetition "
            "rate below both reliably separate diagnostic groups on their "
            "own in this dataset — see docs/sft_scoring.md. Total items is "
            "shown for context, not banded.",
            style="color:#666;font-size:0.9em;max-width:70em;",
        ),
        ui.output_ui("headline_panel"),
        ui.h6("Compared with other participants", style="margin-top:1.2em;"),
        ui.div(
            output_widget("headline_distribution"),
            ui.p(
                "Coloured shapes = each group's distribution, with a dark "
                "line at the group's mean; shaded band = HC typical range "
                "(mean ± 1 SD). The grey dashed line is this participant's "
                "own score.",
                style="color:#888;font-size:0.8em;margin:4px 0 0;",
            ),
            style="background:white;border:1px solid #e0e0e0;border-radius:8px;padding:1rem;",
        ),
        ui.hr(),
        ui.h5("Error & repetition"),
        ui.p(
            "A breakdown of everything that wasn't a newly-named animal. "
            "Repetition rate is banded because it still distinguishes "
            "groups after accounting for correct-item count — a "
            "perseveration-like pattern, not just fewer correct items. "
            "Intrusion and filled-pause rates are shown as raw rates, not "
            "clinical flags — see docs/sft_scoring.md.",
            style="color:#666;font-size:0.9em;max-width:70em;",
        ),
        ui.output_ui("error_panel"),
        ui.div(
            ui.accordion(
                ui.accordion_panel(
                    "Other error/code rates (exploratory)",
                    ui.p(
                        "Intrusion rate: a real raw difference between "
                        "groups turned out to be fully explained by "
                        "correct-item count once controlled for. Filled-"
                        "pause rate: doesn't separate groups in this "
                        "dataset.",
                        style="color:#666;font-size:0.85em;",
                    ),
                    output_widget("error_exploratory_plot"),
                ),
                open=False,
            ),
            style="background:white;border:1px solid #e0e0e0;border-radius:8px;padding:1rem;margin-top:10px;",
        ),
        ui.div(
            ui.accordion(
                ui.accordion_panel(
                    "Additional measures (exploratory — interpret with caution)",
                    ui.p(
                        "Mean age-of-acquisition of the animals named "
                        "correctly. Hasn't been checked yet for whether it "
                        "carries signal independent of correct-item count "
                        "in this task — shown for research use, not as a "
                        "clinical flag.",
                        style="color:#666;font-size:0.85em;",
                    ),
                    ui.output_ui("exploratory_panel"),
                    output_widget("exploratory_plot"),
                ),
                open=False,
            ),
            style="background:white;border:1px solid #e0e0e0;border-radius:8px;padding:1rem;margin-top:10px;",
        ),
        ui.hr(),
        ui.h5("Transcript"),
        ui.output_ui("assistance_note"),
        ui.output_ui("word_display"),
    )


@module.server
def semantic_fluency_server(input, output, session, participant_id):

    @reactive.calc
    def pdata():
        pid = participant_id()
        return sft_words_df[sft_words_df["participant_id"].astype(str) == str(pid)].copy()

    @reactive.calc
    def pmetrics():
        pid = participant_id()
        row = sft_metrics_df[sft_metrics_df["participant_id"].astype(str) == str(pid)]
        return row.iloc[0].to_dict() if len(row) else {}

    # ── Clinical summary cards ──
    @render.ui
    def headline_panel():
        m = pmetrics()
        if not m:
            return ui.p("No data for this participant.")
        cards = [
            _measure_card(meas, m.get(meas.key), *HC_STATS.get(meas.key, (None, None)))
            for meas in HEADLINE_MEASURES
        ] + [
            _measure_card(meas, m.get(meas.key), *HC_STATS.get(meas.key, (None, None)), banded=False)
            for meas in CONTEXT_MEASURES
        ]
        return ui.div(*cards, style="display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;")

    @render.ui
    def error_panel():
        m = pmetrics()
        if not m:
            return ui.p("No data for this participant.")
        cards = [
            _measure_card(meas, m.get(meas.key), *HC_STATS.get(meas.key, (None, None)))
            for meas in ERROR_HEADLINE_MEASURES
        ]
        return ui.div(*cards, style="display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;")

    @render_widget
    def headline_distribution():
        m = pmetrics()
        if not m:
            return go.Figure()
        keys = [meas.key for meas in HEADLINE_MEASURES + CONTEXT_MEASURES]
        labels = [meas.label for meas in HEADLINE_MEASURES + CONTEXT_MEASURES]
        return _headline_distribution_figure(keys, labels, m, HC_STATS)

    @render_widget
    def error_exploratory_plot():
        m = pmetrics()
        if not m:
            return go.Figure()
        keys = list(ERROR_EXPLORATORY_MEASURES.keys())
        labels = list(ERROR_EXPLORATORY_MEASURES.values())
        return _group_distribution_figure(keys, labels, m)

    @render.ui
    def exploratory_panel():
        m = pmetrics()
        if not m:
            return ui.p("No data for this participant.")
        cards = [
            _measure_card(meas, m.get(meas.key), *HC_STATS.get(meas.key, (None, None)), banded=False)
            for meas in EXPLORATORY_MEASURES
        ]
        return ui.div(*cards, style="display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;")

    @render_widget
    def exploratory_plot():
        m = pmetrics()
        if not m:
            return go.Figure()
        keys = [meas.key for meas in EXPLORATORY_MEASURES]
        labels = [meas.label for meas in EXPLORATORY_MEASURES]
        return _group_distribution_figure(keys, labels, m)

    @render.ui
    def assistance_note():
        m = pmetrics()
        count = m.get("oth_turn_count") or 0
        if not count:
            return ui.div()
        return ui.p(
            f"Assistance detected during this trial ({int(count)} turn(s) from "
            "someone other than the participant) — excluded from the scoring "
            "above, but worth knowing when interpreting it.",
            style=(
                "background:#fff8e1;border:1px solid #f0d58a;border-radius:6px;"
                "padding:8px 12px;color:#7a5c00;font-size:0.85em;margin-bottom:8px;"
            ),
        )

    @render.ui
    def word_display():
        df = pdata()
        if len(df) == 0:
            return ui.p("No items found for this participant.")
        spans = []
        for _, row in df.iterrows():
            cls = row["classification"]
            colour = CLASSIFICATION_COLOURS.get(cls, "#333")
            label = CLASSIFICATION_LABELS.get(cls, cls)
            spans.append(ui.span(
                row["word"],
                title=label,
                style=(f"display:inline-block;background-color:{colour}22;color:{colour};"
                       f"border:1.5px solid {colour};border-radius:4px;padding:3px 8px;"
                       f"margin:3px 4px;font-weight:500;"),
            ))
        return ui.div(*spans, style="line-height:2.8;padding:12px;")
