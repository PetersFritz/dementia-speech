"""Phonemic fluency tab module (refactored from original app.py)."""
from collections import Counter

import numpy as np
from shiny import module, ui, render, reactive
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shinywidgets import output_widget, render_widget

from shared import (
    pft_words_df, pft_metrics_df,
    CLASSIFICATION_COLOURS, CLASSIFICATION_LABELS, GROUP_COLOURS, CLUSTER_PALETTE,
)
from dementia_speech.tasks.PFT_new.clustering import assign_cluster_ids
from dementia_speech.tasks.PFT_new.interpretation import (
    HEADLINE_MEASURES, CONTEXT_MEASURES, ERROR_HEADLINE_MEASURES,
    ERROR_EXPLORATORY_MEASURES, EXPLORATORY_MEASURES,
)
from dementia_speech.interpretation import BAND_COLOURS, band

# Reference stats for banding, computed once at import time against the HC
# group — see PFT_new/interpretation.py for why only `C` (and, in the error
# section, `OTH_rate`) are banded, and why several other measures that look
# informative on their own (clustering, switching, mean AoA) aren't.
_HC = pft_metrics_df[pft_metrics_df["diagnosis"] == "HC"]
_ALL_BANDED = HEADLINE_MEASURES + ERROR_HEADLINE_MEASURES
HC_STATS = {m.key: (_HC[m.key].mean(), _HC[m.key].std()) for m in _ALL_BANDED if m.key in _HC}

_GROUPS = ["HC", "MCI", "Dementia"]


def _group_distribution_figure(metric_keys, labels, participant_row):
    """One row per metric: a box plot of each group's actual scores (not the
    group mean's confidence interval) plus a dashed line marking this
    participant's own value.
    """
    fig = make_subplots(rows=len(metric_keys), cols=1,
                        subplot_titles=labels, vertical_spacing=0.06)

    for m_idx, metric in enumerate(metric_keys):
        row = m_idx + 1
        for group in _GROUPS:
            vals = pft_metrics_df.loc[pft_metrics_df["diagnosis"] == group, metric].dropna().values
            if len(vals) < 2:
                continue
            fig.add_trace(go.Box(
                x=vals, y=[group] * len(vals), orientation="h",
                marker_color=GROUP_COLOURS[group], line=dict(width=1.5),
                boxpoints=False, showlegend=False,
            ), row=row, col=1)

        pval = participant_row.get(metric)
        if pval is not None and not (isinstance(pval, float) and np.isnan(pval)):
            fig.add_vline(x=pval, line=dict(color="#2c3e50", width=2, dash="dash"),
                          row=row, col=1)

    fig.update_layout(height=110 * len(metric_keys),
                      margin=dict(t=30, b=10, l=10, r=10),
                      plot_bgcolor="white")
    fig.update_xaxes(gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee")
    return fig


def _measure_card(measure, value, hc_mean, hc_sd, banded=True):
    """One clinical-summary card. banded=False renders a plain context card
    with no Typical/Borderline/Atypical badge (see CONTEXT_MEASURES)."""
    value_str = measure.fmt.format(value) if value is not None and not (isinstance(value, float) and np.isnan(value)) else "—"

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
        ui.div(value_str, style="font-size:1.6em;font-weight:700;margin:2px 0;"),
        ui.div(measure.description, style="color:#666;font-size:0.82em;"),
        ui.div(range_str, style="color:#999;font-size:0.78em;margin-top:4px;"),
        style=(
            "background:white;border:1px solid #e0e0e0;border-left:4px solid "
            f"{colour};border-radius:8px;padding:12px 14px;min-width:12em;flex:1;"
        ),
    )


@module.ui
def phonemic_fluency_ui():
    return ui.div(
        ui.h5("Clinical summary"),
        ui.p(
            "Banded against a healthy-control reference group (within 1 SD = "
            "Typical, 1–2 SD = Borderline, beyond 2 SD = Atypical). Only "
            "\"Words produced correctly\" reliably separates diagnostic groups "
            "on its own in this dataset — clustering, switching, and "
            "vocabulary-difficulty measures turned out to track the same "
            "thing rather than adding independent signal (see "
            "docs/pft_new_scoring.md). Total words is shown for context, not "
            "banded.",
            style="color:#666;font-size:0.9em;max-width:70em;",
        ),
        ui.output_ui("headline_panel"),
        ui.h6("Compared with other participants", style="margin-top:1.2em;"),
        ui.div(
            output_widget("headline_distribution"),
            style="background:white;border:1px solid #e0e0e0;border-radius:8px;padding:1rem;",
        ),
        ui.hr(),
        ui.h5("Error & off-target speech"),
        ui.p(
            "A breakdown of everything that wasn't a correct target-letter "
            "word. Off-target speech rate is banded because it still "
            "distinguishes groups even after accounting for correct-word "
            "count — the rest are shown as raw rates, not clinical flags, "
            "for reasons noted in each caption.",
            style="color:#666;font-size:0.9em;max-width:70em;",
        ),
        ui.output_ui("error_panel"),
        ui.div(
            ui.accordion(
                ui.accordion_panel(
                    "Other error/code rates (exploratory)",
                    ui.p(
                        "Proper-noun rate: ~90% of participants use none at "
                        "all (floor effect). Repetition/similar-word rates: "
                        "structurally biased lower for anyone who produced "
                        "fewer correct words to repeat from. Filled-pause "
                        "rate: doesn't separate groups in this dataset.",
                        style="color:#666;font-size:0.85em;",
                    ),
                    output_widget("error_exploratory_plot"),
                ),
                open=False,
            ),
            style="background:white;border:1px solid #e0e0e0;border-radius:8px;padding:1rem;margin-top:10px;",
        ),
        ui.hr(),
        ui.h5("Transcript"),
        ui.output_ui("word_display"),
        ui.hr(),
        ui.layout_columns(
            ui.div(
                ui.accordion(
                    ui.accordion_panel(
                        "Additional measures (exploratory — interpret with caution)",
                        ui.p(
                            "Clustering, switching, and mean age-of-acquisition "
                            "each looked informative alone, but none differ "
                            "between groups once correct-word count is "
                            "accounted for — they mostly restate reduced "
                            "word retrieval rather than adding new information.",
                            style="color:#666;font-size:0.85em;",
                        ),
                        output_widget("exploratory_plot"),
                    ),
                    open=False,
                ),
                style="background:white;border:1px solid #e0e0e0;border-radius:8px;padding:1rem;",
            ),
            ui.div(
                ui.h5("Words per 15-second bin"),
                output_widget("bin_plot"),
                style="background:white;border:1px solid #e0e0e0;border-radius:8px;padding:1rem;",
            ),
            col_widths=[6, 6],
        ),
    )


@module.server
def phonemic_fluency_server(input, output, session, participant_id):

    @reactive.calc
    def pdata():
        pid = participant_id()
        return pft_words_df[pft_words_df["participant_id"].astype(str) == str(pid)].copy()

    @reactive.calc
    def pmetrics():
        pid = participant_id()
        row = pft_metrics_df[pft_metrics_df["participant_id"].astype(str) == str(pid)]
        return row.iloc[0].to_dict() if len(row) else {}

    @reactive.calc
    def pclusters():
        """Cluster id per row of pdata() (None for non-'C' words), in trial order."""
        df = pdata()
        correct_words = df.loc[df["classification"] == "C", "word"].tolist()
        ids = iter(assign_cluster_ids(correct_words))
        return [next(ids) if cls == "C" else None for cls in df["classification"]]

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
            _measure_card(meas, m.get(meas.key), None, None, banded=False)
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
        return _group_distribution_figure(keys, labels, m)

    @render_widget
    def error_exploratory_plot():
        m = pmetrics()
        if not m:
            return go.Figure()
        keys = list(ERROR_EXPLORATORY_MEASURES.keys())
        labels = list(ERROR_EXPLORATORY_MEASURES.values())
        return _group_distribution_figure(keys, labels, m)

    @render_widget
    def exploratory_plot():
        m = pmetrics()
        if not m:
            return go.Figure()
        keys = list(EXPLORATORY_MEASURES.keys())
        labels = list(EXPLORATORY_MEASURES.values())
        return _group_distribution_figure(keys, labels, m)

    @render.ui
    def word_display():
        df = pdata()
        if len(df) == 0:
            return ui.p("No words found for this participant.")
        cluster_ids = pclusters()
        cluster_sizes = Counter(c for c in cluster_ids if c is not None)
        spans = []
        for (_, row), cluster_id in zip(df.iterrows(), cluster_ids):
            cls = row["classification"]
            colour = CLASSIFICATION_COLOURS.get(cls, "#333")
            label = CLASSIFICATION_LABELS.get(cls, cls)
            title = f"{label} ({row['start']:.2f}s)"
            underline_colour = "transparent"
            if cluster_id is not None:
                underline_colour = CLUSTER_PALETTE[cluster_id % len(CLUSTER_PALETTE)]
                title += f" — cluster {cluster_id + 1} (size {cluster_sizes[cluster_id]})"
            spans.append(ui.span(
                row["word"],
                title=title,
                style=(f"display:inline-block;background-color:{colour}22;color:{colour};"
                       f"border-top:1.5px solid {colour};border-left:1.5px solid {colour};"
                       f"border-right:1.5px solid {colour};border-bottom:4px solid {underline_colour};"
                       f"border-radius:4px;padding:3px 8px 1px 8px;"
                       f"margin:3px 4px;font-weight:500;"),
            ))
        return ui.div(
            *spans,
            ui.p(
                "Coloured underline = phonemic cluster (Troyer et al.); same colour, "
                "consecutive correct words = same cluster. Colours cycle and are only "
                "meaningful within one transcript, not across participants.",
                style="color:#888;font-size:0.85em;margin:8px 4px 0;",
            ),
            style="line-height:2.8;padding:12px;",
        )

    @render_widget
    def bin_plot():
        m = pmetrics()
        if not m:
            return go.Figure()
        bins = ["bin_1", "bin_2", "bin_3", "bin_4"]
        bin_labels = ["0-15s", "15-30s", "30-45s", "45-60s"]
        group_means = pft_metrics_df.groupby("diagnosis")[bins].mean()

        fig = go.Figure()
        for group, colour in GROUP_COLOURS.items():
            if group not in group_means.index:
                continue
            fig.add_trace(go.Scatter(
                x=bin_labels, y=group_means.loc[group, bins].values,
                mode="lines+markers", name=group,
                line=dict(color=colour, width=2, dash="dash"), marker=dict(size=6),
            ))
        fig.add_trace(go.Scatter(
            x=bin_labels, y=[m.get(b, 0) for b in bins],
            mode="lines+markers", name=f"Participant {participant_id()}",
            line=dict(color="#2c3e50", width=3), marker=dict(size=9),
        ))
        fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10),
                          plot_bgcolor="white",
                          xaxis=dict(title="Time bin", gridcolor="#eee"),
                          yaxis=dict(title="Correct words", gridcolor="#eee", rangemode="tozero"),
                          legend=dict(orientation="h", y=-0.2))
        return fig
