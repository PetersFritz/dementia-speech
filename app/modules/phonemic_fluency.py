"""Phonemic fluency tab module (refactored from original app.py)."""
import numpy as np
import pandas as pd
from shiny import module, ui, render, reactive
import plotly.graph_objects as go
from shinywidgets import output_widget, render_widget

from shared import (
    pft_words_df, pft_metrics_df, SPLIT, DEMOGRAPHICS,
    CLASSIFICATION_COLOURS, CLASSIFICATION_LABELS, GROUP_COLOURS,
)
from dementia_speech.tasks.PFT_new.interpretation import (
    HEADLINE_MEASURES, CONTEXT_MEASURES, ERROR_HEADLINE_MEASURES,
    ERROR_EXPLORATORY_MEASURES, EXPLORATORY_MEASURES,
)
from dementia_speech.interpretation import fit_age_adjusted_norms, percentile_age_adjusted

# Age-adjusted reference stats, computed once at import time against the
# TRAIN-split HC group only (see shared.SPLIT), so no participant is ever
# scored against a distribution that includes them or a held-out test case
# -- same fix already applied to SFT and the discrepancy score. See
# PFT_new/interpretation.py for why only `C` (and, in the error section,
# `OTH_rate`) separate groups reliably, and why several other measures that
# look informative on their own (clustering, switching, mean AoA) don't. See
# fit_age_adjusted_norms() for why the reference is a value predicted from
# the participant's own age rather than one flat HC mean/SD.
_pft_split = pft_metrics_df["participant_id"].astype(str).map(SPLIT)
_HC = pft_metrics_df[(pft_metrics_df["diagnosis"] == "HC") & (_pft_split == "TRAIN")]
_HC_AGES = _HC["participant_id"].astype(str).map(lambda pid: DEMOGRAPHICS.get(pid, {}).get("age"))
# CONTEXT_MEASURES are included here too so their cards can show a percentile
# alongside the raw value, same card recipe as HEADLINE_MEASURES -- whether a
# given measure actually separates diagnostic groups is explained in its own
# description text, not a separate badge (see _measure_card).
_ALL_STATS_MEASURES = (
    HEADLINE_MEASURES + ERROR_HEADLINE_MEASURES + CONTEXT_MEASURES
    + ERROR_EXPLORATORY_MEASURES + EXPLORATORY_MEASURES
)
HC_STATS = {
    m.key: fit_age_adjusted_norms(_HC[m.key], _HC_AGES)
    for m in _ALL_STATS_MEASURES if m.key in _HC
}

# All non-key measures, keyed by measure key, for additional_measures_panel
# below.
_ADDITIONAL_MEASURES_BY_KEY = {
    m.key: m for m in ERROR_HEADLINE_MEASURES + ERROR_EXPLORATORY_MEASURES + EXPLORATORY_MEASURES
}
# Fixed display order (user request, 2026-11-17): measures PFT and SFT both
# compute come first, in the SAME order for both tasks, so they land at the
# same position in each task's "Additional measures" panel and line up
# horizontally when the two tasks are shown side by side. PFT-only measures
# (SFT has no PN/SIM codes, and no word-level timing for a latency measure)
# are appended after, rather than interleaved, so they don't shift any
# shared measure's position. Identical list duplicated in
# semantic_fluency.py (see that file for the SFT side) -- same reasoning
# this project already uses for not sharing other per-module helpers.
_ADDITIONAL_MEASURE_ORDER = [
    "error_rate", "OTH_rate", "N_rate", "R_rate", "F_rate", "mean_aoa_correct",
    "PN_rate", "SIM_rate", "time_to_first_correct",
]

# Every participant's age-adjusted z-score on each of the measures above
# (same transform the cards apply to the selected participant, applied here
# to the whole cohort so "Compared with other participants" can plot
# everyone in those same age-adjusted units -- see cookie_theft.py's
# CT_HEADLINE_Z for the same pattern). z is oriented per band()'s
# convention: negative always means the impaired direction.
_pft_ages = pft_metrics_df["participant_id"].astype(str).map(
    lambda pid: DEMOGRAPHICS.get(pid, {}).get("age")
)
PFT_HEADLINE_Z = pd.DataFrame({
    "participant_id": pft_metrics_df["participant_id"],
    "diagnosis": pft_metrics_df["diagnosis"],
})
for _m in _ALL_STATS_MEASURES:
    _intercept, _slope, _residual_sd = HC_STATS.get(_m.key, (None, None, None))
    if _intercept is None or not _residual_sd:
        PFT_HEADLINE_Z[_m.key] = None
        continue
    _predicted = _intercept + _slope * _pft_ages
    _z = (pft_metrics_df[_m.key] - _predicted) / _residual_sd
    PFT_HEADLINE_Z[_m.key] = -_z if _m.higher_is_worse else _z

_GROUPS = ["HC", "MCI", "Dementia"]


def _card_distribution_figure(metric_key, participant_z):
    """Single-measure violin plot sized to sit inside a card -- same visual
    language as cookie_theft.py's own _card_distribution_figure (violin per
    group in age-adjusted z-score units, HC "typical" band shaded at z in
    [-1, 1], this participant's own z picked out with a dashed line): each
    measure card embeds this directly rather than every measure living in
    one tall stacked panel (user request, to match Cookie Theft's layout).
    """
    fig = go.Figure()

    fig.add_vrect(x0=-1, x1=1, fillcolor=GROUP_COLOURS["HC"], opacity=0.12, line_width=0)

    for group in _GROUPS:
        vals = PFT_HEADLINE_Z.loc[PFT_HEADLINE_Z["diagnosis"] == group, metric_key].dropna().values
        if len(vals) < 2 or vals.std() == 0:
            continue
        fig.add_trace(go.Violin(
            x=vals, y=[group] * len(vals), orientation="h", name=group,
            line_color=GROUP_COLOURS[group], fillcolor=GROUP_COLOURS[group],
            opacity=0.35, meanline_visible=True,
            meanline=dict(color="#111", width=2),
            points=False, spanmode="hard", showlegend=False,
        ))
    fig.update_yaxes(categoryorder="array", categoryarray=["Dementia", "MCI", "HC"])

    if participant_z is not None and pd.notna(participant_z):
        fig.add_vline(x=participant_z, line=dict(color="#1565C0", width=2, dash="dash"))

    fig.update_layout(height=150, margin=dict(t=6, b=24, l=6, r=6), plot_bgcolor="white")
    fig.update_xaxes(gridcolor="#eee", tickfont=dict(size=10),
                     title=dict(text="age-adjusted z-score", font=dict(size=10)))
    fig.update_yaxes(gridcolor="#eee", tickfont=dict(size=10))
    return fig


def _ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 3 -> '3rd', 4 -> '4th', 11-13 -> 'th' even
    though they end in 1/2/3 (the standard English exception)."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _classification_legend(keys):
    """Small colour-coded legend for the transcript below -- otherwise the
    only way to learn what a colour means is to hover every single word."""
    chips = [
        ui.span(
            CLASSIFICATION_LABELS.get(key, key),
            style=(
                f"display:inline-block;background-color:{CLASSIFICATION_COLOURS.get(key, '#333')}22;"
                f"color:{CLASSIFICATION_COLOURS.get(key, '#333')};"
                f"border:1.5px solid {CLASSIFICATION_COLOURS.get(key, '#333')};"
                "border-radius:4px;padding:2px 8px;margin:2px 6px 2px 0;"
                "font-size:11px;font-weight:500;"
            ),
        )
        for key in keys
    ]
    return ui.div(*chips, style="margin-bottom:12px;")


def _percentile_text(percentile: float | None) -> str:
    """Plain-language percentile line for a card -- same wording as
    cookie_theft.py's, so the two tabs read the same way. Low percentile
    always means the impaired direction, regardless of the raw measure's
    own higher/lower-is-worse convention (see percentile_age_adjusted())."""
    if percentile is None:
        return "Percentile unavailable"
    return f"{_ordinal(round(percentile))} percentile for age"


def _measure_card(measure, value, age, intercept, slope, residual_sd):
    """Same bare card recipe as cookie_theft.py's: label, value, percentile,
    description, HC reference range -- no Typical/Borderline/Atypical badge.
    PFT/SFT used to show one (see git history); dropped for the same reason
    Cookie Theft's own band display was dropped (see
    dementia_speech.interpretation's module docstring) -- it's a second,
    redundant encoding of the same percentile already on the card, and
    whether a given measure is banded at all is already explained in its own
    description text. The reference is age-adjusted: predicted from this
    participant's own age via a line fit on the HC (TRAIN-split) group, not
    one flat HC mean/SD shared by every age.
    """
    has_value = value is not None and not (isinstance(value, float) and np.isnan(value))
    has_reference = (
        age is not None and not (isinstance(age, float) and np.isnan(age))
        and intercept is not None and slope is not None
        and residual_sd is not None and residual_sd > 0
    )
    predicted_mean = (intercept + slope * age) if has_reference else None

    value_str = measure.fmt.format(value) if has_value else "—"

    percentile = (
        percentile_age_adjusted(value, age, intercept, slope, residual_sd, measure.higher_is_worse)
        if has_value and has_reference
        else None
    )

    range_str = (
        f"HC reference (age {int(age)}): "
        f"{measure.fmt.format(max(0, predicted_mean - residual_sd))} – "
        f"{measure.fmt.format(predicted_mean + residual_sd)}"
        if has_reference
        else "HC reference unavailable"
    )

    return ui.div(
        ui.div(measure.label, style="font-size:14px;font-weight:600;color:#495057;"),
        ui.div(value_str, style="font-size:22px;font-weight:700;color:#212529;margin:6px 0;"),
        ui.div(_percentile_text(percentile), style="font-size:12px;font-weight:600;color:#34495e;"),
        ui.div(measure.description, style="color:#6c757d;font-size:12px;margin-top:4px;"),
        ui.div(output_widget(f"headline_graph_{measure.key}"), style="margin-top:8px;"),
        ui.div(range_str, style="color:#9aa5b1;font-size:11px;margin-top:6px;"),
        style=(
            "background:white;border:1px solid #e1e8ef;border-top:4px solid #527d96;"
            "border-radius:10px;padding:16px 18px;min-width:16em;flex:1;"
            "box-shadow:0 2px 8px rgba(35,55,75,0.05);"
        ),
    )


_PANEL_STYLE = (
    "background:white;border:1px solid #e1e8ef;border-top:4px solid #527d96;"
    "border-radius:12px;padding:22px;min-width:0;"
    "box-shadow:0 2px 8px rgba(35,55,75,0.05);"
)
_SECTION_HEADER_STYLE = "font-weight:700;color:#243447;margin-bottom:4px;"
_SECTION_CAPTION_STYLE = "color:#6c757d;font-size:13px;margin-bottom:16px;"
# Fixed height + internal scroll on the transcript word-list, same technique
# cookie_theft.py uses for its side-by-side panels -- keeps this section's
# height (and so where "Words per 15-second bin"/"Additional language
# analysis" start) consistent regardless of how long a given participant's
# transcript is, so the two fluency tasks' columns stay horizontally
# aligned block-for-block (user request, 2026-11-17).
_TRANSCRIPT_SCROLL_HEIGHT = "360px"
# Fixed height on the bin-plot block (user request, 2026-11-17: bin plot
# directly under Transcript, then Additional measures). SFT has no
# word-level timing and so no equivalent block -- semantic_fluency.py gives
# its own column a placeholder of this same height in the same position so
# "Additional language analysis" still starts at the same height in both
# columns; see that file's comment.
_BIN_PLOT_BLOCK_HEIGHT = "460px"


@module.ui
def phonemic_fluency_cards_ui():
    """Just the clinical-summary cards -- no section heading of its own,
    since app.py now shows ONE shared "Clinical summary" heading above both
    tasks' cards rather than one per task. Must be mounted with the same
    module id as phonemic_fluency_server() (e.g.
    `phonemic_fluency_cards_ui("pft")` alongside
    `phonemic_fluency_server("pft", ...)`) so its outputs resolve -- see
    module.ui's namespacing.
    """
    return ui.output_ui("headline_panel")


@module.ui
def phonemic_fluency_detail_ui():
    """Everything except the clinical-summary (key-measure) cards --
    transcript, then one flat "Additional measures" section for everything
    that isn't a validated key measure, matching Cookie Theft's layout
    (user request, 2026-11-17): one accordion holding context measures,
    the error/code rates that didn't clear the validation bar, and the
    exploratory extras together, rather than three separate sections. Must
    be mounted with the same module id as phonemic_fluency_server() (see
    phonemic_fluency_cards_ui's docstring)."""
    return ui.div(
        ui.div(
            ui.h5("Transcript", style=_SECTION_HEADER_STYLE),
            ui.div(
                ui.p(
                    "Every word produced, colour-coded by classification "
                    "below. Hover a word for its exact type and timestamp.",
                    style=_SECTION_CAPTION_STYLE,
                ),
                _classification_legend(["C", "OTH", "N", "PN", "R", "SIM", "F"]),
                ui.div(
                    ui.output_ui("word_display"),
                    style=f"max-height:{_TRANSCRIPT_SCROLL_HEIGHT};overflow-y:auto;",
                ),
                style=_PANEL_STYLE,
            ),
            style="margin-bottom:26px;",
        ),

        ui.div(
            ui.h6("Words per 15-second bin", style="font-weight:600;color:#34495e;"),
            ui.p(
                "Correct words produced in each 15-second quarter of the "
                "minute, against each diagnostic group's own average "
                "pattern — shows whether output holds steady or drops "
                "off as the trial goes on.",
                style="color:#6c757d;font-size:12px;margin:2px 0 8px;",
            ),
            output_widget("bin_plot"),
            style=_PANEL_STYLE + f"min-height:{_BIN_PLOT_BLOCK_HEIGHT};margin-bottom:26px;",
        ),

        ui.div(
            ui.h5("Additional language analysis", style=_SECTION_HEADER_STYLE),
            ui.div(
                ui.accordion(
                    ui.accordion_panel(
                        "Additional measures",
                        ui.p(
                            "Supporting context for qualitative review, not an "
                            "independent diagnosis on their own -- only \"Words "
                            "produced correctly\" and \"Total words spoken\" "
                            "(above) are shown as this task's key measures. "
                            "\"Off-target speech rate\" below is also "
                            "validated (it separates groups independently of "
                            "correct-word count) but is grouped here with the "
                            "rest rather than promoted to the top. "
                            "(Clustering/switching are computed and saved in "
                            "the metrics file but not shown here -- see "
                            "docs/pft_new_scoring.md.)",
                            style="color:#6c757d;font-size:12px;",
                        ),
                        ui.output_ui("additional_measures_panel"),
                    ),
                    open=False,
                ),
                style=_PANEL_STYLE,
            ),
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

    def _participant_age():
        return DEMOGRAPHICS.get(str(participant_id()), {}).get("age")

    # ── Clinical summary cards: HEADLINE_MEASURES + ERROR_HEADLINE_MEASURES
    # together -- these are this task's only two validated, independently-
    # diagnostic measures, shown with equal prominence (matches Cookie
    # Theft, which doesn't split "main" vs. "error" headline measures
    # either). Everything else lives in additional_measures_panel below.
    @render.ui
    def headline_panel():
        m = pmetrics()
        if not m:
            return ui.p("No data for this participant.")
        age = _participant_age()
        cards = [
            _measure_card(meas, m.get(meas.key), age, *HC_STATS.get(meas.key, (None, None, None)))
            for meas in HEADLINE_MEASURES + CONTEXT_MEASURES
        ]
        return ui.div(*cards, style="display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;")

    def _participant_z(metric_key):
        """This participant's age-adjusted z-score on one measure (see
        PFT_HEADLINE_Z) -- shared by each card's embedded graph below."""
        pid = participant_id()
        z_row = PFT_HEADLINE_Z[PFT_HEADLINE_Z["participant_id"].astype(str) == str(pid)]
        return z_row.iloc[0][metric_key] if len(z_row) else None

    # ── Per-card group-comparison graphs (one per measure shown as a card) ──
    @render_widget
    def headline_graph_C():
        return _card_distribution_figure("C", _participant_z("C"))

    @render_widget
    def headline_graph_total_words():
        return _card_distribution_figure("total_words", _participant_z("total_words"))

    @render_widget
    def headline_graph_OTH_rate():
        return _card_distribution_figure("OTH_rate", _participant_z("OTH_rate"))

    # ── Additional measures: context + error-exploratory + exploratory,
    # all together in one panel (matches Cookie Theft's single "Additional
    # measures" accordion) ──
    @render.ui
    def additional_measures_panel():
        m = pmetrics()
        if not m:
            return ui.p("No data for this participant.")
        age = _participant_age()
        cards = [
            _measure_card(
                _ADDITIONAL_MEASURES_BY_KEY[key], m.get(key), age,
                *HC_STATS.get(key, (None, None, None)),
            )
            for key in _ADDITIONAL_MEASURE_ORDER
            if key in _ADDITIONAL_MEASURES_BY_KEY
        ]
        return ui.div(*cards, style="display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;")

    @render_widget
    def headline_graph_error_rate():
        return _card_distribution_figure("error_rate", _participant_z("error_rate"))

    @render_widget
    def headline_graph_N_rate():
        return _card_distribution_figure("N_rate", _participant_z("N_rate"))

    @render_widget
    def headline_graph_PN_rate():
        return _card_distribution_figure("PN_rate", _participant_z("PN_rate"))

    @render_widget
    def headline_graph_R_rate():
        return _card_distribution_figure("R_rate", _participant_z("R_rate"))

    @render_widget
    def headline_graph_SIM_rate():
        return _card_distribution_figure("SIM_rate", _participant_z("SIM_rate"))

    @render_widget
    def headline_graph_F_rate():
        return _card_distribution_figure("F_rate", _participant_z("F_rate"))

    @render_widget
    def headline_graph_time_to_first_correct():
        return _card_distribution_figure("time_to_first_correct", _participant_z("time_to_first_correct"))

    @render_widget
    def headline_graph_mean_aoa_correct():
        return _card_distribution_figure("mean_aoa_correct", _participant_z("mean_aoa_correct"))

    @render.ui
    def word_display():
        df = pdata()
        if len(df) == 0:
            return ui.p("No words found for this participant.")
        spans = []
        for _, row in df.iterrows():
            cls = row["classification"]
            colour = CLASSIFICATION_COLOURS.get(cls, "#333")
            label = CLASSIFICATION_LABELS.get(cls, cls)
            spans.append(ui.span(
                row["word"],
                title=f"{label} ({row['start']:.2f}s)",
                style=(f"display:inline-block;background-color:{colour}22;color:{colour};"
                       f"border:1.5px solid {colour};border-radius:4px;padding:3px 8px;"
                       f"margin:3px 4px;font-weight:500;"),
            ))
        return ui.div(*spans, style="line-height:2.8;padding:12px;")

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
            line=dict(color="#1565C0", width=3), marker=dict(size=9),
        ))
        fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10),
                          plot_bgcolor="white",
                          xaxis=dict(title="Time bin", gridcolor="#eee"),
                          yaxis=dict(title="Correct words", gridcolor="#eee", rangemode="tozero"),
                          legend=dict(orientation="h", y=-0.2))
        return fig
