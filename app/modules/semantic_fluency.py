"""Semantic/category fluency (SFT) tab module -- mirrors phonemic_fluency.py's
structure, adapted for what SFT actually has: no word-level timing (manually
transcribed free text, not an aligned TextGrid -- see SFT_new/scoring.py), so
no temporal-bin plot and no timestamp in the transcript tooltip; no semantic
clustering yet (deferred), so no cluster underlining or clustering panel.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import module, ui, render, reactive
from shinywidgets import output_widget, render_widget

from shared import (
    sft_words_df, sft_metrics_df, SPLIT, DEMOGRAPHICS,
    CLASSIFICATION_COLOURS, CLASSIFICATION_LABELS, GROUP_COLOURS,
    FLUENCY_DISCREPANCY_BY_PARTICIPANT, FLUENCY_DISCREPANCY_DF,
)
from dementia_speech.tasks.SFT_new.interpretation import (
    HEADLINE_MEASURES, CONTEXT_MEASURES, ERROR_HEADLINE_MEASURES,
    ERROR_EXPLORATORY_MEASURES, EXPLORATORY_MEASURES,
)
from dementia_speech.interpretation import fit_age_adjusted_norms, percentile_age_adjusted

# Age-adjusted reference stats, computed against the TRAIN-split HC group
# only (see shared.SPLIT). See SFT_new/interpretation.py for why `C` and
# `R_rate` are the two measures that actually separate groups here, not
# `OTH_rate` (PFT's one). CONTEXT_MEASURES and EXPLORATORY_MEASURES are
# included here too so their cards can show a percentile alongside the raw
# value, same card recipe as HEADLINE_MEASURES -- see phonemic_fluency.py's
# _measure_card docstring for why there's no separate badge. See
# fit_age_adjusted_norms() for why the reference is a value predicted from
# the participant's own age rather than one flat HC mean/SD.
_sft_split = sft_metrics_df["participant_id"].astype(str).map(SPLIT)
_HC = sft_metrics_df[(sft_metrics_df["diagnosis"] == "HC") & (_sft_split == "TRAIN")]
_HC_AGES = _HC["participant_id"].astype(str).map(lambda pid: DEMOGRAPHICS.get(pid, {}).get("age"))
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
# Same canonical order as phonemic_fluency.py's (see that file's comment) --
# measures both tasks compute come first, in identical order, so they land
# at the same position in each task's panel and line up horizontally side
# by side. SFT has no PN/SIM codes and no word-level timing, so those three
# PFT-only keys are simply absent here (filtered out below) rather than
# needing their own entry.
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
_sft_ages = sft_metrics_df["participant_id"].astype(str).map(
    lambda pid: DEMOGRAPHICS.get(pid, {}).get("age")
)
SFT_HEADLINE_Z = pd.DataFrame({
    "participant_id": sft_metrics_df["participant_id"],
    "diagnosis": sft_metrics_df["diagnosis"],
})
for _m in _ALL_STATS_MEASURES:
    _intercept, _slope, _residual_sd = HC_STATS.get(_m.key, (None, None, None))
    if _intercept is None or not _residual_sd:
        SFT_HEADLINE_Z[_m.key] = None
        continue
    _predicted = _intercept + _slope * _sft_ages
    _z = (sft_metrics_df[_m.key] - _predicted) / _residual_sd
    SFT_HEADLINE_Z[_m.key] = -_z if _m.higher_is_worse else _z

_GROUPS = ["HC", "MCI", "Dementia"]
_MEASURES_BY_KEY = {m.key: m for m in _ALL_STATS_MEASURES}
_DISCREPANCY_FIGURE_HEIGHT = 136


def _discrepancy_distribution_figure(participant_value):
    """Per-group violin of discrepancy_score (z_letter - z_category) plus
    this participant's own value as a dashed line -- lives on the SFT side
    (user request, 2026-11-15: this is a letter-vs-category COMPARISON, as
    relevant to this task as to PFT, and fills what would otherwise be dead
    space next to the "no bin plot" placeholder) -- same visual language as
    every other measure card's embedded graph (see _card_distribution_figure
    above), but with no shaded "typical" band: that convention bands
    against an age-adjusted norm fit for the one measure being shown, which
    hasn't been done for this derived, explicitly experimental score."""
    fig = go.Figure()
    for group in _GROUPS:
        vals = FLUENCY_DISCREPANCY_DF.loc[
            FLUENCY_DISCREPANCY_DF["diagnosis"] == group, "discrepancy_score"
        ].dropna().values
        if len(vals) < 2 or vals.std() == 0:
            continue
        fig.add_trace(go.Violin(
            x=vals, y=[group] * len(vals), orientation="h", name=group,
            line_color=GROUP_COLOURS[group], fillcolor=GROUP_COLOURS[group],
            opacity=0.35, meanline_visible=True,
            meanline=dict(color="#111", width=2),
            points=False, spanmode="hard", showlegend=False,
            hoverinfo="skip",
        ))
    fig.update_yaxes(categoryorder="array", categoryarray=["Dementia", "MCI", "HC"])

    if participant_value is not None and participant_value == participant_value:
        fig.add_vline(x=participant_value, line=dict(color="#0D47A1", width=3, dash="dash"), annotation_text="Participant", annotation_position="top", annotation_font=dict(size=10, color="#0D47A1"))

    fig.update_layout(height=_DISCREPANCY_FIGURE_HEIGHT, margin=dict(t=22, b=24, l=6, r=6), plot_bgcolor="white")
    fig.update_xaxes(gridcolor="#eee", tickfont=dict(size=10),
                     title=dict(text="discrepancy score (z_letter − z_category)", font=dict(size=10)))
    fig.update_yaxes(gridcolor="#eee", tickfont=dict(size=10))
    return fig


def _card_distribution_figure(metric_key, participant_value, participant_age):
    """Single-measure violin plot sized to sit inside a card, in the
    measure's RAW units (same number as the card's headline value, so the
    participant's dashed line lines up with it): each group's raw
    distribution (all participants in that group -- NOT age-adjusted), a
    shaded band showing this participant's AGE-MATCHED HC reference range
    (HC-predicted mean for their age +/- 1 residual SD -- the same range the
    card prints as "HC reference"), and the participant's own raw value as
    a labelled dashed line. The age-adjusted z-score/percentile is reported
    in the card text itself (see _percentile_text), not on this plot --
    purely descriptive, no Typical/Atypical judgement.
    """
    measure = _MEASURES_BY_KEY[metric_key]
    fig = go.Figure()

    all_vals = {
        g: sft_metrics_df.loc[sft_metrics_df["diagnosis"] == g, metric_key].dropna().values
        for g in _GROUPS
    }

    intercept, slope, residual_sd = HC_STATS.get(metric_key, (None, None, None))
    has_band = (
        participant_age is not None and pd.notna(participant_age)
        and intercept is not None and slope is not None
        and residual_sd is not None and pd.notna(residual_sd) and residual_sd > 0
    )
    if has_band:
        predicted = intercept + slope * participant_age
        lo, hi = predicted - residual_sd, predicted + residual_sd
        # Same lower clip the card's printed range uses for non-negative measures.
        if all(len(v) == 0 or v.min() >= 0 for v in all_vals.values()):
            lo = max(0, lo)
        fig.add_vrect(x0=lo, x1=hi, fillcolor=GROUP_COLOURS["HC"], opacity=0.12, line_width=0)

    for group in _GROUPS:
        vals = all_vals[group]
        if len(vals) < 2 or vals.std() == 0:
            continue
        fig.add_trace(go.Violin(
            x=vals, y=[group] * len(vals), orientation="h", name=group,
            line_color=GROUP_COLOURS[group], fillcolor=GROUP_COLOURS[group],
            opacity=0.35, meanline_visible=True,
            meanline=dict(color="#111", width=2),
            points=False, spanmode="hard", showlegend=False,
            hoverinfo="skip",
        ))
    fig.update_yaxes(categoryorder="array", categoryarray=["Dementia", "MCI", "HC"])

    if participant_value is not None and pd.notna(participant_value):
        fig.add_vline(x=participant_value, line=dict(color="#0D47A1", width=3, dash="dash"), annotation_text="Participant", annotation_position="top", annotation_font=dict(size=10, color="#0D47A1"))

    fig.update_layout(height=166, margin=dict(t=22, b=24, l=6, r=6), plot_bgcolor="white")
    fig.update_xaxes(gridcolor="#eee", tickfont=dict(size=10),
                     title=dict(text=measure.label, font=dict(size=10)))
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
    # The age-adjusted z-score (oriented: negative = impaired direction) is
    # recovered from the percentile, which is just 100 * norm.cdf(z) -- so it
    # needs no extra plumbing at any call site.
    from scipy.stats import norm
    z = float(norm.ppf(min(max(percentile, 0.01), 99.99) / 100))
    z_str = f"{z:+.1f}".replace("-", "\u2212")
    return f"{_ordinal(round(percentile))} percentile for age \u00b7 z = {z_str}"


def _measure_card(measure, value, age, intercept, slope, residual_sd):
    """Same bare card recipe as cookie_theft.py's and phonemic_fluency.py's:
    label, value, percentile, description, HC reference range -- no
    Typical/Borderline/Atypical badge (see phonemic_fluency.py's
    _measure_card docstring for why it was dropped). The reference is
    age-adjusted: predicted from this participant's own age via a line fit
    on the HC (TRAIN-split) group, not one flat HC mean/SD shared by every
    age.
    """
    has_value = value is not None and not (isinstance(value, float) and value != value)
    has_reference = (
        age is not None and not (isinstance(age, float) and age != age)
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
# Same fixed height + internal scroll as phonemic_fluency.py's transcript
# panel (see that file's comment) -- keeps both tasks' columns horizontally
# aligned block-for-block regardless of per-participant transcript length.
_TRANSCRIPT_SCROLL_HEIGHT = "360px"
# Same height as phonemic_fluency.py's bin-plot block, so this column's own
# row (placeholder + discrepancy score, see below) keeps "Additional
# language analysis" starting at the same height in both columns (user
# request, 2026-11-17; row shortened and discrepancy score moved into this
# row 2026-11-15).
_BIN_PLOT_BLOCK_HEIGHT = "300px"


@module.ui
def semantic_fluency_cards_ui():
    """Just the clinical-summary cards -- no section heading of its own,
    since app.py now shows ONE shared "Clinical summary" heading above both
    tasks' cards rather than one per task. Must be mounted with the same
    module id as semantic_fluency_server() (see
    phonemic_fluency_cards_ui's docstring for the same pattern)."""
    return ui.output_ui("headline_panel")


@module.ui
def semantic_fluency_detail_ui():
    """Everything except the clinical-summary (key-measure) cards --
    transcript, then one flat "Additional measures" section for everything
    that isn't a validated key measure, matching Cookie Theft's layout
    (user request, 2026-11-17): one accordion holding context measures,
    the error/code rates that didn't clear the validation bar, and the
    exploratory extras together, rather than separate sections. Must be
    mounted with the same module id as semantic_fluency_server() (see
    semantic_fluency_cards_ui's docstring)."""
    return ui.div(
        ui.div(
            ui.h5("Transcript", style=_SECTION_HEADER_STYLE),
            # Caption + legend live INSIDE the panel, same as
            # phonemic_fluency.py, so both tasks' transcript panels have
            # identical structure and line up.
            ui.div(
                ui.p(
                    "Every item produced, colour-coded by classification below. "
                    "Hover an item for its exact type.",
                    style=_SECTION_CAPTION_STYLE,
                ),
                _classification_legend(["C", "OTH", "N", "R", "F"]),
                ui.output_ui("assistance_note"),
                ui.div(
                    ui.output_ui("word_display"),
                    style=f"height:{_TRANSCRIPT_SCROLL_HEIGHT};overflow-y:auto;",
                ),
                style=_PANEL_STYLE,
            ),
            style="margin-bottom:26px;",
        ),

        ui.div(
            ui.output_ui("discrepancy_card"),
            style="margin-bottom:26px;",
        ),

        ui.div(
            ui.h5("Additional language analysis", style=_SECTION_HEADER_STYLE),
            ui.div(
                ui.accordion(
                    ui.accordion_panel(
                        "Additional measures",
                        ui.p(
                            "Supporting context for qualitative review, not an "
                            "independent diagnosis on their own -- only "
                            "\"Animals produced correctly\" and \"Total items "
                            "spoken\" (above) are shown as this task's key "
                            "measures. \"Repetition rate\" below is also "
                            "validated (it separates groups independently of "
                            "correct-item count) but is grouped here with the "
                            "rest rather than promoted to the top. (Semantic "
                            "clustering/switching, Zemla et al. 2020, is "
                            "computed and saved in the metrics file but not "
                            "shown here -- see docs/sft_scoring.md.)",
                            style="color:#6c757d;font-size:12px;",
                        ),
                        ui.output_ui("additional_measures_panel"),
                    ),
                    open=False,
                ),
                style=_PANEL_STYLE,
            ),
        ),
        # display:contents so the three sections above are direct rows of
        # fluency_tab.py's shared grid (keeps PFT/SFT blocks aligned).
        style="display:contents;",
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

    def _participant_age():
        return DEMOGRAPHICS.get(str(participant_id()), {}).get("age")

    # ── Clinical summary cards: HEADLINE_MEASURES + ERROR_HEADLINE_MEASURES
    # together -- this task's only two validated, independently-diagnostic
    # measures, shown with equal prominence (matches Cookie Theft, which
    # doesn't split "main" vs. "error" headline measures either). Everything
    # else lives in additional_measures_panel below.
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

    def _participant_value(metric_key):
        """This participant's raw value on one measure -- shared by each
        card's embedded graph below."""
        return pmetrics().get(metric_key)

    # ── Per-card group-comparison graphs (one per measure shown as a card) ──
    @render_widget
    def headline_graph_C():
        return _card_distribution_figure("C", _participant_value("C"), _participant_age())

    @render_widget
    def headline_graph_total_items():
        return _card_distribution_figure("total_items", _participant_value("total_items"), _participant_age())

    @render_widget
    def headline_graph_R_rate():
        return _card_distribution_figure("R_rate", _participant_value("R_rate"), _participant_age())

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
        return _card_distribution_figure("error_rate", _participant_value("error_rate"), _participant_age())

    @render_widget
    def headline_graph_OTH_rate():
        return _card_distribution_figure("OTH_rate", _participant_value("OTH_rate"), _participant_age())

    @render_widget
    def headline_graph_N_rate():
        return _card_distribution_figure("N_rate", _participant_value("N_rate"), _participant_age())

    @render_widget
    def headline_graph_F_rate():
        return _card_distribution_figure("F_rate", _participant_value("F_rate"), _participant_age())

    @render_widget
    def headline_graph_mean_aoa_correct():
        return _card_distribution_figure("mean_aoa_correct", _participant_value("mean_aoa_correct"), _participant_age())

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

    # ── Letter-vs-category discrepancy score (Wright et al. 2022) -- lives
    # on this side (user request, 2026-11-15), next to the "no bin plot"
    # placeholder. Experimental: single trial per task, and doesn't show
    # the group separation the paper reports on this dataset -- see
    # pipelines/score_fluency_discrepancy.py.
    @render.ui
    def discrepancy_card():
        row = FLUENCY_DISCREPANCY_BY_PARTICIPANT.get(str(participant_id()))
        if not row:
            return ui.div(style=f"height:{_BIN_PLOT_BLOCK_HEIGHT};")
        value = row.get("discrepancy_score")
        value_str = f"{value:+.2f}" if value is not None and value == value else "—"
        return ui.div(
            # Title and "Experimental" badge share one row (saves a line in
            # this fixed-height card).
            ui.div(
                ui.span("Fluency discrepancy score", style="font-weight:600;font-size:1.05em;"),
                ui.span("Experimental", style=(
                    "background:#9e9e9e;color:white;border-radius:12px;"
                    "padding:2px 10px;font-size:0.8em;font-weight:600;"
                    "white-space:nowrap;"
                )),
                style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;",
            ),
            ui.div(value_str, style="font-size:1.6em;font-weight:700;margin:2px 0;"),
            ui.div(
                "Compares how far letter fluency and category fluency each "
                "fall from the healthy-control average (letter z-score minus "
                "category z-score). Positive values mean category fluency is "
                "lower relative to letter fluency.",
                style="color:#6c757d;font-size:12px;margin-bottom:6px;text-wrap:pretty;",
            ),
            output_widget("discrepancy_graph"),
            style=(
                "background:white;border:1px solid #e1e8ef;border-left:4px solid #9e9e9e;"
                "border-radius:10px;padding:16px 18px;"
                f"height:{_BIN_PLOT_BLOCK_HEIGHT};overflow:hidden;"
                "box-shadow:0 2px 8px rgba(35,55,75,0.05);"
            ),
        )

    @render_widget
    def discrepancy_graph():
        row = FLUENCY_DISCREPANCY_BY_PARTICIPANT.get(str(participant_id()))
        value = row.get("discrepancy_score") if row else None
        return _discrepancy_distribution_figure(value)

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
