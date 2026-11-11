"""Cookie Theft picture description tab module."""
import pandas as pd
from shiny import module, ui, render, reactive
import plotly.graph_objects as go
from shinywidgets import output_widget, render_widget

from shared import (
    ct_iu_df, ct_metrics_df, CTD_TRANSCRIPT_PATHS, GROUP_COLOURS, parse_sequence,
    SPLIT, DEMOGRAPHICS,
)
from dementia_speech.tasks.picture_description.graph import (
    render_graph_plotly, GRAPH_FIGURE_HEIGHT,
)
from dementia_speech.tasks.picture_description.scoring import SYNONYMS
from dementia_speech.tasks._shared.manual_transcript import extract_patient_speech
from dementia_speech.tasks.picture_description.interpretation import (
    HEADLINE_MEASURES, EXPLORATORY_MEASURES,
)
from dementia_speech.interpretation import fit_age_adjusted_norms, percentile_age_adjusted

# Age-adjusted HC reference stats, computed once at import time against the
# TRAIN-split HC group only (same reasoning as PFT/SFT's train-only
# HC_STATS: no participant is scored against a distribution that includes
# them or a held-out test case). Covers HEADLINE_MEASURES and
# EXPLORATORY_MEASURES alike -- both show a value, an age-adjusted
# percentile (see percentile_age_adjusted(), no Typical/Borderline/Atypical
# judgement either way -- see interpretation.py for why), and a
# group-comparison graph. See fit_age_adjusted_norms() for why the
# reference is a value predicted from the participant's own age rather than
# one flat HC mean/SD.
_ALL_CT_MEASURES = HEADLINE_MEASURES + EXPLORATORY_MEASURES
_hc_split = ct_metrics_df["participant_id"].astype(str).map(SPLIT)
_HC = ct_metrics_df[(ct_metrics_df["diagnosis"] == "HC") & (_hc_split == "TRAIN")]
_HC_AGES = _HC["participant_id"].astype(str).map(lambda pid: DEMOGRAPHICS.get(pid, {}).get("age"))
HC_STATS = {
    m.key: fit_age_adjusted_norms(_HC[m.key], _HC_AGES)
    for m in _ALL_CT_MEASURES if m.key in _HC
}

# Every participant's age-adjusted z-score on each of the measures above
# (same transform the cards apply to the selected participant, applied here
# to the whole cohort so each card's group-comparison graph can show the
# group distributions in those same age-adjusted units). z is oriented per
# band()'s convention: negative always means the impaired direction,
# regardless of measure.
_ct_ages = ct_metrics_df["participant_id"].astype(str).map(
    lambda pid: DEMOGRAPHICS.get(pid, {}).get("age")
)
CT_HEADLINE_Z = pd.DataFrame({
    "participant_id": ct_metrics_df["participant_id"],
    "diagnosis": ct_metrics_df["diagnosis"],
})
for _m in _ALL_CT_MEASURES:
    _intercept, _slope, _residual_sd = HC_STATS.get(_m.key, (None, None, None))
    if _intercept is None or not _residual_sd:
        CT_HEADLINE_Z[_m.key] = None
        continue
    _predicted = _intercept + _slope * _ct_ages
    _z = (ct_metrics_df[_m.key] - _predicted) / _residual_sd
    CT_HEADLINE_Z[_m.key] = -_z if _m.higher_is_worse else _z

_GROUPS = ["HC", "MCI", "Dementia"]

# Transcript and IU checklist scroll within this height so they line up with
# the picture (GRAPH_FIGURE_HEIGHT) when shown side by side -- see
# picture_description/graph.py for where that height comes from.
_PANEL_SCROLL_HEIGHT = f"{GRAPH_FIGURE_HEIGHT}px"


def _progress_bar(fraction: float, colour: str = "#527d96"):
    """A filled horizontal bar, 0-1 fraction -- only used for iu_score's
    "x/20 content units" card, where a fraction-of-a-whole reading is
    meaningful (unlike e.g. a rate per 100 words). Colour defaults to the
    tab's own accent colour (no band/percentile-derived colour -- see
    headline_panel()'s own comment on why there's no colour coding yet).
    """
    pct = max(0.0, min(1.0, fraction)) * 100
    return ui.div(
        ui.div(style=(
            f"width:{pct:.0f}%;height:100%;background:{colour};"
            "border-radius:5px;transition:width 0.3s;"
        )),
        style="background:#eee;border-radius:5px;height:8px;width:100%;margin-top:8px;overflow:hidden;",
    )


def _ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 3 -> '3rd', 4 -> '4th', 11-13 -> 'th' even
    though they end in 1/2/3 (the standard English exception)."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _percentile_text(percentile: float | None) -> str:
    """Plain-language percentile line for a card -- no colour, no Typical/
    Borderline/Atypical judgement, just where this value falls in the
    age-adjusted HC reference distribution (see percentile_age_adjusted()).
    Low percentile always means the impaired direction, regardless of the
    raw measure's own higher/lower-is-worse convention.
    """
    if percentile is None:
        return "Percentile unavailable"
    return f"{_ordinal(round(percentile))} percentile for age"


def _exploratory_card(measure, value, age, intercept, slope, residual_sd):
    """A card for one EXPLORATORY_MEASURES entry: the value, its age-adjusted
    percentile (no Typical/Borderline/Atypical judgement -- see
    percentile_age_adjusted() and interpretation.py), a group-comparison
    graph (same mechanics as a headline card -- see
    _card_distribution_figure), and a plain-language description of what
    it's actually shown to do in this cohort.
    """
    has_value = value is not None and pd.notna(value)
    value_str = measure.fmt.format(value) if has_value else "—"

    has_reference = (
        age is not None and pd.notna(age)
        and intercept is not None and slope is not None
        and residual_sd is not None and pd.notna(residual_sd)
        and residual_sd > 0
    )
    percentile = None
    if has_value and has_reference:
        percentile = percentile_age_adjusted(
            value, age, intercept, slope, residual_sd, measure.higher_is_worse,
        )

    return ui.div(
        ui.div(measure.label, style="font-size:13px;font-weight:600;color:#495057;"),
        ui.div(value_str, style="font-size:22px;font-weight:700;color:#212529;margin:6px 0;"),
        ui.div(_percentile_text(percentile), style="font-size:12px;font-weight:600;color:#34495e;"),
        ui.div(
            output_widget(f"exploratory_graph_{measure.key}"),
            style="margin-top:8px;"
        ),
        ui.div(measure.description, style="font-size:12px;color:#6c757d;margin-top:8px;"),
        style=(
            "background:#fbfcfd;border:1px solid #e5eaf0;border-radius:10px;"
            "padding:14px 16px;min-width:0;"
        ),
    )


def _card_distribution_figure(metric_key, participant_z):
    """Single-measure violin plot sized to sit inside a card (headline or
    exploratory -- any metric_key present in CT_HEADLINE_Z works): each
    group's age-adjusted z-score distribution shape (not a box summary),
    a shaded reference band showing roughly where the HC group itself
    clusters (z in [-1, 1] -- CT_HEADLINE_Z's z is already oriented so
    negative always means the impaired direction), and this participant's
    own z picked out with a dashed line -- purely descriptive, not a
    Typical/Atypical judgement (see interpretation.py and
    percentile_age_adjusted() for the card's own number). Same visual
    language as the fluency tabs' _headline_distribution_figure (violin +
    shaded reference range + dashed participant line) -- just one measure
    per figure instead of a stacked panel, so it fits inside its own card.
    """
    fig = go.Figure()

    fig.add_vrect(x0=-1, x1=1, fillcolor=GROUP_COLOURS["HC"], opacity=0.12, line_width=0)

    for group in _GROUPS:
        vals = CT_HEADLINE_Z.loc[CT_HEADLINE_Z["diagnosis"] == group, metric_key].dropna().values
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


@module.ui
def cookie_theft_ui():
    return ui.div(
        # PAGE HEADER
        ui.div(
            ui.div(
                ui.h3(
                    "Cookie Theft",
                    style="font-weight:700;color:#243447;margin:0;"
                ),
                ui.p(
                    "Picture description assessment",
                    style="color:#6c757d;margin:4px 0 0 0;"
                ),
            ),
            ui.div(
                ui.span(
                    "CLINICAL ASSESSMENT",
                    style=(
                        "font-size:11px;font-weight:700;letter-spacing:1px;"
                        "color:#426b83;background:#eaf2f6;"
                        "padding:7px 12px;border-radius:20px;"
                    )
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

        # HEADLINE MEASURES
        ui.div(
            ui.div(
                ui.h5(
                    "Clinical summary",
                    style="font-weight:700;color:#243447;margin-bottom:4px;"
                ),
                ui.p(
                    "Key features from the participant's picture description, "
                    "each shown as a value, an age-adjusted percentile, and a "
                    "group-comparison graph — no Typical/Borderline/Atypical "
                    "label here, just where this score falls relative to the "
                    "age-matched healthy-control group, for a clinician to "
                    "weigh themselves (shaded band on each graph = HC typical "
                    "range, z in [-1, 1]).",
                    style="color:#6c757d;font-size:13px;margin-bottom:16px;"
                ),
            ),
            ui.output_ui("headline_panel"),
            style="margin-bottom:26px;"
        ),

        # TRANSCRIPT, PICTURE, AND INFORMATION UNITS -- side by side, same height
        ui.div(
            ui.h5(
                "Transcript and picture content",
                style="font-weight:700;color:#243447;margin-bottom:5px;"
            ),
            ui.p(
                "Matched words and phrases are highlighted in the transcript, "
                "shown alongside the spatio-semantic graph and the "
                "information units checklist.",
                style="color:#6c757d;font-size:13px;margin-bottom:16px;"
            ),
            ui.layout_columns(
                ui.div(
                    ui.h6(
                        "Transcript",
                        style="font-weight:600;color:#34495e;"
                    ),
                    ui.output_ui("transcript_display"),
                    style=(
                        "background:white;"
                        "border:1px solid #e1e8ef;"
                        "border-top:4px solid #527d96;"
                        "border-radius:12px;"
                        "padding:22px;"
                        "min-width:0;"
                        "box-shadow:0 2px 8px rgba(35,55,75,0.05);"
                    ),
                ),
                ui.div(
                    ui.h6(
                        "Spatio-semantic graph",
                        style="font-weight:600;color:#34495e;"
                    ),
                    ui.p(
                        "Shows the order in which picture content was mentioned. "
                        "Circle size reflects how many times that item was "
                        "mentioned; hover a circle for the exact count.",
                        style="color:#6c757d;font-size:12px;"
                    ),
                    output_widget("graph_overlay"),
                    style=(
                        "background:white;"
                        "border:1px solid #e1e8ef;"
                        "border-top:4px solid #527d96;"
                        "border-radius:12px;"
                        "padding:22px;"
                        "min-width:0;"
                        "box-shadow:0 2px 8px rgba(35,55,75,0.05);"
                    ),
                ),
                ui.div(
                    ui.h6(
                        "Information units",
                        style="font-weight:600;color:#34495e;"
                    ),
                    ui.p(
                        "Which picture elements were identified in the transcript.",
                        style="color:#6c757d;font-size:12px;"
                    ),
                    ui.output_ui("iu_checklist_display"),
                    style=(
                        "background:white;"
                        "border:1px solid #e1e8ef;"
                        "border-top:4px solid #527d96;"
                        "border-radius:12px;"
                        "padding:22px;"
                        "min-width:0;"
                        "box-shadow:0 2px 8px rgba(35,55,75,0.05);"
                    ),
                ),
                col_widths=[4, 5, 3],
            ),
            style="margin-bottom:26px;"
        ),

        # ADDITIONAL ANALYSIS
        ui.div(
            ui.h5(
                "Additional language analysis",
                style="font-weight:700;color:#243447;margin-bottom:5px;"
            ),
            ui.p(
                "A wider set of language and scanning measures, sorted by how "
                "well each separates the diagnostic groups in this cohort. "
                "Each only separates some of the three groups from the "
                "others, not all three (see its own description for which) "
                "— treat these as supporting context, not independent flags "
                "in their own right. Measures with no reliable signal here, "
                "or that ran opposite to their expected direction, are left "
                "out rather than shown with a caveat attached.",
                style="color:#6c757d;font-size:13px;margin-bottom:16px;"
            ),
            ui.div(
                ui.accordion(
                    ui.accordion_panel(
                        "Additional measures",
                        ui.p(
                            "Supporting context for qualitative review, not "
                            "an independent diagnosis on their own.",
                            style="color:#6c757d;font-size:12px;"
                        ),
                        ui.output_ui("exploratory_panel"),
                    ),
                    open=False,
                ),
                style=(
                    "background:white;"
                    "border:1px solid #e1e8ef;"
                    "border-top:4px solid #527d96;"
                    "border-radius:12px;"
                    "padding:22px;"
                    "min-width:0;"
                    "box-shadow:0 2px 8px rgba(35,55,75,0.05);"
                ),
            ),
        ),

        style=(
            "background:#f5f7fa;padding:22px;"
            "border-radius:14px;"
        ),
    )

@module.server
def cookie_theft_server(input, output, session, participant_id):
    """participant_id is a reactive returning the selected participant string."""

    @reactive.calc
    def metrics_row():
        pid = participant_id()
        row = ct_metrics_df[ct_metrics_df["participant_id"].astype(str) == str(pid)]
        return row.iloc[0].to_dict() if len(row) else {}

    @reactive.calc
    def iu_rows():
        pid = participant_id()
        return ct_iu_df[ct_iu_df["participant_id"].astype(str) == str(pid)].copy()

    @reactive.calc
    def iu_counts():
        df = iu_rows()
        return dict(zip(df["iu"], df["count"]))

    # ── Clinical summary cards ──
    @render.ui
    def headline_panel():
        m = metrics_row()

        if not m:
            return ui.p("No data for this participant.")

        pid = participant_id()
        age = DEMOGRAPHICS.get(str(pid), {}).get("age")
        age_known = age is not None and pd.notna(age)

        cards = []

        for measure in HEADLINE_MEASURES:
            value = m.get(measure.key)
            intercept, slope, residual_sd = HC_STATS.get(measure.key, (None, None, None))

            # Check that the participant's value is available
            valid_value = value is not None and pd.notna(value)

            # Calculate the percentile only when the reference regression and
            # this participant's own age are both available --
            # percentile_age_adjusted predicts the reference mean from
            # `age`, so without it there's nothing to compare against.
            valid_reference = (
                age_known
                and intercept is not None
                and slope is not None
                and residual_sd is not None
                and pd.notna(residual_sd)
                and residual_sd > 0
            )

            percentile = None
            if valid_value and valid_reference:
                percentile = percentile_age_adjusted(
                    value, age, intercept, slope, residual_sd, measure.higher_is_worse,
                )

            value_str = (
                measure.fmt.format(value)
                if valid_value
                else "—"
            )

            extra_rows = []

            # Show the information-unit count and progress bar
            if (
                measure.key == "iu_score"
                and valid_value
                and pd.notna(m.get("mentioned_ius"))
                and pd.notna(m.get("total_ius"))
            ):
                mentioned = int(m["mentioned_ius"])
                total = int(m["total_ius"])

                value_str = f"{mentioned}/{total}"

                extra_rows.append(
                    _progress_bar(value)
                )

            # Healthy-control reference range, predicted for this
            # participant's own age (see fit_age_adjusted_norms).
            if valid_reference:
                predicted_mean = intercept + slope * age
                lower = predicted_mean - residual_sd
                upper = predicted_mean + residual_sd

                range_str = (
                    f"HC reference (age {int(age)}): "
                    f"{measure.fmt.format(lower)} – "
                    f"{measure.fmt.format(upper)}"
                )
            else:
                range_str = "HC reference unavailable"

            card = ui.div(
                ui.div(
                    measure.label,
                    style="font-size:14px;font-weight:600;color:#495057;"
                ),

                ui.div(
                    value_str,
                    style="font-size:30px;font-weight:700;color:#212529;margin:10px 0;"
                ),

                *extra_rows,

                ui.div(
                    measure.description,
                    style="font-size:13px;color:#6c757d;margin-top:10px;"
                ),

                ui.div(
                    _percentile_text(percentile),
                    style="font-size:13px;font-weight:600;color:#34495e;"
                ),

                ui.div(
                    output_widget(f"headline_graph_{measure.key}"),
                    style="margin-top:10px;"
                ),

                ui.div(
                    range_str,
                    style="font-size:12px;color:#868e96;margin-top:6px;"
                ),

                style=(
                    "background:white;"
                    "border:1px solid #e1e8ef;"
                    "border-top:4px solid #527d96;"
                    "border-radius:12px;"
                    "padding:22px;"
                    "min-width:0;"
                    "box-shadow:0 2px 8px rgba(35,55,75,0.05);"
                )
            )

            cards.append(card)

        return ui.div(
            *cards,
            style=(
                "display:grid;"
                "grid-template-columns:repeat(3,minmax(0,1fr));"
                "gap:16px;"
                "width:100%;"
            )
        )

    # ── Graph overlay ──
    @render_widget
    def graph_overlay():
        m = metrics_row()
        seq_full = parse_sequence(m.get("iu_sequence_full", "[]"))
        if not seq_full:
            return go.Figure()
        return render_graph_plotly(seq_full, iu_counts())

    def _participant_z(metric_key):
        """This participant's age-adjusted z-score on one headline measure
        (see CT_HEADLINE_Z) -- shared by the three per-card graphs below.
        """
        pid = participant_id()
        z_row = CT_HEADLINE_Z[CT_HEADLINE_Z["participant_id"].astype(str) == str(pid)]
        return z_row.iloc[0][metric_key] if len(z_row) else None

    # ── Per-card group-comparison graphs (one per headline measure) ──
    @render_widget
    def headline_graph_iu_score():
        return _card_distribution_figure("iu_score", _participant_z("iu_score"))

    @render_widget
    def headline_graph_total_words():
        return _card_distribution_figure("total_words", _participant_z("total_words"))

    @render_widget
    def headline_graph_path_distance_per_100_words():
        return _card_distribution_figure(
            "path_distance_per_100_words",
            _participant_z("path_distance_per_100_words"),
        )

    # ── Exploratory measures: value + age-adjusted percentile + graph ──
    @render.ui
    def exploratory_panel():
        m = metrics_row()
        if not m:
            return ui.p("No data for this participant.")
        pid = participant_id()
        age = DEMOGRAPHICS.get(str(pid), {}).get("age")
        cards = [
            _exploratory_card(meas, m.get(meas.key), age, *HC_STATS.get(meas.key, (None, None, None)))
            for meas in EXPLORATORY_MEASURES
        ]
        return ui.div(
            *cards,
            style=(
                "display:grid;"
                "grid-template-columns:repeat(3,minmax(0,1fr));"
                "gap:12px;"
            ),
        )

    # ── Per-card group-comparison graphs (one per exploratory measure) --
    # same mechanics as the headline cards' graphs above.
    @render_widget
    def exploratory_graph_mattr():
        return _card_distribution_figure("mattr", _participant_z("mattr"))

    @render_widget
    def exploratory_graph_total_path_per_node():
        return _card_distribution_figure("total_path_per_node", _participant_z("total_path_per_node"))

    @render_widget
    def exploratory_graph_cross_ratio_quadrants():
        return _card_distribution_figure(
            "cross_ratio_quadrants", _participant_z("cross_ratio_quadrants")
        )

    # ── Same, for the 6 measures added 2026-11-17 (see interpretation.py) ──
    @render_widget
    def exploratory_graph_speech_duration_seconds():
        return _card_distribution_figure(
            "speech_duration_seconds", _participant_z("speech_duration_seconds")
        )

    @render_widget
    def exploratory_graph_repetition_rate_per_100_words():
        return _card_distribution_figure(
            "repetition_rate_per_100_words", _participant_z("repetition_rate_per_100_words")
        )

    @render_widget
    def exploratory_graph_noun_verb_ratio():
        return _card_distribution_figure("noun_verb_ratio", _participant_z("noun_verb_ratio"))

    @render_widget
    def exploratory_graph_syllables_per_content_unit():
        return _card_distribution_figure(
            "syllables_per_content_unit", _participant_z("syllables_per_content_unit")
        )

    @render_widget
    def exploratory_graph_left_right_cu_ratio():
        return _card_distribution_figure(
            "left_right_cu_ratio", _participant_z("left_right_cu_ratio")
        )

    @render_widget
    def exploratory_graph_content_units_per_minute():
        return _card_distribution_figure(
            "content_units_per_minute", _participant_z("content_units_per_minute")
        )

    # ── Transcript with IUs highlighted ──
    @render.ui
    def transcript_display():
        import re

        pid = participant_id()

        path = CTD_TRANSCRIPT_PATHS.get(str(pid))

        if path is None:
            return ui.p("Transcript file not found for this participant.")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

        # Same cleaning scoring.py's load_transcript() applies before
        # scoring -- Pat's own speech only, transcriber annotations
        # stripped -- so what's shown here is what was actually scored,
        # not the full file with Oth/Int turns and (pause)/<laugh>-style
        # annotations still in it. Not lowercased (unlike load_transcript):
        # that's only needed for scoring's own matching, and reads worse
        # here for no benefit, since the highlighting below is already
        # case-insensitive.
        text = extract_patient_speech(raw_text).strip()

        # Build term -> IU map for highlighting
        term_to_iu = {}

        for iu, terms in SYNONYMS.items():
            for term in terms:
                term_to_iu[term.lower()] = iu

        # Highlight matched terms, longest first
        terms_sorted = sorted(term_to_iu.keys(), key=len, reverse=True)

        pattern = (
            r'\b('
            + '|'.join(re.escape(term) for term in terms_sorted)
            + r')\b'
        )

        def repl(match):
            word = match.group(0)
            iu = term_to_iu.get(word.lower(), "")

            return (
                f'<mark style="background:#ffe08a;'
                f'border-radius:3px;padding:2px 4px;" '
                f'title="{iu}">{word}</mark>'
            )

        highlighted = re.sub(
            pattern,
            repl,
            text,
            flags=re.IGNORECASE
        )

        return ui.div(
            ui.HTML(highlighted),
            style=(
                "line-height:2;"
                "padding:18px;"
                "background:#fbfcfd;"
                "border:1px solid #e5eaf0;"
                "border-radius:10px;"
                "font-size:14px;"
                "white-space:pre-wrap;"
                f"max-height:{_PANEL_SCROLL_HEIGHT};"
                "overflow-y:auto;"
            ),
        )

    # ── Information unit checklist (now shown next to the picture, not the transcript) ──
    @render.ui
    def iu_checklist_display():
        counts = iu_counts()

        # Build the checklist from the IU categories in SYNONYMS
        checklist_items = []

        for iu in sorted(SYNONYMS.keys()):
            count = counts.get(iu, 0)

            try:
                count = int(count) if pd.notna(count) else 0
            except (TypeError, ValueError):
                count = 0

            found = count > 0

            if found:
                status_text = "Identified"
                status_colour = "#287a55"
                background = "#eaf5ef"
                symbol = "✓"
            else:
                status_text = "Not identified"
                status_colour = "#8a6d3b"
                background = "#f7f3e9"
                symbol = "–"

            checklist_items.append(
                ui.div(
                    ui.div(
                        ui.span(
                            symbol,
                            style=(
                                f"font-weight:700;color:{status_colour};"
                                "margin-right:8px;"
                            )
                        ),
                        ui.span(
                            iu,
                            style="font-size:13px;font-weight:600;color:#34495e;"
                        ),
                        style="display:flex;align-items:center;"
                    ),
                    ui.span(
                        status_text,
                        style=(
                            f"font-size:11px;color:{status_colour};"
                            f"background:{background};"
                            "padding:4px 8px;border-radius:12px;"
                        )
                    ),
                    style=(
                        "display:flex;justify-content:space-between;"
                        "align-items:center;gap:8px;"
                        "padding:9px 10px;"
                        "border-bottom:1px solid #edf0f2;"
                    )
                )
            )

        identified = sum(
            1 for iu in SYNONYMS
            if counts.get(iu, 0) > 0
        )

        total = len(SYNONYMS)

        return ui.div(
            ui.div(
                ui.span(
                    f"{identified}/{total}",
                    style="font-size:27px;font-weight:700;color:#243447;"
                ),
                ui.span(
                    " identified",
                    style="font-size:13px;color:#6c757d;"
                ),
                style="margin-bottom:12px;"
            ),
            ui.div(
                *checklist_items,
                style=(
                    "background:#fbfcfd;"
                    "border:1px solid #e5eaf0;"
                    "border-radius:10px;"
                    "padding:6px 10px;"
                    f"max-height:{_PANEL_SCROLL_HEIGHT};"
                    "overflow-y:auto;"
                ),
            ),
        )
