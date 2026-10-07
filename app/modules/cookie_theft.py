"""Cookie Theft picture description tab module."""
from shiny import module, ui, render, reactive
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shinywidgets import output_widget, render_widget

from shared import (
    ct_iu_df, ct_metrics_df, GROUP_COLOURS, parse_sequence
)
from dementia_speech.tasks.picture_description.graph import render_graph_plotly
from dementia_speech.tasks.picture_description.scoring import SYNONYMS
from dementia_speech.tasks.picture_description.interpretation import (
    HEADLINE_MEASURES, EXPLORATORY_MEASURES,
)
from dementia_speech.interpretation import BAND_COLOURS, band

# HC reference stats for banding, computed once at import time (see
# interpretation.py for why only these 5 measures are banded, and
# docs/cookie_theft_scoring.md for the analysis behind that choice).
_HC = ct_metrics_df[ct_metrics_df["diagnosis"] == "HC"]
HC_STATS = {
    m.key: (_HC[m.key].mean(), _HC[m.key].std())
    for m in HEADLINE_MEASURES if m.key in _HC
}

_GROUPS = ["HC", "MCI", "Dementia"]


def _group_distribution_figure(metric_keys, labels, participant_row):
    """One row per metric: a box plot of each group's actual scores (not the
    group mean's confidence interval — that shows how precisely the average
    is known, not how spread out individuals are, which is what matters for
    "is this patient's score within the normal range of people") plus a
    dashed line marking this participant's own value.
    """
    fig = make_subplots(rows=len(metric_keys), cols=1,
                        subplot_titles=labels, vertical_spacing=0.06)

    for m_idx, metric in enumerate(metric_keys):
        row = m_idx + 1
        for group in _GROUPS:
            vals = ct_metrics_df.loc[ct_metrics_df["diagnosis"] == group, metric].dropna().values
            if len(vals) < 2:
                continue
            fig.add_trace(go.Box(
                x=vals, y=[group] * len(vals), orientation="h",
                marker_color=GROUP_COLOURS[group], line=dict(width=1.5),
                boxpoints=False, showlegend=False,
            ), row=row, col=1)

        pval = participant_row.get(metric)
        if pval is not None:
            fig.add_vline(x=pval, line=dict(color="#2c3e50", width=2, dash="dash"),
                          row=row, col=1)

    fig.update_layout(height=110 * len(metric_keys),
                      margin=dict(t=30, b=10, l=10, r=10),
                      plot_bgcolor="white")
    fig.update_xaxes(gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee")
    return fig


@module.ui
def cookie_theft_ui():
    return ui.div(
        ui.h5("Clinical summary"),
        ui.p(
            "Each measure below is banded against a healthy-control reference "
            "group (within 1 SD = Typical, 1–2 SD = Borderline, beyond 2 SD = "
            "Atypical). Only measures that reliably separate diagnostic groups "
            "in this dataset are shown here — see “Additional measures” "
            "below for the rest.",
            style="color:#666;font-size:0.9em;max-width:70em;",
        ),
        ui.output_ui("headline_panel"),
        ui.h6("Compared with other participants", style="margin-top:1.2em;"),
        ui.p(
            "Each row shows the actual spread of scores in each group "
            "(box = middle 50%, whiskers = full range excluding outliers), "
            "not just the group average — the dashed line is this "
            "participant.",
            style="color:#666;font-size:0.85em;",
        ),
        ui.div(
            output_widget("headline_distribution"),
            style="background:white;border:1px solid #e0e0e0;border-radius:8px;padding:1rem;",
        ),
        ui.hr(),
        ui.h5("Transcript with information units highlighted"),
        ui.output_ui("transcript_display"),
        ui.hr(),
        ui.layout_columns(
            ui.div(
                ui.h5("Spatio-semantic graph"),
                ui.p(
                    "The order in which the participant mentioned picture "
                    "content, plotted over the image and connected in "
                    "sequence — useful for a qualitative look at how the "
                    "description moved around the picture.",
                    style="color:#666;font-size:0.85em;",
                ),
                output_widget("graph_overlay"),
                style="background:white;border:1px solid #e0e0e0;border-radius:8px;padding:1rem;",
            ),
            ui.div(
                ui.accordion(
                    ui.accordion_panel(
                        "Additional measures (exploratory — interpret with caution)",
                        ui.p(
                            "These measures did not reliably separate HC/MCI/"
                            "Dementia groups in this dataset, or moved in the "
                            "opposite direction to what the literature they're "
                            "drawn from reports — shown here as raw numbers "
                            "for research use, not as clinical flags.",
                            style="color:#666;font-size:0.85em;",
                        ),
                        output_widget("exploratory_plot"),
                    ),
                    open=False,
                ),
                style="background:white;border:1px solid #e0e0e0;border-radius:8px;padding:1rem;",
            ),
            col_widths=[7, 5],
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

        cards = []
        for measure in HEADLINE_MEASURES:
            value = m.get(measure.key)
            hc_mean, hc_sd = HC_STATS.get(measure.key, (None, None))
            label, z = band(value, hc_mean, hc_sd, measure.higher_is_worse)
            colour = BAND_COLOURS[label]
            value_str = measure.fmt.format(value) if value is not None else "—"
            # All headline measures are non-negative by construction (counts,
            # rates, ratios), so a mean-minus-1SD lower bound is clipped at 0
            # rather than shown as a nonsensical negative rate.
            range_str = (
                f"typical range ≈ {measure.fmt.format(max(0, hc_mean - hc_sd))}–{measure.fmt.format(hc_mean + hc_sd)}"
                if hc_mean is not None and hc_sd
                else "reference range unavailable"
            )
            cards.append(
                ui.div(
                    ui.div(
                        ui.span(label, style=(
                            f"background:{colour};color:white;border-radius:12px;"
                            "padding:2px 10px;font-size:0.8em;font-weight:600;"
                        )),
                        style="margin-bottom:6px;",
                    ),
                    ui.div(measure.label, style="font-weight:600;font-size:1.05em;"),
                    ui.div(value_str, style="font-size:1.6em;font-weight:700;margin:2px 0;"),
                    ui.div(measure.description, style="color:#666;font-size:0.82em;"),
                    ui.div(range_str, style="color:#999;font-size:0.78em;margin-top:4px;"),
                    style=(
                        "background:white;border:1px solid #e0e0e0;border-left:4px solid "
                        f"{colour};border-radius:8px;padding:12px 14px;min-width:12em;flex:1;"
                    ),
                )
            )

        return ui.div(*cards, style="display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;")

    # ── Graph overlay ──
    @render_widget
    def graph_overlay():
        m = metrics_row()
        seq_full = parse_sequence(m.get("iu_sequence_full", "[]"))
        if not seq_full:
            return go.Figure()
        return render_graph_plotly(seq_full, iu_counts())

    # ── Headline measures vs groups (actual distributions, not group-mean CI) ──
    @render_widget
    def headline_distribution():
        m = metrics_row()
        if not m:
            return go.Figure()
        metric_keys = [meas.key for meas in HEADLINE_MEASURES]
        labels = [meas.label for meas in HEADLINE_MEASURES]
        return _group_distribution_figure(metric_keys, labels, m)

    # ── Exploratory measures vs groups (small multiples) ──
    @render_widget
    def exploratory_plot():
        m = metrics_row()
        if not m:
            return go.Figure()
        metric_keys = list(EXPLORATORY_MEASURES.keys())
        labels = list(EXPLORATORY_MEASURES.values())
        return _group_distribution_figure(metric_keys, labels, m)

    # ── Transcript with IUs highlighted ──
    @render.ui
    def transcript_display():
        import re
        from pathlib import Path
        pid = participant_id()

        # Reconstruct transcript path — adjust to your structure if needed
        transcript_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "text" / "CTD"
        candidates = list(transcript_dir.glob(f"*{pid}*.txt"))
        if not candidates:
            return ui.p("Transcript file not found for this participant.")

        with open(candidates[0], "r", encoding="utf-8") as f:
            text = f.read()

        # Build term -> IU map for highlighting
        term_to_iu = {}
        for iu, terms in SYNONYMS.items():
            for t in terms:
                term_to_iu[t] = iu

        # Highlight matched terms (longest first to avoid partial overlap)
        terms_sorted = sorted(term_to_iu.keys(), key=len, reverse=True)
        pattern = r'\b(' + '|'.join(re.escape(t) for t in terms_sorted) + r')\b'

        def repl(match):
            word = match.group(0)
            iu = term_to_iu.get(word.lower(), "")
            return (
                f'<mark style="background:#ffe08a;border-radius:3px;padding:1px 3px;" '
                f'title="{iu}">{word}</mark>'
            )

        highlighted = re.sub(pattern, repl, text, flags=re.IGNORECASE)

        return ui.div(
            ui.HTML(highlighted),
            style="line-height:1.9;padding:12px;background:white;border:1px solid #e0e0e0;border-radius:8px;",
        )
