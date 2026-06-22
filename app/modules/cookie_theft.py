"""Cookie Theft picture description tab module."""
import numpy as np
from scipy import stats
from shiny import module, ui, render, reactive
import plotly.graph_objects as go
from shinywidgets import output_widget, render_widget

from shared import (
    ct_iu_df, ct_metrics_df, GROUP_COLOURS, parse_sequence
)
from dementia_speech.tasks.picture_description.graph import render_graph_plotly
from dementia_speech.tasks.picture_description.scoring import SYNONYMS

# Graph metrics to compare against groups
GRAPH_METRICS = [
    "mentioned_ius", "total_path_distance", "unique_nodes",
    "cycles", "self_cycles", "cross_ratio_quadrants",
]
GRAPH_METRIC_LABELS = {
    "mentioned_ius": "IUs mentioned",
    "total_path_distance": "Path distance",
    "unique_nodes": "Unique nodes",
    "cycles": "Cycles",
    "self_cycles": "Self-cycles",
    "cross_ratio_quadrants": "Cross-quadrant ratio",
}


@module.ui
def cookie_theft_ui():
    return ui.div(
        ui.layout_columns(
            ui.value_box("IUs Mentioned", ui.output_text("vb_mentioned"), theme="primary"),
            ui.value_box("Total Mentions", ui.output_text("vb_total_mentions"), theme="secondary"),
            ui.value_box("Path Distance", ui.output_text("vb_path"), theme="secondary"),
            ui.value_box("Cycles", ui.output_text("vb_cycles"), theme="warning"),
            col_widths=[3, 3, 3, 3],
        ),
        ui.hr(),
        ui.h5("Transcript with information units highlighted"),
        ui.output_ui("transcript_display"),
        ui.hr(),
        ui.layout_columns(
            ui.div(
                ui.h5("Spatio-semantic graph"),
                output_widget("graph_overlay"),
                style="background:white;border:1px solid #e0e0e0;border-radius:8px;padding:1rem;",
            ),
            ui.div(
                ui.h5("Graph metrics vs groups"),
                output_widget("metrics_plot"),
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

    # ── Value boxes ──
    @render.text
    def vb_mentioned():
        m = metrics_row()
        return f"{int(m.get('mentioned_ius', 0))} / {int(m.get('total_ius', 20))}"

    @render.text
    def vb_total_mentions():
        return str(int(metrics_row().get("total_mentions", 0)))

    @render.text
    def vb_path():
        v = metrics_row().get("total_path_distance", 0)
        return f"{v:.0f}"

    @render.text
    def vb_cycles():
        return str(int(metrics_row().get("cycles", 0)))

    # ── Graph overlay ──
    @render_widget
    def graph_overlay():
        m = metrics_row()
        seq_full = parse_sequence(m.get("iu_sequence_full", "[]"))
        if not seq_full:
            return go.Figure()
        return render_graph_plotly(seq_full, iu_counts())

    # ── Metrics vs groups (small multiples) ──
    @render_widget
    def metrics_plot():
        m = metrics_row()
        if not m:
            return go.Figure()

        from plotly.subplots import make_subplots
        groups = ["HC", "MCI", "Dementia"]
        labels = [GRAPH_METRIC_LABELS[x] for x in GRAPH_METRICS]

        fig = make_subplots(rows=len(GRAPH_METRICS), cols=1,
                            subplot_titles=labels, vertical_spacing=0.08)

        for m_idx, metric in enumerate(GRAPH_METRICS):
            row = m_idx + 1
            for group in groups:
                gd = ct_metrics_df[ct_metrics_df["diagnosis"] == group]
                if len(gd) < 2 or metric not in gd:
                    continue
                vals = gd[metric].dropna().values
                if len(vals) < 2:
                    continue
                mean = np.mean(vals)
                ci = stats.sem(vals) * stats.t.ppf(0.975, df=len(vals) - 1)
                fig.add_trace(go.Scatter(
                    x=[mean], y=[group],
                    error_x=dict(type="data", array=[ci], visible=True,
                                 color=GROUP_COLOURS[group], thickness=2, width=5),
                    mode="markers",
                    marker=dict(color=GROUP_COLOURS[group], size=8),
                    showlegend=False,
                ), row=row, col=1)

            pval = m.get(metric, 0)
            fig.add_vline(x=pval, line=dict(color="#2c3e50", width=2, dash="dash"),
                          row=row, col=1)

        fig.update_layout(height=130 * len(GRAPH_METRICS),
                          margin=dict(t=30, b=10, l=10, r=10),
                          plot_bgcolor="white")
        fig.update_xaxes(gridcolor="#eee")
        fig.update_yaxes(gridcolor="#eee")
        return fig

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