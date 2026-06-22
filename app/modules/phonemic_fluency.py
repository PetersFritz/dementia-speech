"""Phonemic fluency tab module (refactored from original app.py)."""
import numpy as np
from scipy import stats
from shiny import module, ui, render, reactive
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shinywidgets import output_widget, render_widget

from shared import (
    pft_words_df, pft_metrics_df,
    CLASSIFICATION_COLOURS, CLASSIFICATION_LABELS, GROUP_COLOURS,
)


@module.ui
def phonemic_fluency_ui():
    return ui.div(
        ui.layout_columns(
            ui.value_box("Total Words", ui.output_text("m_total"), theme="primary"),
            ui.value_box("Correct", ui.output_text("m_correct"), theme="success"),
            ui.value_box("Intrusions", ui.output_text("m_intrusions"), theme="danger"),
            ui.value_box("Repetitions", ui.output_text("m_repetitions"), theme="warning"),
            ui.value_box("Variants", ui.output_text("m_variants"), theme="secondary"),
            col_widths=[2, 2, 2, 2, 2],
        ),
        ui.hr(),
        ui.h5("Transcript"),
        ui.output_ui("word_display"),
        ui.hr(),
        ui.layout_columns(
            ui.div(
                ui.h5("Performance vs groups"),
                output_widget("summary_plot"),
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

    @render.text
    def m_total():
        return str(pmetrics().get("total_words", "—"))

    @render.text
    def m_correct():
        return str(pmetrics().get("correct", "—"))

    @render.text
    def m_intrusions():
        return str(pmetrics().get("intrusions", "—"))

    @render.text
    def m_repetitions():
        return str(pmetrics().get("repetitions", "—"))

    @render.text
    def m_variants():
        return str(pmetrics().get("variants", "—"))

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
    def summary_plot():
        m = pmetrics()
        if not m:
            return go.Figure()
        metrics_to_show = ["correct", "intrusions", "repetitions"]
        labels = ["Correct", "Intrusions", "Repetitions"]
        groups = ["HC", "MCI", "Dementia"]

        fig = make_subplots(rows=1, cols=len(metrics_to_show),
                            subplot_titles=labels, horizontal_spacing=0.06)

        for m_idx, metric in enumerate(metrics_to_show):
            col = m_idx + 1
            for group in groups:
                gd = pft_metrics_df[pft_metrics_df["diagnosis"] == group]
                vals = gd[metric].dropna().values
                if len(vals) < 2:
                    continue
                mean = np.mean(vals)
                ci = stats.sem(vals) * stats.t.ppf(0.975, df=len(vals) - 1)
                fig.add_trace(go.Scatter(
                    x=[group], y=[mean],
                    error_y=dict(type="data", array=[ci], visible=True,
                                 color=GROUP_COLOURS[group], thickness=2, width=6),
                    mode="markers", marker=dict(color=GROUP_COLOURS[group], size=10),
                    showlegend=False,
                ), row=1, col=col)

            pval = m.get(metric, 0)
            fig.add_shape(type="line", x0=-0.5, x1=2.5, y0=pval, y1=pval,
                          line=dict(color="#2c3e50", width=3), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=[groups[1]], y=[pval], mode="markers+text",
                marker=dict(color="#2c3e50", size=14, symbol="diamond",
                            line=dict(color="white", width=1.5)),
                text=["You"], textposition="top center",
                textfont=dict(color="#2c3e50", size=10),
                showlegend=False,
            ), row=1, col=col)

        fig.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10),
                          plot_bgcolor="white")
        fig.update_yaxes(gridcolor="#eee", rangemode="tozero")
        fig.update_xaxes(showticklabels=False)
        return fig

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