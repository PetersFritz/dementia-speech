"""Phonemic Fluency Dashboard - minimal Shiny app."""
from pathlib import Path
import pandas as pd
from shiny import App, ui, render, reactive
import plotly.graph_objects as go
from shinywidgets import output_widget, render_widget
import numpy as np
from scipy import stats

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
WORDS_PATH = DATA_DIR / "pft_words.csv"
METRICS_PATH = DATA_DIR / "pft_metrics.csv"

words_df = pd.read_csv(WORDS_PATH)
metrics_df = pd.read_csv(METRICS_PATH)

PARTICIPANTS = sorted(words_df["participant_id"].unique().tolist())

# ── Colours ───────────────────────────────────────────────────────────────────
CLASSIFICATION_COLOURS = {
    "correct":    "#2ecc71",   # green
    "intrusion":  "#e74c3c",   # red
    "repetition": "#e67e22",   # orange
    "variant":    "#95a5a6",   # grey
}

CLASSIFICATION_LABELS = {
    "correct":    "Correct",
    "intrusion":  "Intrusion",
    "repetition": "Repetition",
    "variant":    "Variant",
}

GROUP_COLOURS = {
    "HC":       "#2ecc71",
    "MCI":      "#e67e22",
    "Dementia": "#e74c3c",
}

# ── UI ────────────────────────────────────────────────────────────────────────
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Phonemic Fluency"),
        ui.p("Letter: P", style="color: #666; font-size: 0.9em;"),
        ui.hr(),
        ui.input_select(
            "participant",
            "Select Participant",
            choices={str(p): f"Participant {p}" for p in PARTICIPANTS},
        ),
        ui.hr(),
        # Legend
        ui.h6("Legend"),
        *[
            ui.div(
                ui.span(
                    "●",
                    style=f"color: {colour}; font-size: 1.2em; margin-right: 6px;"
                ),
                CLASSIFICATION_LABELS[cls],
                style="margin-bottom: 4px;"
            )
            for cls, colour in CLASSIFICATION_COLOURS.items()
        ],
        width=240,
    ),
    ui.h3(ui.output_text("participant_title")),
    # Metrics row
    ui.layout_columns(
        ui.value_box(
            "Total Words",
            ui.output_text("metric_total"),
            theme="primary",
        ),
        ui.value_box(
            "Correct",
            ui.output_text("metric_correct"),
            theme="success",
        ),
        ui.value_box(
            "Intrusions",
            ui.output_text("metric_intrusions"),
            theme="danger",
        ),
        ui.value_box(
            "Repetitions",
            ui.output_text("metric_repetitions"),
            theme="warning",
        ),
        ui.value_box(
            "Variants",
            ui.output_text("metric_variants"),
            theme="secondary",
        ),
        col_widths=[2, 2, 2, 2, 2],
    ),
    ui.hr(),
    ui.h5("Transcript"),
    ui.output_ui("word_display"),
    ui.hr(),
    ui.h5("Overall Performance vs Group Means"),
    output_widget("summary_plot"),
    ui.hr(),
    ui.h5("Words per 15-Second Bin"),
    output_widget("bin_plot"),
    ui.div(
        ui.h5("Metric profile", style="margin: 0 0 10px;"),
        output_widget("radar_plot"),
        style=(
            "background: var(--bs-body-bg, white);"
            "border: 1px solid #e0e0e0;"
            "border-radius: 8px;"
            "padding: 1rem 1.25rem;"
            "margin-bottom: 1rem;"
        )
    ),
    title="Dementia Speech — Phonemic Fluency",
)

# ── Server ────────────────────────────────────────────────────────────────────
def server(input, output, session):

    @reactive.calc
    def participant_data():
        pid = input.participant()
        return words_df[words_df["participant_id"].astype(str) == str(pid)].copy()

    @reactive.calc
    def participant_metrics():
        pid = input.participant()
        row = metrics_df[metrics_df["participant_id"].astype(str) == str(pid)]
        if len(row) == 0:
            return {}
        return row.iloc[0].to_dict()

    @render.text
    def participant_title():
        return f"Participant {input.participant()} — Letter P"

    @render.text
    def metric_total():
        m = participant_metrics()
        return str(m.get("total_words", "—"))

    @render.text
    def metric_correct():
        m = participant_metrics()
        return str(m.get("correct", "—"))

    @render.text
    def metric_intrusions():
        m = participant_metrics()
        return str(m.get("intrusions", "—"))

    @render.text
    def metric_repetitions():
        m = participant_metrics()
        return str(m.get("repetitions", "—"))

    @render.text
    def metric_variants():
        m = participant_metrics()
        return str(m.get("variants", "—"))

    @render.ui
    def word_display():
        df = participant_data()
        if len(df) == 0:
            return ui.p("No words found for this participant.")

        word_spans = []
        for _, row in df.iterrows():
            word = row["word"]
            cls = row["classification"]
            colour = CLASSIFICATION_COLOURS.get(cls, "#333")
            label = CLASSIFICATION_LABELS.get(cls, cls)

            word_spans.append(
                ui.span(
                    word,
                    title=f"{label} ({row['start']:.2f}s – {row['end']:.2f}s)",
                    style=(
                        f"display: inline-block;"
                        f"background-color: {colour}22;"   # 22 = ~13% opacity
                        f"color: {colour};"
                        f"border: 1.5px solid {colour};"
                        f"border-radius: 4px;"
                        f"padding: 3px 8px;"
                        f"margin: 3px 4px;"
                        f"font-size: 1.05em;"
                        f"font-weight: 500;"
                        f"cursor: default;"
                    )
                )
            )

        return ui.div(
            *word_spans,
            style="line-height: 2.8; padding: 12px;"
        )
    
    @render_widget
    def bin_plot():
        df = participant_metrics()
        if not df:
            return go.Figure()

        bins = ['bin_1', 'bin_2', 'bin_3', 'bin_4']
        bin_labels = ['0-15s', '15-30s', '30-45s', '45-60s']

        group_means = metrics_df.groupby('diagnosis')[bins].mean()

        fig = go.Figure()

        colours = {'HC': '#2ecc71', 'MCI': '#e67e22', 'Dementia': '#e74c3c'}
        for group, colour in colours.items():
            if group in group_means.index:
                fig.add_trace(go.Scatter(
                    x=bin_labels,
                    y=group_means.loc[group, bins].values,
                    mode='lines+markers',
                    name=group,
                    line=dict(color=colour, width=3, dash='dash'),
                    marker=dict(size=6),
                ))

        participant_bins = [df.get(b, 0) for b in bins]
        fig.add_trace(go.Scatter(
            x=bin_labels,
            y=participant_bins,
            mode='lines+markers',
            name=f"Participant {input.participant()}",
            line=dict(color='#2c3e50', width=3),
            marker=dict(size=9),
        ))

        fig.update_layout(
            xaxis_title="Time Bin",
            yaxis_title="Correct Words",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=20, b=20),
            height=350,
            plot_bgcolor='white',
            yaxis=dict(gridcolor='#eee', rangemode='tozero'),
            xaxis=dict(gridcolor='#eee'),
        )

        return fig
    
    @render_widget
    def summary_plot():
        df = participant_metrics()
        if not df:
            return go.Figure()

        metrics_to_show = ["correct", "intrusions", "repetitions"]
        labels = ["Correct", "Intrusions", "Repetitions", "Variants"]
        groups = ["HC", "MCI", "Dementia"]
        colours = {"HC": "#1D9E75", "MCI": "#BA7517", "Dementia": "#D85A30"}

        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1,
            cols=len(metrics_to_show),
            subplot_titles=labels,
            shared_yaxes=False,
            horizontal_spacing=0.05,
        )

        for m_idx, metric in enumerate(metrics_to_show):
            col = m_idx + 1

            for group in groups:
                group_data = metrics_df[metrics_df["diagnosis"] == group]
                if len(group_data) == 0:
                    continue
                values = group_data[metric].dropna().values
                if len(values) < 2:
                    continue
                mean = np.mean(values)
                se = stats.sem(values)
                ci = se * stats.t.ppf(0.975, df=len(values) - 1)
                colour = colours[group]

                fig.add_trace(go.Scatter(
                    x=[group],
                    y=[mean],
                    error_y=dict(
                        type="data",
                        array=[ci],
                        visible=True,
                        color=colour,
                        thickness=2,
                        width=6,
                    ),
                    mode="markers",
                    name=group,
                    marker=dict(color=colour, size=10),
                    showlegend=(m_idx == 0),  # only show legend once
                ),
                row=1, col=col)

            # Participant horizontal line across the panel
            participant_val = df.get(metric, 0)
            fig.add_hline(
                y=participant_val,
                line=dict(color="#ff0101", width=4, dash="dash"),
                row=1, col=col,
                annotation_text="participant" if m_idx == len(metrics_to_show) - 1 else "",
                annotation_font_size=10,
            )

        fig.update_layout(
            height=320,
            margin=dict(t=40, b=10, l=10, r=10),
            plot_bgcolor="white",
            legend=dict(orientation="h", y=-0.15),
        )

        fig.update_yaxes(gridcolor="#eee", rangemode="tozero")
        fig.update_xaxes(showticklabels=False)

        return fig
    
    @render_widget
    def radar_plot():
        df = participant_metrics()
        if not df:
            return go.Figure()

        metrics_to_show = ["correct", "intrusions", "repetitions", "variants"]
        labels = ["Correct", "Intrusions", "Repetitions", "Variants"]
        groups = ["HC", "MCI", "Dementia"]
        colours = {"HC": "#1D9E75", "MCI": "#BA7517", "Dementia": "#D85A30"}

        # Per-metric scale: max of (all group means, participant value)
        metric_maxes = {}
        for m in metrics_to_show:
            group_max = max(
                metrics_df.groupby("diagnosis")[m].mean().max(),
                1  # floor to avoid div by zero
            )
            participant_val = df.get(m, 0)
            metric_maxes[m] = max(group_max, participant_val)

            

        def normalise(values):
            return [v / metric_maxes[m] for v, m in zip(values, metrics_to_show)]

        # Build display labels with max shown so clinician knows the scale
        axis_labels = [
            f"{label}<br><span style='font-size:9px'>(max {int(metric_maxes[m])})</span>"
            for label, m in zip(labels, metrics_to_show)
        ]

        labels_closed = axis_labels + [axis_labels[0]]
        fig = go.Figure()

        # Group mean traces
        for group in groups:
            group_data = metrics_df[metrics_df["diagnosis"] == group]
            if len(group_data) == 0:
                continue
            raw = [np.mean(group_data[m].dropna().values) for m in metrics_to_show]
            norm = normalise(raw)
            norm_closed = norm + [norm[0]]
            colour = colours[group]

            fig.add_trace(go.Scatterpolar(
                r=norm_closed,
                theta=labels_closed,
                mode="lines",
                name=group,
                line=dict(color=colour, width=2, dash="dash"),
                fill="toself",
                fillcolor=f"rgba({int(colour[1:3],16)},{int(colour[3:5],16)},{int(colour[5:7],16)},0.07)",
                hovertemplate=[
                    f"{group}<br>{label}: {raw[i]:.1f}<extra></extra>"
                    for i, label in enumerate(labels)
                ] + [f"{group}<br>{labels[0]}: {raw[0]:.1f}<extra></extra>"],
            ))

        # Participant trace
        participant_raw = [df.get(m, 0) for m in metrics_to_show]
        participant_norm = normalise(participant_raw)
        participant_closed = participant_norm + [participant_norm[0]]

        fig.add_trace(go.Scatterpolar(
            r=participant_closed,
            theta=labels_closed,
            mode="lines+markers",
            name=f"Participant {input.participant()}",
            line=dict(color="#2c3e50", width=3),
            marker=dict(color="#2c3e50", size=8, symbol="diamond"),
            fill="toself",
            fillcolor="rgba(44,62,80,0.1)",
            hovertemplate=[
                f"Participant<br>{label}: {participant_raw[i]}<extra></extra>"
                for i, label in enumerate(labels)
            ] + [f"Participant<br>{labels[0]}: {participant_raw[0]}<extra></extra>"],
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=["0%", "25%", "50%", "75%", "100%"],
                    gridcolor="#eee",
                ),
                angularaxis=dict(gridcolor="#eee"),
                bgcolor="white",
            ),
            legend=dict(orientation="h", y=-0.15),
            margin=dict(t=30, b=30, l=30, r=30),
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
        )

        return fig


app = App(app_ui, server)

# to run enter the following command in terminal:
# shiny run app/app.py --reload