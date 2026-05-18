"""Phonemic Fluency Dashboard - minimal Shiny app."""
from pathlib import Path
import pandas as pd
from shiny import App, ui, render, reactive
import plotly.graph_objects as go
from shinywidgets import output_widget, render_widget

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
    "nonword": "#9b59b6",  # purple
}

CLASSIFICATION_LABELS = {
    "correct":    "Correct",
    "intrusion":  "Intrusion",
    "repetition": "Repetition",
    "variant":    "Variant",
    "nonword":    "Non-word",
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
        ui.value_box(
            "Non-words",
            ui.output_text("metric_nonwords"),
            theme="secondary",
        ),
        col_widths=[2, 2, 2, 2, 2, 2],
    ),
    ui.hr(),
    ui.h5("Transcript"),
    ui.output_ui("word_display"),
    ui.hr(),
    ui.h5("Words per 15-Second Bin"),
    output_widget("bin_plot"),
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
    
    @render.text
    def metric_nonwords():
        m = participant_metrics()
        return str(m.get("nonwords", "—"))

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
                    line=dict(color=colour, width=2, dash='dash'),
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



app = App(app_ui, server)

# to run enter the following command in terminal:
# shiny run app/app.py --reload