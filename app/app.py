"""Phonemic Fluency Dashboard - minimal Shiny app."""
from pathlib import Path
import pandas as pd
from shiny import App, ui, render, reactive

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


app = App(app_ui, server)

# to run enter the following command in terminal:
# shiny run app/app.py --reload