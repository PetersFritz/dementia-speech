"""Independent validation analysis: which Cookie Theft measures are worth
showing on the dashboard, and in which direction.

Deliberately separate from the Shiny app and from
dementia_speech.interpretation.fit_age_adjusted_norms() -- the point is an
external check of the age-adjustment approach, not a restatement of code the
dashboard already runs, so this reimplements the age-adjusted z-score from
scratch with statsmodels rather than importing the app's own helper.

Pipeline:
  1. Load pipelines/scratch/export_cookie_theft_measures.py's CSV export and
     attach each participant's age from shared.DEMOGRAPHICS.
  2. Fit `measure ~ age` with OLS on TRAIN-split HC participants only.
  3. Apply that model to every participant (every split, every diagnosis)
     to get an age-adjusted z-score: (observed - predicted) / HC residual SD.
  4. Orient the sign (negative = poorer performance) wherever a direction is
     already established in interpretation.py; plot every measure by
     diagnosis, split into TRAIN/TEST panels, either way.
  5. Print a median/IQR table per measure x split x diagnosis.

Caveats to read the output with:
  - TRAIN HC participants are both the reference sample AND one of the
    groups being plotted, so their residuals are optimistically well-
    centred by construction. TEST HC is the real check of whether the
    age model generalises -- weight that panel more than TRAIN's.
  - This assumes a linear age relationship and roughly constant residual
    spread across age. Not checked here -- eyeball the age-vs-measure
    scatter for any candidate measure before trusting its z-scores.
  - A measure with no established direction (`None` below) is still
    plotted/summarised on its raw (non-oriented) z-score, flagged as such --
    seeing which group actually scores lower is itself evidence for
    deciding a direction, not just a prerequisite for plotting one.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "app"))

from shared import DEMOGRAPHICS, GROUP_COLOURS  # noqa: E402  (needs the sys.path insert above)
from dementia_speech.paths import DATA_DIR  # noqa: E402
from dementia_speech.tasks.picture_description.interpretation import (  # noqa: E402
    HEADLINE_MEASURES, EXPLORATORY_MEASURES,
)

EXPORT_PATH = DATA_DIR / "processed" / "cookie_theft_measures_export.csv"
SCORES_OUTPUT_PATH = DATA_DIR / "processed" / "cookie_theft_age_adjusted_scores.csv"
PLOTS_DIR = DATA_DIR / "processed" / "cookie_theft_measure_plots"

MIN_TRAIN_HC = 20

# Every measure currently computed by score_cookie_theft.py that isn't a
# raw position/count the project's own docs already call purely descriptive
# (avg_x/avg_y/stdev_x/stdev_y/self_cycles/nodes/unique_nodes/
# self_cycles_quadrants -- see docs/cookie_theft_scoring.md). Includes both
# the 3 current headline measures and everything in the exploratory list,
# plus a few raw/dropped ones worth re-checking independently here.
CANDIDATE_MEASURES = [
    "iu_score",
    "total_words",
    "semantic_relevance",
    "iu_mention_density",
    "iu_mentions_per_100_words",
    "lexical_density",
    "pronoun_ratio",
    "mattr",
    "lexical_diversity",
    "indefinite_term_rate",
    "cycles_per_100_words",
    "path_distance_per_100_words",
    "total_path_distance",
    "total_path_per_unique_nodes",
    "total_path_per_node",
    "cross_ratio_quadrants",
    "spatial_left_bias",
]

# True = higher raw score means poorer performance, False = higher means
# better, None = direction not (yet) established -- taken directly from
# interpretation.py wherever a measure already appears there (headline or
# exploratory), so this stays in sync with the dashboard's own assumptions
# rather than duplicating a second, possibly-drifting copy of the same
# judgement calls. Anything not currently in either list (lexical_density,
# pronoun_ratio, lexical_diversity, iu_mention_density, total_path_distance)
# has no established direction yet -- that's exactly what this analysis is
# for deciding, not something to assume up front.
_ESTABLISHED_DIRECTIONS = {
    m.key: m.higher_is_worse for m in HEADLINE_MEASURES + EXPLORATORY_MEASURES
}
HIGHER_IS_WORSE = {m: _ESTABLISHED_DIRECTIONS.get(m) for m in CANDIDATE_MEASURES}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(EXPORT_PATH)
    df["age"] = df["participant_id"].astype(str).map(
        lambda pid: DEMOGRAPHICS.get(pid, {}).get("age")
    )
    print(df.groupby(["split", "diagnosis"]).size())
    print(f"{df['age'].isna().sum()} participants missing age\n")
    return df


def calculate_age_adjusted_z(df: pd.DataFrame, measure: str, min_hc: int = MIN_TRAIN_HC):
    """Fit `measure ~ age` on TRAIN-split HC participants only, then apply
    that model to every participant. Returns (df_with_new_columns, model):
      <measure>_expected   -- value predicted for that participant's own age
      <measure>_adjusted_z -- (observed - expected) / HC residual SD

    model is the fitted statsmodels OLS result (HC-train only) -- its
    .params / .rsquared are printed by the caller as fit diagnostics.
    """
    hc = df[(df["diagnosis"] == "HC") & (df["split"] == "TRAIN")][[measure, "age"]].dropna()
    if len(hc) < min_hc:
        raise ValueError(f"Too few training HC participants for {measure}: {len(hc)}")

    X_hc = sm.add_constant(hc["age"])
    model = sm.OLS(hc[measure], X_hc).fit()

    residual_sd = float(np.sqrt(model.mse_resid))
    if not np.isfinite(residual_sd) or residual_sd <= 0:
        raise ValueError(f"Invalid residual SD for {measure}")

    result = df.copy()
    valid = result[["age", measure]].notna().all(axis=1)

    expected = pd.Series(np.nan, index=result.index)
    X_all = sm.add_constant(result.loc[valid, "age"], has_constant="add")
    expected.loc[valid] = model.predict(X_all)

    result[f"{measure}_expected"] = expected
    result[f"{measure}_adjusted_z"] = (result[measure] - expected) / residual_sd
    return result, model


def fit_all_measures(df: pd.DataFrame):
    """Fit every candidate measure, skipping (with a warning, not a crash)
    any that don't have enough TRAIN HC data or a usable residual SD.
    """
    adjusted_df = df.copy()
    models = {}
    for measure in CANDIDATE_MEASURES:
        try:
            adjusted_df, model = calculate_age_adjusted_z(adjusted_df, measure)
        except ValueError as e:
            print(f"  SKIPPED {measure}: {e}")
            continue
        models[measure] = model
        print(
            f"  {measure}: age coefficient={model.params['age']:+.4f}, "
            f"R-squared={model.rsquared:.3f}, n(TRAIN HC)={int(model.nobs)}"
        )

    for measure, direction in HIGHER_IS_WORSE.items():
        z_col = f"{measure}_adjusted_z"
        if z_col not in adjusted_df.columns:
            continue
        if direction is None:
            continue  # no oriented_z -- plotted/summarised on the raw z instead
        adjusted_df[f"{measure}_oriented_z"] = (
            -adjusted_df[z_col] if direction else adjusted_df[z_col]
        )
    return adjusted_df, models


def _z_column_and_label(measure: str, data: pd.DataFrame):
    """Prefer the oriented z-score; fall back to the raw (unoriented) one
    with a label flagging that this measure's direction isn't established
    yet -- still worth plotting, since the pattern itself is evidence for
    which direction to assign.
    """
    oriented_col = f"{measure}_oriented_z"
    if oriented_col in data.columns:
        return oriented_col, "Age-adjusted z-score (negative = poorer performance)"
    raw_col = f"{measure}_adjusted_z"
    if raw_col in data.columns:
        return raw_col, "Age-adjusted z-score (direction not yet established)"
    return None, None


def plot_measure(measure: str, data: pd.DataFrame, output_dir: Path):
    z_col, y_label = _z_column_and_label(measure, data)
    if z_col is None:
        print(f"No adjusted z-score available for {measure}")
        return None

    plot_data = data.dropna(subset=[z_col]).copy()
    fig = px.box(
        plot_data,
        x="diagnosis",
        y=z_col,
        color="diagnosis",
        category_orders={"diagnosis": ["HC", "MCI", "Dementia"]},
        color_discrete_map=GROUP_COLOURS,
        points="all",
        facet_col="split",
        title=f"{measure}: age-adjusted scores by diagnosis",
        labels={z_col: y_label, "diagnosis": "Diagnostic group", "split": "Dataset split"},
    )
    fig.add_hline(y=0, line_dash="solid", line_color="#999", line_width=1)
    fig.add_hline(
        y=-1.645, line_dash="dash", line_color="#c0392b", line_width=1,
        annotation_text="-1.645 (conventional lower-tail reference, not a cutoff)",
        annotation_position="bottom right",
    )
    fig.update_layout(showlegend=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{measure}_age_adjusted.html"
    fig.write_html(out_path)
    print(f"  wrote {out_path}")
    return fig


def summarise_measure(measure: str, data: pd.DataFrame):
    z_col, _ = _z_column_and_label(measure, data)
    if z_col is None:
        return None
    return (
        data.groupby(["split", "diagnosis"])[z_col]
        .agg(
            n="count",
            median="median",
            mean="mean",
            sd="std",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .round(2)
    )


def main():
    print("=== Step 1: load data ===")
    df = load_data()

    print("=== Step 2+3: fit age-adjusted HC reference models ===")
    adjusted_df, models = fit_all_measures(df)

    print("\n=== Step 4: plots (one HTML file per measure in", PLOTS_DIR, ") ===")
    for measure in CANDIDATE_MEASURES:
        plot_measure(measure, adjusted_df, PLOTS_DIR)

    print("\n=== Step 5: summary tables ===")
    for measure in CANDIDATE_MEASURES:
        summary = summarise_measure(measure, adjusted_df)
        if summary is not None:
            print(f"\n--- {measure} ---")
            print(summary)

    adjusted_df.to_csv(SCORES_OUTPUT_PATH, index=False)
    print(f"\nWrote full adjusted scores to {SCORES_OUTPUT_PATH}")

    print(
        "\nReminder: TRAIN HC rows were used to FIT each model, so their own "
        "residuals are optimistically well-centred -- judge each measure "
        "mainly on the TEST panel of its plot/table, which is an unseen check."
    )


if __name__ == "__main__":
    main()
