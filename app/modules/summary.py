"""Summary tab -- a cross-task "at a glance" view: demographics, a written
divergence summary, and a radar chart spanning Phonemic Fluency, Semantic
Fluency, Picture Description, a general-language axis, Finance management,
and Recent News (user request, 2026-11-12). Demographics moved here from
the Overview tab (user request, 2026-11-15) -- this is now the first thing
a clinician sees for a participant, so age/gender/diagnosis belongs here
rather than one tab over.

Deliberately continuous-percentile, not Typical/Borderline/Atypical: this
project removed categorical banding project-wide (2026-11-14, see
dementia_speech.interpretation's module docstring and
picture_description/interpretation.py) because fixed cutoffs overstated
confidence this cohort's modest individual-level discrimination doesn't
support. This tab reuses that same percentile convention rather than
reintroducing a band anywhere -- divergence language below is plain
"lower than most people their age", not a label.

Reuses each task's own HC_STATS (already fit against the TRAIN-split HC
group in phonemic_fluency.py/semantic_fluency.py/cookie_theft.py) rather
than re-fitting anything here -- single source of truth per measure.

Axes:
  - Phonemic Fluency, Semantic Fluency, Picture Description: each an
    age-adjusted percentile (0-100, low = impaired direction, per
    percentile_age_adjusted()'s convention) on an existing validated
    HEADLINE_MEASURES entry.
  - General Language (exploratory): mattr ("vocabulary variety", adjusted
    for verbosity) -- deliberately NOT total_words. total_words is
    Cookie Theft's own validated HEADLINE measure, but it's literally
    labelled "Verbal output" on that tab's own card -- an amount, not a
    language-quality construct, so it's a poor fit for an axis meant to
    represent general language ability (the original choice here).
    mattr is the closest thing in this codebase to that construct, but
    it's EXPLORATORY in Cookie Theft's own categorization (d=-0.34 HC vs
    Dementia, weaker than the 3 headline measures, demoted for that
    reason, not dropped) -- hence the honest "(exploratory)" suffix on
    the axis label, same convention as "Recent News (exploratory)".
  - Finance management: not a percentile against HC (finance.csv has no
    continuous measure to norm) -- a plain 0-100 score from
    changed_recently/change_reason, on the same low=concerning scale by
    construction (100 = no recent change; 0 = changed specifically
    because of cognitive decline). See _finance_score.
  - Recent News (exploratory): real per-participant data as of 2026-11-15,
    from news_scores.csv's `score` column (0-3, category-based -- see
    dementia_speech.tasks.news.scoring), rescaled to 0-100. Still
    EXPLORATORY, not headline: per the Tier 2 TODO and
    docs/news_scoring_rules.md, this needs a hand-rating pass (50-100
    answers) against the pipeline's own output before it's shown as more
    than descriptive -- that hasn't happened yet. None (missing, not 0)
    for 'unverifiable'/'no_date' categories, or a participant with no STN
    transcript at all -- see _news_score.

Deliberately NOT on the chart: memory complaints and who-is-concerned.
More self-reported complaints isn't reliably "worse" -- MCI often
over-reports relative to actual impairment, advanced Dementia often
under-reports (anosognosia) -- so there's no defensible single direction
to put on a "higher/lower" axis. The mismatch between the two (reports no
concern themselves, but someone else has noticed) is exactly the kind of
thing worth surfacing, but as a sentence, not a number -- see
_insight_mismatch_bullet.
"""
import pandas as pd
from scipy.stats import norm
from shiny import module, ui, render
import plotly.graph_objects as go
from shinywidgets import output_widget, render_widget

from shared import (
    pft_metrics_df, sft_metrics_df, ct_metrics_df, DEMOGRAPHICS,
    FINANCE_BY_PARTICIPANT, MEMORY_COMPLAINTS_BY_PARTICIPANT,
    WHO_CONCERNED_BY_PARTICIPANT, NEWS_BY_PARTICIPANT, GROUP_COLOURS,
)
from modules.phonemic_fluency import HC_STATS as _PFT_HC_STATS, PFT_HEADLINE_Z as _PFT_Z
from modules.semantic_fluency import HC_STATS as _SFT_HC_STATS, SFT_HEADLINE_Z as _SFT_Z
from modules.cookie_theft import HC_STATS as _CT_HC_STATS, CT_HEADLINE_Z as _CT_Z
from dementia_speech.tasks.PFT_new.interpretation import HEADLINE_MEASURES as _PFT_HEADLINE
from dementia_speech.tasks.SFT_new.interpretation import HEADLINE_MEASURES as _SFT_HEADLINE
from dementia_speech.tasks.picture_description.interpretation import (
    HEADLINE_MEASURES as _CT_HEADLINE, EXPLORATORY_MEASURES as _CT_EXPLORATORY,
)
from dementia_speech.interpretation import percentile_age_adjusted

# Same page-header/panel chrome as cookie_theft.py and phonemic_fluency.py,
# so this tab reads as part of the same app (user request, 2026-11-12) --
# duplicated rather than shared, same convention this project already uses
# for these small per-module style constants.
_PANEL_STYLE = (
    "background:white;border:1px solid #e1e8ef;border-top:4px solid #527d96;"
    "border-radius:12px;padding:22px;min-width:0;"
    "box-shadow:0 2px 8px rgba(35,55,75,0.05);"
)
_SECTION_HEADER_STYLE = "font-weight:700;color:#243447;margin-bottom:4px;"
_SECTION_CAPTION_STYLE = "color:#6c757d;font-size:13px;margin-bottom:16px;"


def _measure_by_key(measures, key):
    return next(m for m in measures if m.key == key)


_PFT_C = _measure_by_key(_PFT_HEADLINE, "C")
_SFT_C = _measure_by_key(_SFT_HEADLINE, "C")
_CT_IU_SCORE = _measure_by_key(_CT_HEADLINE, "iu_score")
# mattr ("vocabulary variety", adjusted for how much the participant
# talked) is a genuine language-QUALITY construct, unlike total_words
# (Cookie Theft's own card literally labels that one "Verbal output" --
# it's an amount, not a quality). mattr is EXPLORATORY in Cookie Theft's
# own categorization, not HEADLINE (d=-0.34 HC vs Dementia, weaker than
# the 3 headline measures, demoted for that reason, not dropped) -- so
# this axis is honestly labelled "(exploratory)" below, same convention as
# "Recent News (exploratory)".
_CT_MATTR = _measure_by_key(_CT_EXPLORATORY, "mattr")

# news_scores.csv's `score` column is 0-3 (SCORE_BY_CATEGORY in
# dementia_speech.tasks.news.scoring) -- rescaled onto the same 0-100 scale
# as every other axis here.
_NEWS_MAX_SCORE = 3

# changed_recently == "yes" + this change_reason -> this score. "unclear"
# (missing from this dict) scores None (missing, not a guess) -- same
# "unsure becomes missing, never zero/invented" convention the news task
# itself uses. 0 = the one reason that's a recognised functional-decline
# marker (GPCOG's own finances item is scored this way); the other two
# reasons aren't cognitively driven, so they're closer to the "no change"
# end without being identical to it.
_FINANCE_CHANGE_REASON_SCORE = {
    "cognitive_decline": 0.0,
    "physical_health": 50.0,
    "convenience_or_preference": 75.0,
    "other": 75.0,
}

# Below this age-adjusted percentile, a quantitative domain is called out
# in its own bullet rather than folded into the "typical range" line. A
# round, descriptive choice (bottom quartile) -- not a validated clinical
# cut-off, consistent with this project having deliberately dropped fixed
# cutoffs elsewhere; see module docstring.
_DIVERGENCE_PERCENTILE = 25


def _participant_age(participant_id):
    return DEMOGRAPHICS.get(str(participant_id), {}).get("age")


def _percentile_for(metrics_df, hc_stats, measure, participant_id, age):
    rows = metrics_df[metrics_df["participant_id"].astype(str) == str(participant_id)]
    if rows.empty or age is None or pd.isna(age):
        return None
    value = rows.iloc[0].get(measure.key)
    if value is None or pd.isna(value):
        return None
    intercept, slope, residual_sd = hc_stats.get(measure.key, (None, None, None))
    if intercept is None or slope is None or not residual_sd:
        return None
    return percentile_age_adjusted(value, age, intercept, slope, residual_sd, measure.higher_is_worse)


def _finance_score(participant_id):
    row = FINANCE_BY_PARTICIPANT.get(str(participant_id))
    if not row or row.get("status") != "extracted":
        return None
    changed = row.get("changed_recently")
    if changed == "no":
        return 100.0
    if changed == "yes":
        return _FINANCE_CHANGE_REASON_SCORE.get(row.get("change_reason"))
    return None  # "unclear"


def _news_score(participant_id):
    row = NEWS_BY_PARTICIPANT.get(str(participant_id))
    if not row:
        return None
    score = row.get("score")
    if score is None or pd.isna(score):
        return None  # unverifiable/no_date -- a gap in the measure, not a 0
    return float(score) / _NEWS_MAX_SCORE * 100


def _domain_scores(participant_id):
    """{axis_label: (percentile_or_score, is_placeholder)} for all six axes,
    in the fixed order the radar chart and bullets both use."""
    age = _participant_age(participant_id)
    return {
        "Phonemic Fluency": (_percentile_for(pft_metrics_df, _PFT_HC_STATS, _PFT_C, participant_id, age), False),
        "Semantic Fluency": (_percentile_for(sft_metrics_df, _SFT_HC_STATS, _SFT_C, participant_id, age), False),
        "Picture Description": (_percentile_for(ct_metrics_df, _CT_HC_STATS, _CT_IU_SCORE, participant_id, age), False),
        "General Language (exploratory)": (_percentile_for(ct_metrics_df, _CT_HC_STATS, _CT_MATTR, participant_id, age), False),
        "Finance Management": (_finance_score(participant_id), False),
        "Recent News (exploratory)": (_news_score(participant_id), False),
    }


_GROUPS = ["HC", "MCI", "Dementia"]


def _group_mean_percentiles(z_df: pd.DataFrame, measure_key: str) -> dict:
    """{group: mean age-adjusted percentile} for one measure, from its
    already-computed z-oriented column in z_df (the same PFT_HEADLINE_Z/
    SFT_HEADLINE_Z/CT_HEADLINE_Z every task tab already builds for its own
    group-comparison graphs -- see e.g. cookie_theft.py's
    _card_distribution_figure). Each participant's own z is converted to a
    percentile first (norm.cdf), then averaged within each group -- the
    group mean of exactly the same number shown for the individual
    participant, not a mean of raw z-scores re-converted afterwards."""
    if measure_key not in z_df.columns:
        return {g: None for g in _GROUPS}
    percentiles = z_df[measure_key].apply(lambda z: float(norm.cdf(z)) * 100 if pd.notna(z) else None)
    means = percentiles.groupby(z_df["diagnosis"]).mean()
    return {g: (float(means[g]) if g in means.index and pd.notna(means[g]) else None) for g in _GROUPS}


def _mean_by_group(score_fn) -> dict:
    """{group: mean score}, from every participant score_fn resolves to a
    real value for -- missing/unscoreable participants (score_fn returns
    None) are excluded from the average entirely, not counted as 0. Shared
    between Finance and Recent News, the two axes with no HC-norm percentile
    to average via _group_mean_percentiles."""
    rows = []
    for pid in DEMOGRAPHICS:
        score = score_fn(pid)
        if score is not None:
            rows.append({"diagnosis": DEMOGRAPHICS[pid].get("diagnosis"), "score": score})
    if not rows:
        return {g: None for g in _GROUPS}
    means = pd.DataFrame(rows).groupby("diagnosis")["score"].mean()
    return {g: (float(means[g]) if g in means.index and pd.notna(means[g]) else None) for g in _GROUPS}


# Computed once at import time, same convention as HC_STATS elsewhere --
# doesn't depend on which participant is selected.
_DOMAIN_GROUP_MEANS = {
    "Phonemic Fluency": _group_mean_percentiles(_PFT_Z, _PFT_C.key),
    "Semantic Fluency": _group_mean_percentiles(_SFT_Z, _SFT_C.key),
    "Picture Description": _group_mean_percentiles(_CT_Z, _CT_IU_SCORE.key),
    "General Language (exploratory)": _group_mean_percentiles(_CT_Z, _CT_MATTR.key),
    "Finance Management": _mean_by_group(_finance_score),
    "Recent News (exploratory)": _mean_by_group(_news_score),
}


def _insight_mismatch_bullet(participant_id):
    """A self-report vs. informant-report mismatch is a narrative finding,
    not a number (see module docstring) -- only flagged in one direction
    (no self-reported complaint, but someone else is concerned), since
    that's the clinically notable gap; the reverse (self-concerned, no one
    else mentioned) isn't a mismatch worth flagging on its own."""
    findings = MEMORY_COMPLAINTS_BY_PARTICIPANT.get(str(participant_id))
    who_row = WHO_CONCERNED_BY_PARTICIPANT.get(str(participant_id))
    if not findings or not who_row:
        return None
    complaint_status = findings[0].get("status")
    who_concerned = who_row.get("who_concerned")
    if complaint_status == "no_complaint" and who_concerned in ("other", "both"):
        return (
            "Does not report memory concerns themselves, but the transcript "
            "indicates someone else (or both) is concerned -- worth probing "
            "the gap between self- and informant-report directly."
        )
    return None


def _news_status_bullet(participant_id):
    """Recent News always gets a caveat, even when it scored fine (see
    module docstring -- exploratory, not yet checked for a real group
    difference or hand-rated) -- plus, when applicable, the specific reason
    this participant's own answer couldn't be scored at all (score is None
    for 'unverifiable'/'no_date', or if there's no STN row for them)."""
    row = NEWS_BY_PARTICIPANT.get(str(participant_id))
    category = row.get("category") if row else "no_date"
    reason = {
        "no_date": "no assessment date is on record for this participant yet",
        "unverifiable": "the answer was specific but didn't match any cached news event",
    }.get(category)
    base = (
        "Recent News is exploratory -- not yet checked for a real group "
        "difference, shown for reference only."
    )
    return f"Recent-news recall could not be scored ({reason}). {base}" if reason else base


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def generate_bullets(participant_id) -> list[str]:
    scores = _domain_scores(participant_id)
    divergent, typical = [], []
    for label, (value, is_placeholder) in scores.items():
        if is_placeholder or value is None:
            continue
        if value < _DIVERGENCE_PERCENTILE:
            divergent.append(f"{label} was lower than most people their age ({_ordinal(round(value))} percentile).")
        else:
            typical.append(label)

    bullets = list(divergent)
    if typical:
        bullets.append("Within the typical range for their age on: " + ", ".join(typical) + ".")

    mismatch = _insight_mismatch_bullet(participant_id)
    if mismatch:
        bullets.append(mismatch)

    bullets.append(_news_status_bullet(participant_id))

    if not divergent and not mismatch:
        bullets.insert(0, "No notable divergence from same-age peers found across the measures below.")

    return bullets


_RADAR_AXES = [
    "Phonemic Fluency", "Semantic Fluency", "Picture Description",
    "General Language (exploratory)", "Finance Management", "Recent News (exploratory)",
]


def _radar_figure(participant_id):
    scores = _domain_scores(participant_id)
    values = [scores[axis][0] for axis in _RADAR_AXES]
    # Plotly's polar trace needs the loop closed back to the first point.
    # A missing value stays None (not coerced to 0): 0 is the worst
    # possible position on every one of these axes, so silently plotting
    # a gap there as "0" would misrepresent missing data as a severe
    # finding -- same "missing is never zero" principle as the news task's
    # own scoring. Plotly breaks the line at a None point rather than
    # drawing to it.
    theta = _RADAR_AXES + [_RADAR_AXES[0]]
    r = values + [values[0]]

    fig = go.Figure()
    # Group means first, so the participant's own (filled, bold) trace
    # draws on top rather than being obscured by them.
    for group in _GROUPS:
        group_values = [_DOMAIN_GROUP_MEANS[axis].get(group) for axis in _RADAR_AXES]
        group_r = group_values + [group_values[0]]
        fig.add_trace(go.Scatterpolar(
            r=group_r, theta=theta, mode="lines", name=f"{group} mean",
            line=dict(color=GROUP_COLOURS.get(group, "#999"), width=1.5, dash="dot"),
        ))
    fig.add_trace(go.Scatterpolar(
        r=r, theta=theta, mode="lines+markers", name="This participant",
        line=dict(color="#1565C0", width=3),
        marker=dict(size=7),
        fill="toself", fillcolor="rgba(21,101,192,0.15)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], ticksuffix="", showticklabels=True, tickfont=dict(size=9))),
        showlegend=True, legend=dict(orientation="h", y=-0.1),
        margin=dict(t=30, b=30, l=40, r=40), height=420,
    )
    return fig


@module.ui
def summary_ui():
    return ui.div(
        # PAGE HEADER -- same chrome as Cookie Theft / Fluency tabs.
        ui.div(
            ui.div(
                ui.h3("Summary", style="font-weight:700;color:#243447;margin:0;"),
                ui.p(
                    "Cross-task overview: where this participant diverges from "
                    "same-age peers, and where they don't.",
                    style="color:#6c757d;margin:4px 0 0 0;",
                ),
            ),
            ui.div(
                ui.span(
                    "CLINICAL ASSESSMENT",
                    style=(
                        "font-size:11px;font-weight:700;letter-spacing:1px;"
                        "color:#426b83;background:#eaf2f6;"
                        "padding:7px 12px;border-radius:20px;"
                    ),
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

        ui.div(
            ui.h5("Demographics", style=_SECTION_HEADER_STYLE),
            ui.output_ui("demographics_card"),
            style="margin-bottom:26px;",
        ),

        ui.layout_columns(
            ui.div(
                ui.h5("Key findings", style=_SECTION_HEADER_STYLE),
                ui.p(
                    "Plain-language read of the measures on the right -- no "
                    "Typical/Borderline/Atypical label, just how this "
                    "participant compares with same-age healthy controls.",
                    style=_SECTION_CAPTION_STYLE,
                ),
                ui.output_ui("bullets_panel"),
                style=_PANEL_STYLE,
            ),
            ui.div(
                ui.h5("Domains at a glance", style=_SECTION_HEADER_STYLE),
                ui.p(
                    "Each axis is an age-adjusted percentile (low = further "
                    "from the healthy-control reference, regardless of "
                    "direction) except Finance Management and Recent News, "
                    "which are 0-100 scores from reported change/reason and "
                    "news-recall accuracy respectively, not HC percentiles. "
                    "General Language and Recent News are both "
                    "exploratory -- weaker, not-yet-headline signals shown "
                    "for reference, not validated findings. Dotted lines "
                    "show each diagnostic group's own mean.",
                    style=_SECTION_CAPTION_STYLE,
                ),
                output_widget("radar_chart"),
                style=_PANEL_STYLE,
            ),
            col_widths=[5, 7],
        ),

        style="background:#f5f7fa;padding:22px;border-radius:14px;",
    )


@module.server
def summary_server(input, output, session, participant_id, show_diagnosis=True):
    """show_diagnosis=False hides the Diagnosis field in demographics_card --
    used by study_app.py (blinded rater study, 2026-11-19): a rater guessing
    the diagnosis can't be shown it on the same page. Nothing else on this
    tab leaks the participant's own diagnosis -- every other "diagnosis"
    reference here (_group_mean_percentiles, _mean_by_group) is a group-level
    aggregate shown identically regardless of which participant is selected,
    not this participant's own label."""

    @render.ui
    def demographics_card():
        demo = DEMOGRAPHICS.get(str(participant_id()), {})
        age = demo.get("age")
        gender = demo.get("gender")

        def field(label, value):
            return ui.div(
                ui.div(label, style="color:#666;font-size:0.82em;"),
                ui.div(
                    str(value) if pd.notna(value) else "—",
                    style="font-size:1.4em;font-weight:700;",
                ),
            )

        fields = [field("Age", age), field("Gender", gender)]
        if show_diagnosis:
            fields.append(field("Diagnosis", demo.get("diagnosis")))

        return ui.div(
            *fields,
            style=_PANEL_STYLE + "display:flex;gap:32px;",
        )

    @render.ui
    def bullets_panel():
        bullets = generate_bullets(participant_id())
        return ui.tags.ul(
            *[ui.tags.li(b, style="margin-bottom:8px;") for b in bullets],
            style="padding-left:20px;margin:0;",
        )

    @render_widget
    def radar_chart():
        return _radar_figure(participant_id())
