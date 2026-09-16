# ============================================================
# pages/6_AI_Risk_Prediction.py
# NER-SMART — AI DISRUPTION RISK INTELLIGENCE
# ============================================================

import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (
    init_state,
    train_risk_model,
    predict_risk,
    explain_risk,
    ROADS,
    DISTRICTS,
    build_live_graph,
    edges_dataframe,
    risk_to_status,
)


# ============================================================
# INITIALIZATION
# ============================================================

init_state()

st.set_page_config(
    page_title="NER-SMART AI Risk",
    page_icon="🤖",
    layout="wide",
)

model = train_risk_model()


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.risk-card {
    padding: 22px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,.25);
    background: rgba(128,128,128,.06);
    margin-bottom: 12px;
}

.big-risk {
    font-size: 48px;
    font-weight: 800;
}

.small-label {
    font-size: 13px;
    opacity: .70;
}

.warning-box {
    padding: 18px;
    border-radius: 14px;
    margin: 10px 0;
}

.metric-card {
    padding: 16px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,.20);
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def risk_category(risk):

    risk = float(risk)

    if risk >= 80:
        return "CRITICAL"

    if risk >= 60:
        return "HIGH"

    if risk >= 30:
        return "MEDIUM"

    return "LOW"


def risk_emoji(risk):

    risk = float(risk)

    if risk >= 80:
        return "🔴"

    if risk >= 60:
        return "🟠"

    if risk >= 30:
        return "🟡"

    return "🟢"


def risk_color(risk):

    risk = float(risk)

    if risk >= 80:
        return "#dc2626"

    if risk >= 60:
        return "#f97316"

    if risk >= 30:
        return "#eab308"

    return "#16a34a"


def weather_severity(rainfall, wind):

    score = 0

    if rainfall >= 100:
        score += 60

    elif rainfall >= 50:
        score += 40

    elif rainfall >= 20:
        score += 20

    elif rainfall >= 5:
        score += 10

    if wind >= 60:
        score += 40

    elif wind >= 40:
        score += 25

    elif wind >= 25:
        score += 10

    return min(score, 100)


def confidence_indicator(
    rainfall,
    slope,
    historical,
    condition,
    traffic,
):

    """
    Prototype input-range indicator.

    This is NOT calibrated statistical confidence.
    """

    penalty = 0

    if rainfall > 200:
        penalty += 10

    if slope > 55:
        penalty += 10

    if historical > 10:
        penalty += 10

    if condition > 0.90:
        penalty += 8

    if traffic > 0.95:
        penalty += 8

    return max(
        60,
        min(
            97,
            92 - penalty,
        ),
    )


def operational_action(risk):

    if risk >= 80:

        return (
            "🚨 CRITICAL ACTION",
            "Consider stopping or rerouting non-essential logistics. "
            "Trigger emergency corridor verification and field confirmation.",
        )

    if risk >= 60:

        return (
            "⚠️ HIGH-RISK ACTION",
            "Use a risk-aware alternate route, increase monitoring frequency "
            "and notify fleet operators.",
        )

    if risk >= 30:

        return (
            "🟡 CAUTION",
            "Continue operations with increased monitoring and prepare an "
            "alternate route.",
        )

    return (
        "🟢 NORMAL OPERATIONS",
        "Continue normal logistics operations while monitoring forecast changes.",
    )


def make_gauge(risk):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(risk),
            number={
                "suffix": "%",
                "font": {
                    "size": 40,
                },
            },
            title={
                "text": "Disruption Risk",
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                },
                "steps": [
                    {
                        "range": [0, 30],
                    },
                    {
                        "range": [30, 60],
                    },
                    {
                        "range": [60, 80],
                    },
                    {
                        "range": [80, 100],
                    },
                ],
                "threshold": {
                    "line": {
                        "width": 5,
                    },
                    "value": float(risk),
                },
            },
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig


def future_risk(
    rainfall,
    slope,
    historical,
    condition,
    traffic,
    hour,
):

    if hour <= 3:
        multiplier = 1.00

    elif hour <= 6:
        multiplier = 1.05

    elif hour <= 12:
        multiplier = 1.10

    elif hour <= 24:
        multiplier = 1.15

    else:
        multiplier = 1.20

    projected_rainfall = (
        rainfall * multiplier
    )

    return predict_risk(
        model,
        min(
            projected_rainfall,
            250,
        ),
        slope,
        historical,
        condition,
        traffic,
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🤖 AI Disruption Risk Intelligence"
)

st.caption(
    "Multi-factor AI risk assessment, explainable risk factors, "
    "early-warning intelligence and network-wide corridor monitoring."
)


# ============================================================
# LIVE NETWORK STATUS
# ============================================================

st.subheader(
    "🌐 Live Network Risk Overview"
)

G = build_live_graph()

network_df = edges_dataframe(G)

if network_df.empty:

    st.warning(
        "Live road intelligence is currently unavailable."
    )

else:

    total_roads = len(
        network_df
    )

    blocked = int(
        (
            network_df["status"]
            == "BLOCKED"
        ).sum()
    )

    high_risk = int(
        (
            network_df["status"]
            == "HIGH RISK"
        ).sum()
    )

    partial = int(
        (
            network_df["status"]
            == "PARTIALLY DISRUPTED"
        ).sum()
    )

    normal = int(
        (
            network_df["status"]
            == "NORMAL"
        ).sum()
    )

    accessible = round(
        (
            total_roads - blocked
        )
        / total_roads
        * 100,
        1,
    )

    n1, n2, n3, n4, n5 = st.columns(5)

    n1.metric(
        "Network Accessibility",
        f"{accessible}%",
    )

    n2.metric(
        "Blocked",
        blocked,
    )

    n3.metric(
        "High Risk",
        high_risk,
    )

    n4.metric(
        "Partially Disrupted",
        partial,
    )

    n5.metric(
        "Normal",
        normal,
    )


# ============================================================
# CORRIDOR SELECTION
# ============================================================

st.divider()

st.subheader(
    "🛣️ Corridor Intelligence"
)

road_options = {
    (
        f"{road['name']} | "
        f"{road['a']} → {road['b']}"
    ): road
    for road in ROADS
}

selected_name = st.selectbox(
    "Select a logistics corridor",
    list(
        road_options.keys()
    ),
)

road = road_options[
    selected_name
]


# ============================================================
# LIVE CORRIDOR DATA
# ============================================================

live_edge = None

if G.has_edge(
    road["a"],
    road["b"],
):

    live_edge = G.edges[
        road["a"],
        road["b"]
    ]

elif G.has_edge(
    road["b"],
    road["a"],
):

    live_edge = G.edges[
        road["b"],
        road["a"]
    ]


if live_edge:

    live_risk = float(
        live_edge.get(
            "risk_pct",
            0,
        )
    )

    live_status = live_edge.get(
        "status",
        risk_to_status(
            live_risk
        ),
    )

else:

    live_risk = road[
        "base_risk"
    ] * 100

    live_status = risk_to_status(
        live_risk
    )


c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Corridor",
    road["name"],
)

c2.metric(
    "Distance",
    f"{road['dist']} km",
)

c3.metric(
    "Terrain",
    road["road_type"].upper(),
)

c4.metric(
    "AI Risk",
    f"{live_risk:.0f}%",
)

c5.metric(
    "Status",
    live_status,
)


# ============================================================
# ENVIRONMENTAL INPUTS
# ============================================================

st.divider()

st.subheader(
    "🌧️ Environmental & Operational Inputs"
)

left, right = st.columns(2)

with left:

    rainfall = st.slider(
        "🌧️ Rainfall — next 24 hours (mm)",
        min_value=0,
        max_value=250,
        value=40,
        step=5,
    )

    slope = st.slider(
        "⛰️ Terrain slope (degrees)",
        min_value=0,
        max_value=60,
        value=int(
            road["slope"]
        ),
        step=1,
    )

    historical = st.slider(
        "📊 Historical / recent incidents",
        min_value=0,
        max_value=15,
        value=2,
        step=1,
    )

    river_exposure = st.slider(
        "🌊 Flood / river exposure",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05,
    )


with right:

    road_condition = st.slider(
        "🛣️ Road condition degradation",
        min_value=0.0,
        max_value=1.0,
        value=round(
            road["base_risk"],
            2,
        ),
        step=0.05,
        help="0 = good condition, 1 = severely degraded",
    )

    traffic = st.slider(
        "🚚 Traffic level",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.05,
    )

    wind = st.slider(
        "💨 Wind speed (km/h)",
        min_value=0,
        max_value=120,
        value=20,
        step=5,
    )

    visibility = st.slider(
        "👁️ Visibility degradation",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.05,
    )


# ============================================================
# AI RISK ENGINE
# ============================================================

ml_risk = predict_risk(
    model,
    rainfall,
    slope,
    historical,
    road_condition,
    traffic,
)


weather_risk = weather_severity(
    rainfall,
    wind,
)


flood_penalty = (
    river_exposure * 12
)


visibility_penalty = (
    visibility * 8
)


advanced_risk = (
    ml_risk * 0.78
    +
    weather_risk * 0.10
    +
    flood_penalty
    +
    visibility_penalty
)


advanced_risk = max(
    0,
    min(
        100,
        advanced_risk,
    ),
)


category = risk_category(
    advanced_risk
)


confidence = confidence_indicator(
    rainfall,
    slope,
    historical,
    road_condition,
    traffic,
)


# ============================================================
# RISK RESULT
# ============================================================

st.divider()

st.subheader(
    "🎯 AI Risk Assessment"
)

a, b = st.columns(
    [1, 1]
)


with a:

    st.plotly_chart(
        make_gauge(
            advanced_risk
        ),
        use_container_width=True,
    )


with b:

    st.markdown(
        f"""
<div class="risk-card">

<div class="small-label">
CURRENT PREDICTED RISK
</div>

<div class="big-risk">
{risk_emoji(advanced_risk)}
{advanced_risk:.0f}%
</div>

<h2>{category}</h2>

<div class="small-label">
Input-range confidence indicator
</div>

<h3>{confidence}%</h3>

</div>
""",
        unsafe_allow_html=True,
    )

    st.progress(
        min(
            1.0,
            advanced_risk / 100,
        )
    )


# ============================================================
# ACTION
# ============================================================

action_title, action_description = (
    operational_action(
        advanced_risk
    )
)

if advanced_risk >= 80:

    box_color = "rgba(220,38,38,.12)"

elif advanced_risk >= 60:

    box_color = "rgba(249,115,22,.12)"

elif advanced_risk >= 30:

    box_color = "rgba(234,179,8,.12)"

else:

    box_color = "rgba(22,163,74,.12)"


st.markdown(
    f"""
<div class="warning-box"
style="background:{box_color};">

<h3>{action_title}</h3>

{action_description}

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# RISK FACTOR EXPLANATION
# ============================================================

st.divider()

st.subheader(
    "🔍 Explainable AI — Why Is This Corridor At Risk?"
)

contributions = explain_risk(
    rainfall,
    slope,
    historical,
    road_condition,
    traffic,
)


contributions[
    "Flood exposure"
] = flood_penalty


contributions[
    "Visibility"
] = visibility_penalty


contributions[
    "Wind / weather"
] = (
    weather_risk * 0.10
)


contrib_df = pd.DataFrame(
    {
        "Factor": list(
            contributions.keys()
        ),
        "Contribution": list(
            contributions.values()
        ),
    }
)


contrib_df = (
    contrib_df
    .sort_values(
        "Contribution",
        ascending=True,
    )
)


fig = px.bar(
    contrib_df,
    x="Contribution",
    y="Factor",
    orientation="h",
    color="Contribution",
    color_continuous_scale="OrRd",
)


fig.update_layout(
    height=430,
    showlegend=False,
    coloraxis_showscale=False,
    xaxis_title="Relative contribution",
    yaxis_title="",
)


st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# INPUT SUMMARY
# ============================================================

with st.expander(
    "📋 View AI Input Data"
):

    input_df = pd.DataFrame(
        {
            "Feature": [
                "Rainfall",
                "Terrain slope",
                "Historical incidents",
                "Road condition",
                "Traffic",
                "Flood exposure",
                "Wind",
                "Visibility degradation",
            ],
            "Value": [
                f"{rainfall} mm",
                f"{slope}°",
                historical,
                f"{road_condition:.2f}",
                f"{traffic:.2f}",
                f"{river_exposure:.2f}",
                f"{wind} km/h",
                f"{visibility:.2f}",
            ],
        }
    )

    st.dataframe(
        input_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FORECAST
# ============================================================

st.divider()

st.subheader(
    "🔮 Future Disruption Risk"
)

forecast_hours = [
    0,
    3,
    6,
    12,
    24,
    48,
]

forecast_rows = []

for hour in forecast_hours:

    if hour == 0:

        risk = advanced_risk

    else:

        risk = future_risk(
            rainfall,
            slope,
            historical,
            road_condition,
            traffic,
            hour,
        )

        risk = min(
            100,
            risk
            + flood_penalty
            + visibility_penalty,
        )

    forecast_rows.append(
        {
            "Hour": hour,
            "Time": (
                "NOW"
                if hour == 0
                else f"+{hour}h"
            ),
            "Risk": round(
                risk,
                1,
            ),
            "Level": risk_category(
                risk
            ),
        }
    )


forecast_df = pd.DataFrame(
    forecast_rows
)


fig = px.line(
    forecast_df,
    x="Time",
    y="Risk",
    markers=True,
    text="Risk",
)


fig.update_traces(
    texttemplate="%{text}%",
    textposition="top center",
)


fig.update_layout(
    height=380,
    yaxis_title="Disruption Risk %",
    xaxis_title="Forecast Horizon",
    yaxis_range=[
        0,
        100,
    ],
)


st.plotly_chart(
    fig,
    use_container_width=True,
)


st.dataframe(
    forecast_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# EARLY WARNING
# ============================================================

st.divider()

st.subheader(
    "🚨 Early Warning System"
)

max_forecast = float(
    forecast_df["Risk"].max()
)

if max_forecast >= 80:

    st.error(
        "🚨 CRITICAL EARLY WARNING — "
        "The prototype forecast reaches the critical disruption range."
    )

elif max_forecast >= 60:

    st.warning(
        "⚠️ HIGH-RISK EARLY WARNING — "
        "The prototype forecast reaches the high-risk range."
    )

elif max_forecast >= 30:

    st.info(
        "🟡 CAUTION — "
        "Moderate disruption risk is present in the forecast."
    )

else:

    st.success(
        "🟢 No major disruption signal detected in the current prototype forecast."
    )


# ============================================================
# WEATHER INTELLIGENCE
# ============================================================

st.divider()

st.subheader(
    "🌦️ Environmental Intelligence"
)

w1, w2, w3, w4 = st.columns(4)

w1.metric(
    "Rainfall",
    f"{rainfall} mm",
)

w2.metric(
    "Wind",
    f"{wind} km/h",
)

w3.metric(
    "Weather Risk",
    f"{weather_risk:.0f}%",
)

w4.metric(
    "Flood Exposure",
    f"{river_exposure * 100:.0f}%",
)


# ============================================================
# DISASTER WHAT-IF
# ============================================================

st.divider()

st.subheader(
    "🧪 What-If Disaster Scenario"
)

scenario = st.selectbox(
    "Select scenario",
    [
        "Current Conditions",
        "Heavy Rainfall",
        "Extreme Rainfall",
        "Landslide Conditions",
        "Heavy Traffic",
        "Poor Road Condition",
        "Combined Disaster",
    ],
)


scenario_rain = rainfall
scenario_slope = slope
scenario_history = historical
scenario_condition = road_condition
scenario_traffic = traffic


if scenario == "Heavy Rainfall":

    scenario_rain *= 2


elif scenario == "Extreme Rainfall":

    scenario_rain *= 4


elif scenario == "Landslide Conditions":

    scenario_rain *= 2
    scenario_slope += 15
    scenario_history += 3


elif scenario == "Heavy Traffic":

    scenario_traffic = 1.0


elif scenario == "Poor Road Condition":

    scenario_condition = 1.0


elif scenario == "Combined Disaster":

    scenario_rain *= 4
    scenario_slope += 15
    scenario_history += 5
    scenario_condition = 1.0
    scenario_traffic = 1.0


scenario_risk = predict_risk(
    model,
    min(
        scenario_rain,
        250,
    ),
    min(
        scenario_slope,
        60,
    ),
    min(
        scenario_history,
        15,
    ),
    min(
        scenario_condition,
        1,
    ),
    min(
        scenario_traffic,
        1,
    ),
)


scenario_risk = min(
    100,
    scenario_risk
    + flood_penalty
    + visibility_penalty,
)


s1, s2, s3 = st.columns(3)


s1.metric(
    "Current Risk",
    f"{advanced_risk:.0f}%",
)


s2.metric(
    "Scenario Risk",
    f"{scenario_risk:.0f}%",
)


s3.metric(
    "Risk Change",
    f"{scenario_risk - advanced_risk:+.0f}%",
)


# ============================================================
# NETWORK-WIDE RISK RANKING
# ============================================================

st.divider()

st.subheader(
    "🌐 NER Network-Wide Risk Ranking"
)

if not network_df.empty:

    ranking = network_df[
        [
            "id",
            "name",
            "a",
            "b",
            "dist",
            "risk_pct",
            "status",
        ]
    ].copy()

    ranking = ranking.rename(
        columns={
            "id": "Road ID",
            "name": "Road",
            "a": "From",
            "b": "To",
            "dist": "Distance (km)",
            "risk_pct": "AI Risk %",
            "status": "Status",
        }
    )

    ranking[
        "AI Risk %"
    ] = ranking[
        "AI Risk %"
    ].round(1)

    st.dataframe(
        ranking,
        use_container_width=True,
        hide_index=True,
    )

    chart_df = ranking.head(
        10
    ).copy()

    fig = px.bar(
        chart_df,
        x="AI Risk %",
        y="Road",
        orientation="h",
        color="AI Risk %",
        color_continuous_scale="OrRd",
    )

    fig.update_layout(
        height=450,
        yaxis={
            "categoryorder": "total ascending"
        },
        coloraxis_showscale=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# INCIDENT INTELLIGENCE
# ============================================================

st.divider()

st.subheader(
    "🚧 Active Field Incidents"
)

incidents = (
    st.session_state.incidents
)

if (
    incidents is not None
    and not incidents.empty
):

    open_incidents = incidents[
        incidents[
            "status"
        ] == "OPEN"
    ]

    if open_incidents.empty:

        st.success(
            "No open field incidents."
        )

    else:

        st.dataframe(
            open_incidents,
            use_container_width=True,
            hide_index=True,
        )

else:

    st.info(
        "No field incidents have been reported."
    )


# ============================================================
# AI MODEL ARCHITECTURE
# ============================================================

st.divider()

with st.expander(
    "🧠 AI Model Architecture & Methodology"
):

   st.markdown(
    """
    ## 🧠 AI Model Architecture

    ```text
    Input Data
        ↓
    Feature Engineering
        ↓
    Random Forest
        ↓
    Risk Score
        ↓
    Route Status
    ```
    """
)