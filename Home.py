import streamlit as st
import pandas as pd

from common import (
    init_state,
    build_live_graph,
    edges_dataframe,
    STATUS_COLORS,
)

st.set_page_config(
    page_title="NER-SMART | Command Center",
    page_icon="🚚",
    layout="wide",
)

init_state()

# ============================================================
# HEADER
# ============================================================

st.title("🛰️ NER-SMART")
st.subheader("AI-Powered Logistics & Accessibility Intelligence Platform")

st.caption(
    "North Eastern Region • Logistics Intelligence • Accessibility Monitoring • "
    "AI Risk Prediction • Fleet Tracking"
)

st.markdown(
    "**Predict. Navigate. Deliver. Connect the Northeast.**"
)

# ============================================================
# NETWORK DATA
# ============================================================

G = build_live_graph()
edges_df = edges_dataframe(G)

total_roads = len(edges_df)

blocked = int(
    (edges_df["status"] == "BLOCKED").sum()
)

high_risk = int(
    (edges_df["status"] == "HIGH RISK").sum()
)

partial = int(
    (edges_df["status"] == "PARTIALLY DISRUPTED").sum()
)

normal = int(
    (edges_df["status"] == "NORMAL").sum()
)

accessible_pct = (
    round(100 * (total_roads - blocked) / total_roads, 1)
    if total_roads
    else 0
)

vehicles = st.session_state.vehicles

active_vehicles = 0
stopped_vehicles = 0

if hasattr(vehicles, "empty") and not vehicles.empty:
    active_vehicles = int(
        (vehicles["status"].astype(str).str.upper() == "ACTIVE").sum()
    )
    stopped_vehicles = int(
        (vehicles["status"].astype(str).str.upper() == "STOPPED").sum()
    )
else:
    for vehicle in vehicles:
        status = str(vehicle.get("status", "")).upper()

        if status == "ACTIVE":
            active_vehicles += 1
        elif status == "STOPPED":
            stopped_vehicles += 1

incidents = st.session_state.incidents

if hasattr(incidents, "empty") and not incidents.empty:
    open_incidents = int(
        (
            incidents["status"]
            .astype(str)
            .str.upper()
            == "OPEN"
        ).sum()
    )
else:
    open_incidents = 0


# ============================================================
# TOP OPERATIONAL METRICS
# ============================================================

st.markdown("### 📊 Regional Operations")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Network Accessible",
    f"{accessible_pct}%",
)

c2.metric(
    "🚧 Blocked Roads",
    blocked,
)

c3.metric(
    "⚠️ High-Risk Corridors",
    high_risk,
)

c4.metric(
    "🚚 Active Vehicles",
    active_vehicles,
)

c5.metric(
    "🚨 Open Incidents",
    open_incidents,
)

st.divider()


# ============================================================
# SYSTEM HEALTH
# ============================================================

st.markdown("### 🟢 System Overview")

s1, s2, s3, s4 = st.columns(4)

s1.success(
    f"**{normal}** normal corridors"
)

s2.warning(
    f"**{partial}** partially disrupted"
)

s3.error(
    f"**{blocked}** blocked corridors"
)

s4.info(
    f"**{stopped_vehicles}** stopped vehicles"
)

st.divider()


# ============================================================
# MAIN DASHBOARD
# ============================================================

left, right = st.columns([2.1, 1])


# ============================================================
# ROAD NETWORK
# ============================================================

with left:

    st.subheader("🛣️ Road Network Accessibility")

    display_df = edges_df[
        [
            "name",
            "a",
            "b",
            "dist",
            "risk_pct",
            "status",
        ]
    ].rename(
        columns={
            "name": "Route",
            "a": "From",
            "b": "To",
            "dist": "Distance (km)",
            "risk_pct": "AI Risk %",
            "status": "Accessibility",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=390,
    )


# ============================================================
# STATUS PANEL
# ============================================================

with right:

    st.subheader("🚦 Network Status")

    counts = (
        edges_df["status"]
        .value_counts()
        .reindex(
            [
                "NORMAL",
                "PARTIALLY DISRUPTED",
                "HIGH RISK",
                "BLOCKED",
            ]
        )
        .fillna(0)
    )

    for status, count in counts.items():

        color = STATUS_COLORS.get(
            status,
            "#64748b",
        )

        st.markdown(
            f"""
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                padding:10px 12px;
                margin-bottom:7px;
                border-radius:8px;
                background:{color}22;
                border-left:5px solid {color};
            ">
                <span>{status}</span>
                <b>{int(count)}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🚨 Recent Incidents")

    if incidents.empty:

        st.caption(
            "No incidents reported."
        )

    else:

        recent = (
            incidents
            .tail(5)
            .iloc[::-1]
        )

        for _, incident in recent.iterrows():

            severity = str(
                incident.get(
                    "severity",
                    "UNKNOWN",
                )
            )

            st.markdown(
                f"""
                **{incident.get('type', 'Incident')}**  
                📍 {incident.get('location', 'Unknown')}  
                Severity: **{severity}**
                """
            )


st.divider()


# ============================================================
# LIVE FLEET SUMMARY
# ============================================================

st.subheader("🚚 Essential Logistics Fleet")

if hasattr(vehicles, "empty"):

    if vehicles.empty:

        st.info(
            "No GPS vehicles currently available."
        )

    else:

        fleet_display = vehicles.copy()

        columns = [
            c
            for c in [
                "id",
                "status",
                "speed",
                "cargo",
                "origin",
                "destination",
            ]
            if c in fleet_display.columns
        ]

        st.dataframe(
            fleet_display[columns],
            use_container_width=True,
            hide_index=True,
        )

else:

    fleet_rows = []

    for vehicle in vehicles:

        fleet_rows.append(
            {
                "Vehicle": vehicle.get(
                    "id",
                    "Unknown",
                ),
                "Status": vehicle.get(
                    "status",
                    "UNKNOWN",
                ),
                "Speed": f"{vehicle.get('speed', 0)} km/h",
                "Cargo": vehicle.get(
                    "cargo",
                    "—",
                ),
                "Origin": vehicle.get(
                    "origin",
                    "—",
                ),
                "Destination": vehicle.get(
                    "destination",
                    "—",
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(fleet_rows),
        use_container_width=True,
        hide_index=True,
    )


st.divider()


# ============================================================
# PLATFORM CAPABILITIES
# ============================================================

st.subheader("🧠 NER-SMART Intelligence Modules")

modules = [
    (
        "🗺️ GIS Map",
        "Real road network, districts, incidents, accessibility and risk corridors.",
    ),
    (
        "🤖 AI Risk Prediction",
        "Predict potential disruption from rainfall, terrain, incidents and road conditions.",
    ),
    (
        "🛣️ Route Optimizer",
        "Calculate routes considering distance, accessibility, weather and AI risk.",
    ),
    (
        "🚚 Fleet Tracking",
        "Monitor essential commodity vehicles using GPS and road-following routes.",
    ),
    (
        "📋 Field Reports",
        "Geo-tagged incident reporting with photographs and operational updates.",
    ),
    (
        "🌧️ Disaster Simulator",
        "Test network accessibility during floods, heavy rainfall and landslides.",
    ),
    (
        "🧠 AI Copilot",
        "Ask natural-language questions about current logistics conditions.",
    ),
]

module_cols = st.columns(2)

for i, (title, description) in enumerate(modules):

    with module_cols[i % 2]:

        st.markdown(
            f"""
            <div style="
                padding:15px;
                margin-bottom:12px;
                border:1px solid #e2e8f0;
                border-radius:10px;
                background:#ffffff;
            ">
                <h4 style="margin:0 0 6px 0;">
                    {title}
                </h4>
                <p style="margin:0;color:#64748b;">
                    {description}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.divider()


# ============================================================
# PROBLEM STATEMENT COVERAGE
# ============================================================

st.subheader("🎯 Problem Statement Coverage")

coverage = pd.DataFrame(
    [
        ["A", "Real-time road & transport accessibility", "GIS Map"],
        ["B", "Disruption prediction", "AI Risk Prediction"],
        ["C", "Alternate route & delay estimation", "Route Optimizer"],
        ["D", "GPS vehicle tracking", "Fleet Tracking"],
        ["E", "Alerts & high-risk corridors", "Alerts + AI Risk"],
        ["F", "Geo-tagged field reporting", "Field Reports"],
        ["G", "Centralized logistics dashboard", "NER-SMART Home"],
        ["H", "Multilingual + offline support", "Field / Mobile Layer"],
    ],
    columns=[
        "Requirement",
        "Problem Statement",
        "NER-SMART Module",
    ],
)

st.dataframe(
    coverage,
    use_container_width=True,
    hide_index=True,
)


st.divider()

st.caption(
    "NER-SMART • AI-Based Smart Logistics & Accessibility Intelligence Platform "
    "for the North Eastern Region"
)