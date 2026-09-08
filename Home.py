import pandas as pd
import streamlit as st

from common import init_state, build_live_graph, edges_dataframe, STATUS_COLORS

init_state()

st.title("\U0001F6F0 NER-SMART")
st.caption("AI-Powered Logistics & Accessibility Intelligence Platform — North Eastern Region")
st.markdown("*Predict. Navigate. Deliver. Connect the Northeast.*")

G = build_live_graph()
edges_df = edges_dataframe(G)

total_roads = len(edges_df)
blocked = (edges_df.status == "BLOCKED").sum()
high_risk = (edges_df.status == "HIGH RISK").sum()
normal = (edges_df.status == "NORMAL").sum()
accessible_pct = round(100 * (total_roads - blocked) / total_roads, 1)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Network Accessible", f"{accessible_pct}%")
col2.metric("Blocked Roads", int(blocked))
col3.metric("High-Risk Corridors", int(high_risk))
col4.metric("Vehicles Active", len(st.session_state.vehicles))
col5.metric("Open Incidents", int((st.session_state.incidents.status == "OPEN").sum()))

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Road Network Status")
    st.dataframe(
        edges_df[["name", "a", "b", "dist", "risk_pct", "status"]]
        .rename(columns={"name": "Route", "a": "From", "b": "To",
                          "dist": "Distance (km)", "risk_pct": "Risk %",
                          "status": "Status"}),
        use_container_width=True, hide_index=True, height=380,
    )

with right:
    st.subheader("Status Breakdown")
    counts = edges_df["status"].value_counts().reindex(
        ["NORMAL", "PARTIALLY DISRUPTED", "HIGH RISK", "BLOCKED"]).fillna(0)
    for status, count in counts.items():
        color = STATUS_COLORS[status]
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:6px 10px;margin-bottom:4px;border-radius:6px;"
            f"background:{color}22;border-left:5px solid {color}'>"
            f"<span>{status}</span><b>{int(count)}</b></div>",
            unsafe_allow_html=True,
        )

    st.subheader("Recent Incidents")
    if st.session_state.incidents.empty:
        st.caption("No incidents reported yet — try the Field Reports page.")
    else:
        recent = st.session_state.incidents.tail(5).iloc[::-1]
        for _, r in recent.iterrows():
            st.markdown(f"**{r['type']}** — {r['location']} ({r['severity']})")

st.divider()
st.subheader("Where to go next")
st.markdown(
    "- **GIS Map** — see every district and road on an interactive map\n"
    "- **AI Risk Prediction** — explore why a road scores the way it does\n"
    "- **Route Optimizer** — get the safest vs. fastest route between two districts\n"
    "- **Fleet Tracking** — watch simulated vehicles move along live routes\n"
    "- **Field Reports** — submit an incident and watch the whole system react\n"
    "- **Disaster Simulator** — stress-test the network with a rainfall/landslide scenario\n"
    "- **AI Copilot** — ask plain-English questions about current conditions"
)
