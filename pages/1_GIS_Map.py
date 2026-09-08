import plotly.graph_objects as go
import streamlit as st

from common import init_state, build_live_graph, DISTRICTS, STATUS_COLORS

init_state()
st.title("\U0001F5FA\uFE0F GIS Accessibility Map")
st.caption("Live digital twin of the NER road network. Click a road in the table to inspect it.")

G = build_live_graph()

layers = st.multiselect(
    "Map layers", ["Roads", "Districts", "Vehicles"],
    default=["Roads", "Districts", "Vehicles"],
)

fig = go.Figure()

if "Roads" in layers:
    for a, b, d in G.edges(data=True):
        lat1, lon1 = DISTRICTS[a]
        lat2, lon2 = DISTRICTS[b]
        color = STATUS_COLORS[d["status"]]
        fig.add_trace(go.Scattermapbox(
            lat=[lat1, lat2], lon=[lon1, lon2], mode="lines",
            line=dict(width=4, color=color),
            hoverinfo="text",
            text=f"{d['name']} ({a} \u2192 {b})<br>Status: {d['status']}<br>Risk: {d['risk_pct']:.0f}%",
            showlegend=False,
        ))

if "Districts" in layers:
    fig.add_trace(go.Scattermapbox(
        lat=[v[0] for v in DISTRICTS.values()],
        lon=[v[1] for v in DISTRICTS.values()],
        mode="markers+text",
        marker=dict(size=12, color="#2c3e50"),
        text=list(DISTRICTS.keys()), textposition="top right",
        name="Districts",
    ))

if "Vehicles" in layers:
    vlat, vlon, vtext = [], [], []
    for v in st.session_state.vehicles:
        edges = list(zip(v["path"][:-1], v["path"][1:]))
        if not edges:
            continue
        idx = min(int(v["progress"] * len(edges)), len(edges) - 1)
        a, b = edges[idx]
        frac_within = (v["progress"] * len(edges)) - idx
        lat1, lon1 = DISTRICTS[a]
        lat2, lon2 = DISTRICTS[b]
        vlat.append(lat1 + (lat2 - lat1) * frac_within)
        vlon.append(lon1 + (lon2 - lon1) * frac_within)
        vtext.append(f"{v['id']} ({v['cargo']}) \u2192 {v['destination']}")
    fig.add_trace(go.Scattermapbox(
        lat=vlat, lon=vlon, mode="markers", marker=dict(size=14, color="#3498db", symbol="circle"),
        text=vtext, name="Vehicles",
    ))

fig.update_layout(
    mapbox=dict(style="open-street-map", zoom=5.2,
                center=dict(lat=25.7, lon=92.5)),
    margin=dict(l=0, r=0, t=0, b=0), height=560,
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Road Detail")
road_names = {f"{d['name']} ({a} \u2192 {b})": (a, b) for a, b, d in G.edges(data=True)}
choice = st.selectbox("Select a road", list(road_names.keys()))
a, b = road_names[choice]
d = G.edges[a, b]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Status", d["status"])
c2.metric("Risk", f"{d['risk_pct']:.0f}%")
c3.metric("Distance", f"{d['dist']} km")
c4.metric("Rainfall", f"{d['rainfall']:.0f} mm")
