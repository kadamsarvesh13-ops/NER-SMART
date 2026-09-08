import plotly.graph_objects as go
import streamlit as st

from common import init_state, build_live_graph, DISTRICTS, route_summary

init_state()
st.title("\U0001F69A Fleet Tracking")
st.caption("Simulated GPS vehicles moving along their currently assigned routes.")

G = build_live_graph()

c1, c2 = st.columns([1, 3])
with c1:
    if st.button("\u25B6\uFE0F Advance simulation (+30 min)", use_container_width=True):
        for v in st.session_state.vehicles:
            edges = list(zip(v["path"][:-1], v["path"][1:]))
            n_edges = max(len(edges), 1)
            step = (v["speed_kmh"] * 0.5) / max(sum(G.edges[a, b]["dist"] for a, b in edges), 1)
            v["progress"] = min(1.0, v["progress"] + step)
        st.session_state.tick += 1
        st.rerun()
    st.caption(f"Simulation ticks elapsed: {st.session_state.tick}")

fig = go.Figure()
for a, b, d in G.edges(data=True):
    lat1, lon1 = DISTRICTS[a]
    lat2, lon2 = DISTRICTS[b]
    fig.add_trace(go.Scattermap(
        lat=[lat1, lat2], lon=[lon1, lon2], mode="lines",
        line=dict(width=2, color="#bdc3c7"), hoverinfo="skip", showlegend=False,
    ))

fig.add_trace(go.Scattermap(
    lat=[v[0] for v in DISTRICTS.values()], lon=[v[1] for v in DISTRICTS.values()],
    mode="markers+text", marker=dict(size=10, color="#2c3e50"),
    text=list(DISTRICTS.keys()), textposition="top right", name="Districts",
))

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
    edge_data = G.edges[a, b]
    vtext.append(
        f"{v['id']} \u2014 {v['cargo']}<br>{v['origin']} \u2192 {v['destination']}<br>"
        f"On: {edge_data['name']} (risk {edge_data['risk_pct']:.0f}%)"
    )

fig.add_trace(go.Scattermap(
    lat=vlat, lon=vlon, mode="markers", marker=dict(size=16, color="#e74c3c"),
    text=vtext, hoverinfo="text", name="Vehicles",
))
fig.update_layout(mapbox=dict(style="open-street-map", zoom=5.2, center=dict(lat=25.7, lon=92.5)),
                   margin=dict(l=0, r=0, t=0, b=0), height=480)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Vehicle Manifest")
for v in st.session_state.vehicles:
    summary = route_summary(G, v["path"])
    risk_ahead = max((e["risk_pct"] for e in summary["edges"]), default=0)
    status = "\U0001F7E2 ON TRACK" if risk_ahead < 60 else "\U0001F7E0 AT RISK"
    with st.expander(f"{v['id']} \u2014 {v['cargo']} ({v['origin']} \u2192 {v['destination']}) {status}"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vehicle type", v["vehicle_type"])
        c2.metric("Speed", f"{v['speed_kmh']} km/h")
        c3.metric("Progress", f"{v['progress']*100:.0f}%")
        c4.metric("Route risk (max)", f"{risk_ahead:.0f}%")
        st.write("Route: " + " \u2192 ".join(v["path"]))
        remaining_km = summary["distance_km"] * (1 - v["progress"])
        st.caption(f"Estimated remaining distance: {remaining_km:.0f} km "
                   f"(~{remaining_km / v['speed_kmh']:.1f} h at current speed)")
