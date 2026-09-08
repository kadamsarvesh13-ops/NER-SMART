import networkx as nx
import streamlit as st

from common import init_state, build_live_graph, DISTRICTS, CARGO_PRIORITY, route_summary

init_state()
st.title("\U0001F6E3\uFE0F AI Route Optimizer")
st.caption("Not just \u201croad is blocked\u201d \u2014 the safest and fastest way to get there instead.")

G = build_live_graph()
names = list(DISTRICTS.keys())

c1, c2, c3, c4 = st.columns(4)
origin = c1.selectbox("Origin", names, index=names.index("Guwahati"))
destination = c2.selectbox("Destination", names, index=names.index("Aizawl"))
cargo = c3.selectbox("Cargo", list(CARGO_PRIORITY.keys()))
priority = c4.selectbox("Priority override", ["Auto (from cargo)", "Speed", "Safety"])

if origin == destination:
    st.warning("Choose two different districts.")
    st.stop()

# Route A: risk-aware shortest path (uses the blended weight built in common.py)
try:
    safe_path = nx.shortest_path(G, origin, destination, weight="weight")
except nx.NetworkXNoPath:
    safe_path = None

# Route B: pure distance shortest path (ignores risk) for comparison
try:
    fast_path = nx.shortest_path(G, origin, destination, weight="dist")
except nx.NetworkXNoPath:
    fast_path = None

# Route C: a k-shortest-paths alternative, if one exists, to show a 3rd option
alt_path = None
try:
    paths = list(nx.shortest_simple_paths(G, origin, destination, weight="dist"))
    for p in paths:
        if p != safe_path and p != fast_path:
            alt_path = p
            break
except nx.NetworkXNoPath:
    pass

st.divider()
cols = st.columns(3)
medals = ["\U0001F947", "\U0001F948", "\U0001F949"]
routes = [("AI Recommended (risk-aware)", safe_path), ("Fastest (distance-only)", fast_path),
          ("Alternative", alt_path)]

cargo_prio = CARGO_PRIORITY[cargo]
for col, medal, (label, path) in zip(cols, medals, routes):
    with col:
        st.markdown(f"### {medal} {label}")
        if not path:
            st.info("No path available.")
            continue
        summary = route_summary(G, path)
        st.write(" \u2192 ".join(path))
        st.metric("Distance", f"{summary['distance_km']} km")
        st.metric("Max segment risk", f"{summary['max_risk']}%")
        st.metric("ETA", f"{summary['eta_hours']} h")
        if any(e["status"] == "BLOCKED" for e in summary["edges"]):
            st.error("Route crosses a BLOCKED segment.")

st.divider()
if safe_path:
    summary = route_summary(G, safe_path)
    reason = []
    if cargo_prio == 1:
        reason.append(f"**{cargo}** is Priority 1 (critical), so the engine weighs risk far more heavily than raw distance.")
    elif cargo_prio == 2:
        reason.append(f"**{cargo}** is Priority 2, so risk and ETA are balanced.")
    else:
        reason.append(f"**{cargo}** is Priority 3, so the fastest reasonably safe route is preferred.")
    if fast_path and fast_path != safe_path:
        fast_summary = route_summary(G, fast_path)
        reason.append(
            f"The distance-only route is {fast_summary['distance_km'] - summary['distance_km']:+d} km "
            f"shorter but its riskiest segment sits at {fast_summary['max_risk']}% vs "
            f"{summary['max_risk']}% on the recommended route."
        )
    st.info("**Why this route:** " + " ".join(reason))

st.subheader("Segment-by-segment breakdown")
if safe_path:
    for e in route_summary(G, safe_path)["edges"]:
        st.markdown(
            f"- **{e['name']}** ({e['a']} \u2192 {e['b']}) \u2014 {e['dist']} km, "
            f"risk **{e['risk_pct']:.0f}%**, status **{e['status']}**"
        )
