import streamlit as st

from common import init_state, build_live_graph, edges_dataframe

init_state()
st.title("\U0001F9EA Disaster What-If Simulator")
st.caption("Turn the dials to stress-test the network before a real event happens.")

scenario = st.selectbox("Preset scenario", [
    "Normal Operations", "Extreme Rainfall", "Major Landslide Belt",
    "Multiple Road Closures", "Custom",
])

presets = {
    "Normal Operations": dict(rainfall_multiplier=1.0, landslide_bias=0.0, road_condition_penalty=0.0),
    "Extreme Rainfall": dict(rainfall_multiplier=3.0, landslide_bias=1.0, road_condition_penalty=0.1),
    "Major Landslide Belt": dict(rainfall_multiplier=2.0, landslide_bias=4.0, road_condition_penalty=0.2),
    "Multiple Road Closures": dict(rainfall_multiplier=2.5, landslide_bias=3.0, road_condition_penalty=0.35),
}

if scenario != "Custom":
    defaults = presets[scenario]
else:
    defaults = st.session_state.sim_controls

c1, c2, c3 = st.columns(3)
rainfall_multiplier = c1.slider("Rainfall multiplier", 0.5, 4.0, float(defaults["rainfall_multiplier"]), 0.1)
landslide_bias = c2.slider("Landslide / historical-incident bias", 0.0, 5.0, float(defaults["landslide_bias"]), 0.5)
road_condition_penalty = c3.slider("Road condition penalty", 0.0, 0.5, float(defaults["road_condition_penalty"]), 0.05)

# ------------------------------------------------------------
# "Current" state — reflects whatever scenario is active RIGHT
# NOW, before any new dial values are applied.
# ------------------------------------------------------------
current_G = build_live_graph()
current_df = edges_dataframe(current_G)
current_blocked = int((current_df.status == "BLOCKED").sum())
current_high = int((current_df.status == "HIGH RISK").sum())

if st.button("\u26A1 Run Simulation", type="primary"):
    # Snapshot the state as it stood BEFORE this click, so the
    # comparison after rerun is meaningful instead of comparing
    # the new state against itself.
    st.session_state.sim_before_snapshot = dict(
        blocked=current_blocked,
        high=current_high,
    )
    st.session_state.sim_controls = dict(
        rainfall_multiplier=rainfall_multiplier, landslide_bias=landslide_bias,
        road_condition_penalty=road_condition_penalty, active_scenario=scenario,
    )
    st.rerun()

st.divider()

# ------------------------------------------------------------
# "After" state — the graph under whatever scenario is active
# now (same as current_*, but kept as its own read for clarity
# and so this block still works correctly on the very first
# page load, before any simulation has ever been run).
# ------------------------------------------------------------
after_G = build_live_graph()
after_df = edges_dataframe(after_G)
after_blocked = int((after_df.status == "BLOCKED").sum())
after_high = int((after_df.status == "HIGH RISK").sum())

affected_vehicles = 0
for v in st.session_state.vehicles:
    edges = list(zip(v["path"][:-1], v["path"][1:]))
    if any(after_G.edges[a, b]["status"] in ("HIGH RISK", "BLOCKED") for a, b in edges):
        affected_vehicles += 1

before_snapshot = st.session_state.get("sim_before_snapshot")

st.subheader(f"Result — active scenario: {st.session_state.sim_controls['active_scenario']}")
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Blocked roads",
    after_blocked,
    (after_blocked - before_snapshot["blocked"]) if before_snapshot else None,
)
c2.metric(
    "High-risk corridors",
    after_high,
    (after_high - before_snapshot["high"]) if before_snapshot else None,
)
c3.metric("Vehicles affected", affected_vehicles)
c4.metric("Total roads", len(after_df))

if not before_snapshot:
    st.caption("Run a simulation to see before → after deltas above.")

st.subheader("Current road network under this scenario")
st.dataframe(
    after_df[["name", "a", "b", "dist", "risk_pct", "status"]]
    .rename(columns={"name": "Route", "a": "From", "b": "To",
                      "dist": "Distance (km)", "risk_pct": "Risk %", "status": "Status"}),
    use_container_width=True, hide_index=True,
)

st.info(
    "Go to **Route Optimizer** or **Fleet Tracking** now — routing and vehicle risk "
    "there reflect whatever scenario is active here, since every page reads the same "
    "live graph."
)