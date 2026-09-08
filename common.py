"""
common.py
Shared data model, synthetic ML risk model, and routing graph for the
NER-SMART Logistics Intelligence prototype.

Every Streamlit page imports `init_state()` and calls it first, which
seeds st.session_state exactly once per browser session. All pages then
read/write the same session_state objects, so the app behaves like one
connected system instead of independent demos.
"""

import random
from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# --------------------------------------------------------------------------
# 1. STATIC GEOGRAPHY: districts (nodes) and roads (edges)
# --------------------------------------------------------------------------

DISTRICTS = {
    "Guwahati":    (26.1445, 91.7362),
    "Shillong":    (25.5788, 91.8933),
    "Aizawl":      (23.7271, 92.7176),
    "Imphal":      (24.8170, 93.9368),
    "Kohima":      (25.6751, 94.1086),
    "Itanagar":    (27.0844, 93.6053),
    "Agartala":    (23.8315, 91.2868),
    "Gangtok":     (27.3389, 88.6065),
    "Dimapur":     (25.9091, 93.7264),
    "Tawang":      (27.5859, 91.8594),
    "Silchar":     (24.8333, 92.7789),
    "Dima Hasao":  (25.3667, 93.0167),
}

# Each road: id, endpoints, highway name, base distance (km),
# base risk (0-1, structural/terrain risk independent of weather),
# road_type, elevation/slope proxy used by the risk model.
ROADS = [
    dict(id="R1", a="Guwahati", b="Shillong",   name="NH-6",   dist=100, base_risk=0.15, road_type="highway", slope=18),
    dict(id="R2", a="Guwahati", b="Dimapur",    name="NH-27",  dist=270, base_risk=0.25, road_type="highway", slope=22),
    dict(id="R3", a="Dimapur",  b="Kohima",     name="NH-29",  dist=75,  base_risk=0.35, road_type="hill",    slope=35),
    dict(id="R4", a="Kohima",   b="Imphal",     name="NH-2",   dist=140, base_risk=0.40, road_type="hill",    slope=38),
    dict(id="R5", a="Guwahati", b="Silchar",    name="NH-6",   dist=310, base_risk=0.30, road_type="highway", slope=20),
    dict(id="R6", a="Silchar",  b="Aizawl",     name="NH-306", dist=180, base_risk=0.55, road_type="hill",    slope=42),
    dict(id="R7", a="Silchar",  b="Agartala",   name="NH-8",   dist=160, base_risk=0.35, road_type="highway", slope=24),
    dict(id="R8", a="Agartala", b="Aizawl",     name="NH-108", dist=250, base_risk=0.50, road_type="hill",    slope=40),
    dict(id="R9", a="Guwahati", b="Itanagar",   name="NH-15",  dist=310, base_risk=0.35, road_type="highway", slope=26),
    dict(id="R10", a="Itanagar", b="Tawang",    name="NH-13",  dist=320, base_risk=0.65, road_type="hill",    slope=48),
    dict(id="R11", a="Guwahati", b="Gangtok",   name="NH-10",  dist=280, base_risk=0.30, road_type="highway", slope=28),
    dict(id="R12", a="Shillong", b="Silchar",   name="NH-6",   dist=200, base_risk=0.28, road_type="highway", slope=22),
    dict(id="R13", a="Dimapur",  b="Imphal",    name="NH-2",   dist=215, base_risk=0.32, road_type="highway", slope=25),
    dict(id="R14", a="Dima Hasao", b="Silchar", name="NH-27",  dist=90,  base_risk=0.45, road_type="hill",    slope=36),
    dict(id="R15", a="Dima Hasao", b="Guwahati", name="NH-27", dist=200, base_risk=0.38, road_type="hill",    slope=30),
]

CARGO_PRIORITY = {
    "Medicines": 1, "Medical Equipment": 1,
    "Food": 2, "Water": 2,
    "Construction Materials": 3, "Agricultural Produce": 3,
}

INCIDENT_TYPES = ["Landslide", "Flood", "Heavy Rainfall", "Road Damage",
                   "Bridge Damage", "Road Blockage", "Traffic", "Other"]

STATUS_COLORS = {
    "NORMAL": "#2ecc71", "PARTIALLY DISRUPTED": "#f1c40f",
    "HIGH RISK": "#e67e22", "BLOCKED": "#e74c3c", "UNKNOWN": "#7f8c8d",
}


def risk_to_status(risk_pct: float) -> str:
    if risk_pct >= 80:
        return "BLOCKED"
    if risk_pct >= 60:
        return "HIGH RISK"
    if risk_pct >= 30:
        return "PARTIALLY DISRUPTED"
    return "NORMAL"


# --------------------------------------------------------------------------
# 2. SYNTHETIC ML RISK MODEL
#    A small RandomForest trained on synthetically generated examples that
#    encode a plausible relationship between weather/terrain/traffic and
#    disruption probability. This is a stand-in for a model that, in
#    production, would be trained on historical incident + weather data.
# --------------------------------------------------------------------------

FEATURES = ["rainfall_mm", "slope_deg", "historical_incidents",
            "road_condition", "traffic_level"]


@st.cache_resource(show_spinner=False)
def train_risk_model():
    rng = np.random.default_rng(42)
    n = 4000
    rainfall = rng.gamma(2.0, 25, n)                      # 0 - 250mm-ish
    slope = rng.uniform(5, 55, n)                         # degrees
    historical = rng.poisson(2.5, n)                      # incident count
    road_condition = rng.uniform(0, 1, n)                 # 0 good -> 1 poor
    traffic = rng.uniform(0, 1, n)                         # 0 low -> 1 heavy

    # ground-truth generating function (nonlinear, with noise)
    score = (
        0.32 * (rainfall / 250)
        + 0.28 * (slope / 55)
        + 0.15 * (historical / 8).clip(0, 1)
        + 0.15 * road_condition
        + 0.10 * traffic
    )
    score = score + rng.normal(0, 0.05, n)
    score = np.clip(score, 0, 1) * 100

    X = pd.DataFrame({
        "rainfall_mm": rainfall, "slope_deg": slope,
        "historical_incidents": historical,
        "road_condition": road_condition, "traffic_level": traffic,
    })
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X, score)
    return model


def predict_risk(model, rainfall_mm, slope_deg, historical_incidents,
                  road_condition, traffic_level):
    X = pd.DataFrame([{
        "rainfall_mm": rainfall_mm, "slope_deg": slope_deg,
        "historical_incidents": historical_incidents,
        "road_condition": road_condition, "traffic_level": traffic_level,
    }])
    pred = float(model.predict(X)[0])
    return float(np.clip(pred, 0, 100))


def explain_risk(rainfall_mm, slope_deg, historical_incidents,
                  road_condition, traffic_level):
    """Lightweight, transparent contribution breakdown (not SHAP, but the
    same linear form used to generate training labels) so officials can see
    *why* a score was produced -- explainability matters more than raw
    model fidelity for a decision-support tool like this."""
    contributions = {
        "Rainfall": 0.32 * (rainfall_mm / 250) * 100,
        "Terrain slope": 0.28 * (slope_deg / 55) * 100,
        "Historical incidents": 0.15 * min(historical_incidents / 8, 1) * 100,
        "Road condition": 0.15 * road_condition * 100,
        "Traffic": 0.10 * traffic_level * 100,
    }
    return contributions


# --------------------------------------------------------------------------
# 3. SESSION STATE INITIALISATION
# --------------------------------------------------------------------------

def _init_weather():
    rng = random.Random(7)
    rows = []
    for d in DISTRICTS:
        rows.append(dict(
            district=d,
            rainfall_mm=round(rng.uniform(5, 60), 1),
            temperature_c=round(rng.uniform(15, 30), 1),
            wind_kmh=round(rng.uniform(5, 25), 1),
            forecast=rng.choice(["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Thunderstorm"]),
        ))
    return pd.DataFrame(rows)


def _init_vehicles():
    G = build_base_graph()
    templates = [
        ("NER-1023", "Medicines", "Guwahati", "Aizawl", "Heavy Truck"),
        ("NER-2044", "Food", "Guwahati", "Kohima", "Truck"),
        ("NER-3087", "Water", "Silchar", "Agartala", "Truck"),
        ("NER-4821", "Medical Equipment", "Guwahati", "Itanagar", "Van"),
        ("NER-5560", "Construction Materials", "Guwahati", "Gangtok", "Heavy Truck"),
        ("NER-6132", "Agricultural Produce", "Dimapur", "Imphal", "Truck"),
    ]
    vehicles = []
    for vid, cargo, origin, dest, vtype in templates:
        try:
            path = nx.shortest_path(G, origin, dest, weight="dist")
        except nx.NetworkXNoPath:
            path = [origin, dest]
        vehicles.append(dict(
            id=vid, cargo=cargo, vehicle_type=vtype, origin=origin,
            destination=dest, path=path, progress=round(random.uniform(0.05, 0.7), 2),
            speed_kmh=random.randint(30, 55),
        ))
    return vehicles


def build_base_graph():
    """Graph with only static/base risk -- used for initial vehicle path
    assignment before any live conditions are simulated."""
    G = nx.Graph()
    for d, (lat, lon) in DISTRICTS.items():
        G.add_node(d, lat=lat, lon=lon)
    for r in ROADS:
        G.add_edge(r["a"], r["b"], **r)
    return G


def build_live_graph():
    """Graph whose edge weights reflect current weather + incidents +
    the disaster-simulator sliders. Rebuilt on every page render so route
    suggestions always reflect the latest state."""
    model = train_risk_model()
    weather = st.session_state.weather.set_index("district")
    incidents = st.session_state.incidents

    G = nx.Graph()
    for d, (lat, lon) in DISTRICTS.items():
        G.add_node(d, lat=lat, lon=lon)

    sim = st.session_state.sim_controls

    for r in ROADS:
        a_w = weather.loc[r["a"]]
        b_w = weather.loc[r["b"]]
        rainfall = max(a_w.rainfall_mm, b_w.rainfall_mm) * sim["rainfall_multiplier"]
        traffic = min(1.0, 0.3 + (0.4 if r["road_type"] == "highway" else 0.15))
        road_condition = min(1.0, r["base_risk"] + sim["road_condition_penalty"])

        # incident penalty: any open incident reported on this road pushes
        # historical_incidents up sharply for the live calculation
        open_incidents = incidents[(incidents["road_id"] == r["id"]) & (incidents["status"] == "OPEN")]
        historical = 1 + len(open_incidents) * 3 + sim["landslide_bias"]

        risk_pct = predict_risk(model, rainfall, r["slope"], historical,
                                 road_condition, traffic)
        status = risk_to_status(risk_pct)
        if not open_incidents.empty and open_incidents["severity"].eq("CRITICAL").any():
            status = "BLOCKED"
            risk_pct = max(risk_pct, 92)

        edge = dict(r)
        edge.update(risk_pct=risk_pct, status=status, rainfall=rainfall,
                     traffic=traffic, road_condition=road_condition,
                     historical=historical)
        # routing weight blends distance and risk; weight explodes for
        # BLOCKED roads so Dijkstra avoids them unless there's no choice
        block_penalty = 5000 if status == "BLOCKED" else 0
        edge["weight"] = r["dist"] * (1 + risk_pct / 40) + block_penalty
        G.add_edge(r["a"], r["b"], **edge)
    return G


def init_state():
    st.set_page_config(page_title="NER-SMART Logistics Intelligence",
                        page_icon="\U0001F6F0", layout="wide")

    if "initialized" in st.session_state:
        return

    st.session_state.weather = _init_weather()
    st.session_state.incidents = pd.DataFrame(columns=[
        "timestamp", "road_id", "road_name", "location", "type",
        "severity", "description", "reported_by", "status",
    ])
    st.session_state.vehicles = _init_vehicles()
    st.session_state.sim_controls = dict(
        rainfall_multiplier=1.0, landslide_bias=0.0, road_condition_penalty=0.0,
        active_scenario="Normal Operations",
    )
    st.session_state.tick = 0
    st.session_state.initialized = True


def add_incident(road_id, road_name, location, itype, severity, description, reported_by="Field Officer"):
    row = dict(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        road_id=road_id, road_name=road_name, location=location, type=itype,
        severity=severity, description=description, reported_by=reported_by,
        status="OPEN",
    )
    st.session_state.incidents = pd.concat(
        [st.session_state.incidents, pd.DataFrame([row])], ignore_index=True)


def edges_dataframe(G):
    rows = []
    for a, b, d in G.edges(data=True):
        rows.append(dict(
            id=d["id"], name=d["name"], a=a, b=b, dist=d["dist"],
            risk_pct=round(d["risk_pct"], 1), status=d["status"],
            road_type=d["road_type"],
        ))
    return pd.DataFrame(rows).sort_values("risk_pct", ascending=False)


def route_edges(G, path):
    """Return the list of edge dicts along a node path, in order."""
    edges = []
    for a, b in zip(path[:-1], path[1:]):
        edges.append(G.edges[a, b])
    return edges


def route_summary(G, path):
    edges = route_edges(G, path)
    total_dist = sum(e["dist"] for e in edges)
    max_risk = max((e["risk_pct"] for e in edges), default=0)
    avg_speed = 45 if max_risk < 40 else 30
    eta_hours = total_dist / avg_speed
    return dict(distance_km=total_dist, max_risk=round(max_risk, 1),
                eta_hours=round(eta_hours, 1), edges=edges)
