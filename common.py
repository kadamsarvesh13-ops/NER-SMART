"""
NER-SMART
Shared Intelligence Core

This file provides:
- NER geography
- Road network data
- AI risk prediction
- Shared Streamlit state
- Live weather state
- Field incidents
- Fleet state
- Disaster simulation controls
- Route optimization
- Common utilities used by every page

Architecture:

Weather + Terrain + Traffic + Incidents
                ↓
          AI Risk Engine
                ↓
       Live Accessibility Graph
                ↓
       Route Optimization
                ↓
      Fleet / GIS / Alerts
"""

import random
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 1. NER GEOGRAPHY
# ============================================================

DISTRICTS = {
    "Guwahati": (26.1445, 91.7362),
    "Shillong": (25.5788, 91.8933),
    "Aizawl": (23.7271, 92.7176),
    "Imphal": (24.8170, 93.9368),
    "Kohima": (25.6751, 94.1086),
    "Itanagar": (27.0844, 93.6053),
    "Agartala": (23.8315, 91.2868),
    "Gangtok": (27.3389, 88.6065),
    "Dimapur": (25.9091, 93.7264),
    "Tawang": (27.5859, 91.8594),
    "Silchar": (24.8333, 92.7789),
    "Dima Hasao": (25.3667, 93.0167),
}


# ============================================================
# 2. ROAD NETWORK
# ============================================================

ROADS = [
    dict(
        id="R1",
        a="Guwahati",
        b="Shillong",
        name="NH-6",
        dist=100,
        base_risk=0.15,
        road_type="highway",
        slope=18,
    ),

    dict(
        id="R2",
        a="Guwahati",
        b="Dimapur",
        name="NH-27",
        dist=270,
        base_risk=0.25,
        road_type="highway",
        slope=22,
    ),

    dict(
        id="R3",
        a="Dimapur",
        b="Kohima",
        name="NH-29",
        dist=75,
        base_risk=0.35,
        road_type="hill",
        slope=35,
    ),

    dict(
        id="R4",
        a="Kohima",
        b="Imphal",
        name="NH-2",
        dist=140,
        base_risk=0.40,
        road_type="hill",
        slope=38,
    ),

    dict(
        id="R5",
        a="Guwahati",
        b="Silchar",
        name="NH-6",
        dist=310,
        base_risk=0.30,
        road_type="highway",
        slope=20,
    ),

    dict(
        id="R6",
        a="Silchar",
        b="Aizawl",
        name="NH-306",
        dist=180,
        base_risk=0.55,
        road_type="hill",
        slope=42,
    ),

    dict(
        id="R7",
        a="Silchar",
        b="Agartala",
        name="NH-8",
        dist=160,
        base_risk=0.35,
        road_type="highway",
        slope=24,
    ),

    dict(
        id="R8",
        a="Agartala",
        b="Aizawl",
        name="NH-108",
        dist=250,
        base_risk=0.50,
        road_type="hill",
        slope=40,
    ),

    dict(
        id="R9",
        a="Guwahati",
        b="Itanagar",
        name="NH-15",
        dist=310,
        base_risk=0.35,
        road_type="highway",
        slope=26,
    ),

    dict(
        id="R10",
        a="Itanagar",
        b="Tawang",
        name="NH-13",
        dist=320,
        base_risk=0.65,
        road_type="hill",
        slope=48,
    ),

    dict(
        id="R11",
        a="Guwahati",
        b="Gangtok",
        name="NH-10",
        dist=280,
        base_risk=0.30,
        road_type="highway",
        slope=28,
    ),

    dict(
        id="R12",
        a="Shillong",
        b="Silchar",
        name="NH-6",
        dist=200,
        base_risk=0.28,
        road_type="highway",
        slope=22,
    ),

    dict(
        id="R13",
        a="Dimapur",
        b="Imphal",
        name="NH-2",
        dist=215,
        base_risk=0.32,
        road_type="highway",
        slope=25,
    ),

    dict(
        id="R14",
        a="Dima Hasao",
        b="Silchar",
        name="NH-27",
        dist=90,
        base_risk=0.45,
        road_type="hill",
        slope=36,
    ),

    dict(
        id="R15",
        a="Dima Hasao",
        b="Guwahati",
        name="NH-27",
        dist=200,
        base_risk=0.38,
        road_type="hill",
        slope=30,
    ),
]


# ============================================================
# 3. CARGO PRIORITY
# ============================================================

CARGO_PRIORITY = {
    "Medicines": 1,
    "Medical Equipment": 1,
    "Food": 2,
    "Water": 2,
    "Construction Materials": 3,
    "Agricultural Produce": 3,
}


# ============================================================
# 4. INCIDENT TYPES
# ============================================================

INCIDENT_TYPES = [
    "Landslide",
    "Flood",
    "Heavy Rainfall",
    "Road Damage",
    "Bridge Damage",
    "Road Blockage",
    "Traffic",
    "Other",
]


# ============================================================
# 5. STATUS COLORS
# ============================================================

STATUS_COLORS = {
    "NORMAL": "#2ecc71",
    "PARTIALLY DISRUPTED": "#f1c40f",
    "HIGH RISK": "#e67e22",
    "BLOCKED": "#e74c3c",
    "UNKNOWN": "#7f8c8d",
}


# ============================================================
# 6. RISK STATUS
# ============================================================

def risk_to_status(risk_pct: float) -> str:

    risk_pct = float(risk_pct)

    if risk_pct >= 80:
        return "BLOCKED"

    if risk_pct >= 60:
        return "HIGH RISK"

    if risk_pct >= 30:
        return "PARTIALLY DISRUPTED"

    return "NORMAL"


# ============================================================
# 7. AI MODEL FEATURES
# ============================================================

FEATURES = [
    "rainfall_mm",
    "slope_deg",
    "historical_incidents",
    "road_condition",
    "traffic_level",
]


# ============================================================
# 8. TRAIN AI RISK MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def train_risk_model():

    rng = np.random.default_rng(42)

    n = 4000

    rainfall = rng.gamma(
        2.0,
        25,
        n,
    )

    rainfall = np.clip(
        rainfall,
        0,
        250,
    )

    slope = rng.uniform(
        5,
        55,
        n,
    )

    historical = rng.poisson(
        2.5,
        n,
    )

    road_condition = rng.uniform(
        0,
        1,
        n,
    )

    traffic = rng.uniform(
        0,
        1,
        n,
    )

    score = (
        0.32 * (rainfall / 250)
        +
        0.28 * (slope / 55)
        +
        0.15
        * np.clip(
            historical / 8,
            0,
            1,
        )
        +
        0.15 * road_condition
        +
        0.10 * traffic
    )

    score += rng.normal(
        0,
        0.05,
        n,
    )

    score = (
        np.clip(
            score,
            0,
            1,
        )
        * 100
    )

    X = pd.DataFrame(
        {
            "rainfall_mm": rainfall,
            "slope_deg": slope,
            "historical_incidents": historical,
            "road_condition": road_condition,
            "traffic_level": traffic,
        }
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X,
        score,
    )

    return model


# ============================================================
# 9. AI RISK PREDICTION
# ============================================================

def predict_risk(
    model,
    rainfall_mm,
    slope_deg,
    historical_incidents,
    road_condition,
    traffic_level,
):

    X = pd.DataFrame(
        [
            {
                "rainfall_mm": float(
                    rainfall_mm
                ),
                "slope_deg": float(
                    slope_deg
                ),
                "historical_incidents": float(
                    historical_incidents
                ),
                "road_condition": float(
                    road_condition
                ),
                "traffic_level": float(
                    traffic_level
                ),
            }
        ]
    )

    prediction = float(
        model.predict(X)[0]
    )

    return float(
        np.clip(
            prediction,
            0,
            100,
        )
    )


# ============================================================
# 10. RISK EXPLANATION
# ============================================================

def explain_risk(
    rainfall_mm,
    slope_deg,
    historical_incidents,
    road_condition,
    traffic_level,
):

    return {
        "Rainfall": (
            0.32
            * min(
                float(rainfall_mm) / 250,
                1,
            )
            * 100
        ),

        "Terrain slope": (
            0.28
            * min(
                float(slope_deg) / 55,
                1,
            )
            * 100
        ),

        "Historical incidents": (
            0.15
            * min(
                float(
                    historical_incidents
                ) / 8,
                1,
            )
            * 100
        ),

        "Road condition": (
            0.15
            * float(road_condition)
            * 100
        ),

        "Traffic": (
            0.10
            * float(traffic_level)
            * 100
        ),
    }


# ============================================================
# 11. BASE GRAPH
# ============================================================

def build_base_graph():

    G = nx.Graph()

    for district, (
        lat,
        lon,
    ) in DISTRICTS.items():

        G.add_node(
            district,
            lat=lat,
            lon=lon,
        )

    for road in ROADS:

        G.add_edge(
            road["a"],
            road["b"],
            **road,
        )

    return G


# ============================================================
# 12. WEATHER INITIALIZATION
# ============================================================

def _init_weather():

    rng = random.Random(7)

    rows = []

    for district in DISTRICTS:

        rows.append(
            {
                "district": district,
                "rainfall_mm": round(
                    rng.uniform(
                        5,
                        60,
                    ),
                    1,
                ),
                "temperature_c": round(
                    rng.uniform(
                        15,
                        30,
                    ),
                    1,
                ),
                "wind_kmh": round(
                    rng.uniform(
                        5,
                        25,
                    ),
                    1,
                ),
                "forecast": rng.choice(
                    [
                        "Clear",
                        "Cloudy",
                        "Light Rain",
                        "Heavy Rain",
                        "Thunderstorm",
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 13. VEHICLE INITIALIZATION
# ============================================================

def _init_vehicles():

    G = build_base_graph()

    templates = [
        (
            "NER-1023",
            "Medicines",
            "Guwahati",
            "Aizawl",
            "Heavy Truck",
        ),

        (
            "NER-2044",
            "Food",
            "Guwahati",
            "Kohima",
            "Truck",
        ),

        (
            "NER-3087",
            "Water",
            "Silchar",
            "Agartala",
            "Truck",
        ),

        (
            "NER-4821",
            "Medical Equipment",
            "Guwahati",
            "Itanagar",
            "Van",
        ),

        (
            "NER-5560",
            "Construction Materials",
            "Guwahati",
            "Gangtok",
            "Heavy Truck",
        ),

        (
            "NER-6132",
            "Agricultural Produce",
            "Dimapur",
            "Imphal",
            "Truck",
        ),
    ]

    vehicles = []

    for (
        vehicle_id,
        cargo,
        origin,
        destination,
        vehicle_type,
    ) in templates:

        try:

            path = nx.shortest_path(
                G,
                origin,
                destination,
                weight="dist",
            )

        except nx.NetworkXNoPath:

            path = [
                origin,
                destination,
            ]

        origin_coord = DISTRICTS[
            origin
        ]

        vehicles.append(
            {
                "id": vehicle_id,
                "cargo": cargo,
                "vehicle_type": vehicle_type,
                "origin": origin,
                "destination": destination,
                "path": path,

                "progress": round(
                    random.uniform(
                        0.05,
                        0.70,
                    ),
                    2,
                ),

                "speed_kmh": random.randint(
                    30,
                    55,
                ),

                "lat": origin_coord[0],
                "lon": origin_coord[1],

                "heading": 0,

                "status": "ACTIVE",

                "gps_status": "CONNECTED",

                "last_update": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        )

    return vehicles


# ============================================================
# 14. LIVE GRAPH
# ============================================================

def build_live_graph():

    model = train_risk_model()

    weather = (
        st.session_state.weather
        .set_index("district")
    )

    incidents = (
        st.session_state.incidents
    )

    sim = (
        st.session_state.sim_controls
    )

    G = nx.Graph()

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    for district, (
        lat,
        lon,
    ) in DISTRICTS.items():

        G.add_node(
            district,
            lat=lat,
            lon=lon,
        )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    for road in ROADS:

        a = road["a"]
        b = road["b"]

        try:

            a_weather = weather.loc[a]
            b_weather = weather.loc[b]

        except KeyError:

            rainfall = 20

        else:

            rainfall = max(
                float(
                    a_weather[
                        "rainfall_mm"
                    ]
                ),
                float(
                    b_weather[
                        "rainfall_mm"
                    ]
                ),
            )

        rainfall *= float(
            sim.get(
                "rainfall_multiplier",
                1.0,
            )
        )

        # ---------------------------------------------
        # Traffic
        # ---------------------------------------------

        if road["road_type"] == "highway":

            traffic = 0.30

        else:

            traffic = 0.45

        # ---------------------------------------------
        # Road condition
        # ---------------------------------------------

        road_condition = min(
            1.0,
            road["base_risk"]
            +
            float(
                sim.get(
                    "road_condition_penalty",
                    0,
                )
            ),
        )

        # ---------------------------------------------
        # Incidents
        # ---------------------------------------------

        try:

            open_incidents = incidents[
                (
                    incidents[
                        "road_id"
                    ]
                    == road["id"]
                )
                &
                (
                    incidents[
                        "status"
                    ]
                    == "OPEN"
                )
            ]

        except (
            KeyError,
            TypeError,
        ):

            open_incidents = pd.DataFrame()

        historical = (
            1
            +
            len(
                open_incidents
            )
            * 3
            +
            float(
                sim.get(
                    "landslide_bias",
                    0,
                )
            )
        )

        # ---------------------------------------------
        # AI prediction
        # ---------------------------------------------

        risk_pct = predict_risk(
            model,
            rainfall,
            road["slope"],
            historical,
            road_condition,
            traffic,
        )

        status = risk_to_status(
            risk_pct
        )

        # ---------------------------------------------
        # Critical incident
        # ---------------------------------------------

        try:

            critical_incident = (
                not open_incidents.empty
                and
                open_incidents[
                    "severity"
                ]
                .astype(str)
                .str.upper()
                .eq("CRITICAL")
                .any()
            )

        except Exception:

            critical_incident = False

        if critical_incident:

            status = "BLOCKED"

            risk_pct = max(
                risk_pct,
                92,
            )

        # ---------------------------------------------
        # Routing weight
        # ---------------------------------------------

        if status == "BLOCKED":

            routing_weight = (
                road["dist"]
                * 100
            )

        else:

            routing_weight = (
                road["dist"]
                *
                (
                    1
                    +
                    risk_pct / 40
                )
            )

        edge = dict(
            road
        )

        edge.update(
            {
                "risk_pct": round(
                    risk_pct,
                    2,
                ),

                "status": status,

                "rainfall": rainfall,

                "traffic": traffic,

                "road_condition": road_condition,

                "historical": historical,

                "weight": routing_weight,
            }
        )

        G.add_edge(
            a,
            b,
            **edge,
        )

    return G


# ============================================================
# 15. SESSION STATE
# ============================================================

def init_state():

    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    if "weather" not in st.session_state:

        st.session_state.weather = (
            _init_weather()
        )

    # --------------------------------------------------------
    # INCIDENTS
    # --------------------------------------------------------

    if "incidents" not in st.session_state:

        st.session_state.incidents = (
            pd.DataFrame(
                columns=[
                    "timestamp",
                    "road_id",
                    "road_name",
                    "location",
                    "type",
                    "severity",
                    "description",
                    "reported_by",
                    "status",
                ]
            )
        )

    # --------------------------------------------------------
    # VEHICLES
    # --------------------------------------------------------

    if (
        "vehicles"
        not in st.session_state
    ):

        st.session_state.vehicles = (
            _init_vehicles()
        )

    # --------------------------------------------------------
    # DISASTER SIMULATOR
    # --------------------------------------------------------

    if (
        "sim_controls"
        not in st.session_state
    ):

        st.session_state.sim_controls = {
            "rainfall_multiplier": 1.0,
            "landslide_bias": 0.0,
            "road_condition_penalty": 0.0,
            "active_scenario": "Normal Operations",
        }

    # --------------------------------------------------------
    # ROUTES
    # --------------------------------------------------------

    if (
        "calculated_routes"
        not in st.session_state
    ):

        st.session_state.calculated_routes = []

    # --------------------------------------------------------
    # SYSTEM TICK
    # --------------------------------------------------------

    if "tick" not in st.session_state:

        st.session_state.tick = 0

    # --------------------------------------------------------
    # INITIALIZED
    # --------------------------------------------------------

    st.session_state.initialized = True


# ============================================================
# 16. ADD FIELD INCIDENT
# ============================================================

def add_incident(
    road_id,
    road_name,
    location,
    itype,
    severity,
    description,
    reported_by="Field Officer",
):

    row = {
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "road_id": road_id,

        "road_name": road_name,

        "location": location,

        "type": itype,

        "severity": severity,

        "description": description,

        "reported_by": reported_by,

        "status": "OPEN",
    }

    new_row = pd.DataFrame(
        [row]
    )

    st.session_state.incidents = (
        pd.concat(
            [
                st.session_state.incidents,
                new_row,
            ],
            ignore_index=True,
        )
    )


# ============================================================
# 17. CLOSE INCIDENT
# ============================================================

def close_incident(index):

    if (
        "incidents"
        not in st.session_state
    ):
        return

    if (
        index < 0
        or index >= len(
            st.session_state.incidents
        )
    ):
        return

    st.session_state.incidents.loc[
        index,
        "status"
    ] = "CLOSED"


# ============================================================
# 18. ROAD DATAFRAME
# ============================================================

def edges_dataframe(G):

    rows = []

    for a, b, data in G.edges(
        data=True
    ):

        rows.append(
            {
                "id": data.get(
                    "id",
                    "",
                ),

                "name": data.get(
                    "name",
                    "",
                ),

                "a": a,

                "b": b,

                "dist": data.get(
                    "dist",
                    0,
                ),

                "risk_pct": round(
                    float(
                        data.get(
                            "risk_pct",
                            0,
                        )
                    ),
                    1,
                ),

                "status": data.get(
                    "status",
                    "UNKNOWN",
                ),

                "road_type": data.get(
                    "road_type",
                    "",
                ),
            }
        )

    if not rows:

        return pd.DataFrame(
            columns=[
                "id",
                "name",
                "a",
                "b",
                "dist",
                "risk_pct",
                "status",
                "road_type",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "risk_pct",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 19. ROUTE EDGES
# ============================================================

def route_edges(
    G,
    path,
):

    edges = []

    if not path or len(path) < 2:

        return edges

    for a, b in zip(
        path[:-1],
        path[1:],
    ):

        if G.has_edge(
            a,
            b,
        ):

            edges.append(
                G.edges[
                    a,
                    b
                ]
            )

    return edges


# ============================================================
# 20. ROUTE SUMMARY
# ============================================================

def route_summary(
    G,
    path,
):

    edges = route_edges(
        G,
        path,
    )

    total_distance = sum(
        float(
            edge.get(
                "dist",
                0,
            )
        )
        for edge in edges
    )

    max_risk = max(
        (
            float(
                edge.get(
                    "risk_pct",
                    0,
                )
            )
            for edge in edges
        ),
        default=0,
    )

    if max_risk < 30:

        average_speed = 50

    elif max_risk < 60:

        average_speed = 40

    elif max_risk < 80:

        average_speed = 30

    else:

        average_speed = 15

    if average_speed > 0:

        eta_hours = (
            total_distance
            / average_speed
        )

    else:

        eta_hours = 0

    return {
        "distance_km": round(
            total_distance,
            1,
        ),

        "max_risk": round(
            max_risk,
            1,
        ),

        "eta_hours": round(
            eta_hours,
            1,
        ),

        "edges": edges,

        "status": risk_to_status(
            max_risk
        ),
    }


# ============================================================
# 21. OPTIMIZED ROUTE
# ============================================================

def calculate_route(
    origin,
    destination,
):

    G = build_live_graph()

    if (
        origin not in G
        or destination not in G
    ):

        return None

    try:

        path = nx.shortest_path(
            G,
            origin,
            destination,
            weight="weight",
        )

    except nx.NetworkXNoPath:

        return None

    summary = route_summary(
        G,
        path,
    )

    result = {
        "origin": origin,
        "destination": destination,
        "path": path,
        **summary,
        "calculated_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    return result


# ============================================================
# 22. NETWORK STATISTICS
# ============================================================

def network_statistics():

    G = build_live_graph()

    df = edges_dataframe(
        G
    )

    if df.empty:

        return {
            "total_roads": 0,
            "blocked": 0,
            "high_risk": 0,
            "partially_disrupted": 0,
            "normal": 0,
            "accessibility": 0,
        }

    total = len(df)

    blocked = int(
        (
            df["status"]
            == "BLOCKED"
        ).sum()
    )

    high_risk = int(
        (
            df["status"]
            == "HIGH RISK"
        ).sum()
    )

    partial = int(
        (
            df["status"]
            == "PARTIALLY DISRUPTED"
        ).sum()
    )

    normal = int(
        (
            df["status"]
            == "NORMAL"
        ).sum()
    )

    accessibility = round(
        (
            (
                total
                - blocked
            )
            / total
        )
        * 100,
        1,
    )

    return {
        "total_roads": total,
        "blocked": blocked,
        "high_risk": high_risk,
        "partially_disrupted": partial,
        "normal": normal,
        "accessibility": accessibility,
    }


# ============================================================
# 23. UPDATE VEHICLE GPS
# ============================================================

def update_vehicle_gps(
    vehicle_id,
    lat,
    lon,
    speed=None,
    heading=None,
):

    vehicles = (
        st.session_state.vehicles
    )

    for vehicle in vehicles:

        if (
            vehicle["id"]
            == vehicle_id
        ):

            vehicle["lat"] = float(
                lat
            )

            vehicle["lon"] = float(
                lon
            )

            if speed is not None:

                vehicle[
                    "speed_kmh"
                ] = float(
                    speed
                )

            if heading is not None:

                vehicle[
                    "heading"
                ] = float(
                    heading
                )

            vehicle[
                "last_update"
            ] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            vehicle[
                "gps_status"
            ] = "CONNECTED"

            return True

    return False