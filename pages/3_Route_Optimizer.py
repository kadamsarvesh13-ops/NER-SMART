import math
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import folium

from streamlit_folium import st_folium

from common import (
    init_state,
    build_live_graph,
    edges_dataframe,
    ROADS,
    risk_to_status,
)


# ============================================================
# INITIALIZE SHARED NER SMART STATE
# ============================================================

# IMPORTANT:
# init_state() already handles the shared application state.
# Do NOT call st.set_page_config() again if common.py already does.
init_state()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🛣️ NER Smart — AI Route Optimizer")

st.caption(
    "Real-road logistics optimization using OpenStreetMap, "
    "OSRM routing, live weather intelligence and the "
    "NER Smart AI risk engine."
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .route-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 12px;
    }

    .route-title {
        font-size: 21px;
        font-weight: 700;
    }

    .small-text {
        font-size: 13px;
        opacity: 0.75;
    }

    .status-normal {
        font-weight: 700;
    }

    .status-risk {
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NER LOCATIONS
# ============================================================

NER_CENTER = [25.7, 93.0]


AVAILABLE_LOCATIONS = {
    "Guwahati": {
        "lat": 26.1445,
        "lon": 91.7362,
        "state": "Assam",
    },
    "Dibrugarh": {
        "lat": 27.4728,
        "lon": 94.9120,
        "state": "Assam",
    },
    "Tinsukia": {
        "lat": 27.4922,
        "lon": 95.3468,
        "state": "Assam",
    },
    "Silchar": {
        "lat": 24.8333,
        "lon": 92.7789,
        "state": "Assam",
    },
    "Tezpur": {
        "lat": 26.6528,
        "lon": 92.7926,
        "state": "Assam",
    },
    "Itanagar": {
        "lat": 27.0844,
        "lon": 93.6053,
        "state": "Arunachal Pradesh",
    },
    "Tawang": {
        "lat": 27.5860,
        "lon": 91.8650,
        "state": "Arunachal Pradesh",
    },
    "Pasighat": {
        "lat": 28.0660,
        "lon": 95.3260,
        "state": "Arunachal Pradesh",
    },
    "Shillong": {
        "lat": 25.5788,
        "lon": 91.8933,
        "state": "Meghalaya",
    },
    "Tura": {
        "lat": 25.5146,
        "lon": 90.2030,
        "state": "Meghalaya",
    },
    "Imphal": {
        "lat": 24.8170,
        "lon": 93.9368,
        "state": "Manipur",
    },
    "Aizawl": {
        "lat": 23.7271,
        "lon": 92.7176,
        "state": "Mizoram",
    },
    "Lunglei": {
        "lat": 22.8897,
        "lon": 92.7460,
        "state": "Mizoram",
    },
    "Kohima": {
        "lat": 25.6751,
        "lon": 94.1086,
        "state": "Nagaland",
    },
    "Dimapur": {
        "lat": 25.8629,
        "lon": 93.7538,
        "state": "Nagaland",
    },
    "Agartala": {
        "lat": 23.8315,
        "lon": 91.2868,
        "state": "Tripura",
    },
    "Gangtok": {
        "lat": 27.3389,
        "lon": 88.6065,
        "state": "Sikkim",
    },
}


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "route_origin": None,
    "route_destination": None,
    "route_origin_name": "",
    "route_destination_name": "",
    "calculated_routes": None,
    "route_cargo": "Medicines",
    "route_timestamp": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CARGO CONFIGURATION
# ============================================================

CARGO_CONFIG = {
    "Medicines": {
        "priority": 1,
        "risk_weight": 1.70,
        "time_weight": 1.50,
        "distance_weight": 0.20,
    },
    "Emergency Supplies": {
        "priority": 1,
        "risk_weight": 1.70,
        "time_weight": 1.50,
        "distance_weight": 0.20,
    },
    "Food & Essential Commodities": {
        "priority": 2,
        "risk_weight": 1.30,
        "time_weight": 1.25,
        "distance_weight": 0.25,
    },
    "Agricultural Produce": {
        "priority": 3,
        "risk_weight": 0.90,
        "time_weight": 1.10,
        "distance_weight": 0.30,
    },
    "Fuel": {
        "priority": 2,
        "risk_weight": 1.40,
        "time_weight": 1.30,
        "distance_weight": 0.25,
    },
    "Construction Materials": {
        "priority": 3,
        "risk_weight": 0.70,
        "time_weight": 0.90,
        "distance_weight": 0.40,
    },
    "General Cargo": {
        "priority": 3,
        "risk_weight": 0.60,
        "time_weight": 0.85,
        "distance_weight": 0.45,
    },
}


# ============================================================
# LOCATION HELPERS
# ============================================================

def find_known_location(lat, lon, tolerance_km=35):
    """
    Finds the nearest known NER location.

    Used to connect arbitrary searched locations to the
    existing NER Smart network model.
    """

    nearest_name = None
    nearest_distance = float("inf")

    for name, data in AVAILABLE_LOCATIONS.items():

        distance = haversine_km(
            lat,
            lon,
            data["lat"],
            data["lon"],
        )

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_name = name

    if nearest_distance <= tolerance_km:
        return nearest_name, nearest_distance

    return None, nearest_distance


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance between two coordinates.
    """

    radius = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * radius * math.asin(
        math.sqrt(a)
    )


# ============================================================
# NOMINATIM SEARCH
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def search_location(query):

    if not query or not query.strip():
        return []

    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": f"{query}, India",
        "format": "json",
        "limit": 5,
        "countrycodes": "in",
        "addressdetails": 1,
    }

    headers = {
        "User-Agent": "NER-Smart-Logistics/1.0",
    }

    try:

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=15,
        )

        response.raise_for_status()

        data = response.json()

        results = []

        for item in data:

            results.append(
                {
                    "name": item.get(
                        "display_name",
                        query,
                    ),
                    "lat": float(
                        item["lat"]
                    ),
                    "lon": float(
                        item["lon"]
                    ),
                }
            )

        return results

    except Exception:
        return []


# ============================================================
# OSRM ROUTING
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_osrm_routes(
    origin,
    destination,
):

    lat1, lon1 = origin
    lat2, lon2 = destination

    url = (
        "https://router.project-osrm.org/"
        "route/v1/driving/"
        f"{lon1},{lat1};{lon2},{lat2}"
    )

    params = {
        "alternatives": "true",
        "overview": "full",
        "steps": "true",
        "geometries": "geojson",
    }

    headers = {
        "User-Agent": "NER-Smart-Logistics/1.0",
    }

    try:

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=45,
        )

        response.raise_for_status()

        data = response.json()

        if data.get("code") != "Ok":
            return []

        return data.get(
            "routes",
            [],
        )

    except Exception:
        return []


# ============================================================
# WEATHER
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def get_weather(lat, lon):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": (
            "temperature_2m,"
            "precipitation,"
            "rain,"
            "showers,"
            "wind_speed_10m,"
            "weather_code"
        ),
        "timezone": "auto",
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=15,
        )

        response.raise_for_status()

        return response.json().get(
            "current",
            {},
        )

    except Exception:
        return {}


# ============================================================
# WEATHER RISK
# ============================================================

def weather_risk_score(weather):

    if not weather:
        return 20.0

    rain = float(
        weather.get("rain", 0) or 0
    )

    precipitation = float(
        weather.get("precipitation", 0) or 0
    )

    wind = float(
        weather.get(
            "wind_speed_10m",
            0,
        )
        or 0
    )

    effective_rain = max(
        rain,
        precipitation,
    )

    score = 10.0

    if effective_rain >= 50:
        score += 55
    elif effective_rain >= 25:
        score += 40
    elif effective_rain >= 10:
        score += 25
    elif effective_rain >= 5:
        score += 10

    if wind >= 60:
        score += 30
    elif wind >= 40:
        score += 20
    elif wind >= 25:
        score += 10

    return min(
        score,
        100,
    )


# ============================================================
# WEATHER SEVERITY
# ============================================================

def weather_status(score):

    if score >= 80:
        return "🔴 Severe"

    if score >= 60:
        return "🟠 High"

    if score >= 30:
        return "🟡 Moderate"

    return "🟢 Normal"


# ============================================================
# LIVE NETWORK RISK
# ============================================================

def get_live_network_intelligence():

    try:

        graph = build_live_graph()

        if graph is None:
            return None, None

        df = edges_dataframe(graph)

        if df is None:
            return graph, None

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        return graph, df

    except Exception:
        return None, None


# ============================================================
# GENERIC NUMERIC EXTRACTION
# ============================================================

def numeric_value(row, possible_names, default=0):

    for name in possible_names:

        if name in row.index:

            try:
                value = float(
                    row[name]
                )

                if math.isfinite(value):
                    return value

            except Exception:
                pass

    return float(default)


# ============================================================
# NETWORK RISK SUMMARY
# ============================================================

def calculate_network_risk(network_df):

    if (
        network_df is None
        or network_df.empty
    ):
        return 20.0, 0, 0

    risk_values = []

    blocked = 0
    high_risk = 0

    for _, row in network_df.iterrows():

        risk = numeric_value(
            row,
            [
                "risk",
                "risk_score",
                "Risk",
                "Risk Score",
            ],
            20,
        )

        risk = max(
            0,
            min(
                100,
                risk,
            ),
        )

        risk_values.append(risk)

        status = str(
            row.get(
                "status",
                row.get(
                    "Status",
                    "",
                ),
            )
        ).upper()

        if (
            "BLOCK" in status
            or risk >= 80
        ):
            blocked += 1

        elif (
            "HIGH" in status
            or risk >= 60
        ):
            high_risk += 1

    if not risk_values:
        return 20.0, blocked, high_risk

    return (
        sum(risk_values)
        / len(risk_values),
        blocked,
        high_risk,
    )


# ============================================================
# ROUTE-SPECIFIC LIVE RISK
# ============================================================

def calculate_route_live_risk(
    route,
    origin,
    destination,
    origin_weather,
    destination_weather,
    network_df,
):
    """
    Combines:
      1. endpoint weather risk
      2. live network risk from common.py
      3. route length exposure
      4. NER corridor proximity

    This is an operational route-risk estimate, not a claim
    that OSRM itself knows about road closures.
    """

    weather_origin = weather_risk_score(
        origin_weather
    )

    weather_destination = weather_risk_score(
        destination_weather
    )

    weather_component = (
        weather_origin
        + weather_destination
    ) / 2

    network_component = 20.0

    if (
        network_df is not None
        and not network_df.empty
    ):

        network_risk, _, _ = (
            calculate_network_risk(
                network_df
            )
        )

        network_component = network_risk

    # --------------------------------------------------------
    # Try to identify known NER endpoints
    # --------------------------------------------------------

    origin_name, origin_distance = (
        find_known_location(
            origin[0],
            origin[1],
        )
    )

    destination_name, destination_distance = (
        find_known_location(
            destination[0],
            destination[1],
        )
    )

    corridor_component = network_component

    # If route corresponds closely to a known NER corridor,
    # inspect the live graph for that corridor.
    if (
        origin_name
        and destination_name
        and network_df is not None
        and not network_df.empty
    ):

        text_df = network_df.astype(
            str
        )

        candidate_rows = []

        for idx, row in text_df.iterrows():

            joined = " ".join(
                row.astype(str).tolist()
            ).lower()

            if (
                origin_name.lower()
                in joined
                and destination_name.lower()
                in joined
            ):
                candidate_rows.append(
                    idx
                )

            elif (
                origin_name.lower()
                in joined
                or destination_name.lower()
                in joined
            ):
                candidate_rows.append(
                    idx
                )

        if candidate_rows:

            selected = network_df.loc[
                candidate_rows
            ]

            values = []

            for _, row in selected.iterrows():

                values.append(
                    numeric_value(
                        row,
                        [
                            "risk",
                            "risk_score",
                            "Risk",
                            "Risk Score",
                        ],
                        network_component,
                    )
                )

            if values:
                corridor_component = max(
                    values
                )

    # --------------------------------------------------------
    # Long routes have greater environmental exposure
    # --------------------------------------------------------

    distance_km = (
        route.get(
            "distance",
            0,
        )
        / 1000
    )

    exposure = min(
        distance_km / 1000 * 8,
        12,
    )

    # --------------------------------------------------------
    # Combined live risk
    # --------------------------------------------------------

    final_risk = (
        weather_component * 0.40
        + corridor_component * 0.45
        + exposure * 0.15
    )

    return max(
        0,
        min(
            100,
            final_risk,
        ),
    )


# ============================================================
# ROUTE SCORE
# ============================================================

def calculate_route_score(
    distance_km,
    duration_hours,
    risk,
    cargo,
):
    """
    Lower score = better route.

    Cargo priority changes the relative importance of
    safety and ETA.
    """

    config = CARGO_CONFIG[
        cargo
    ]

    return (
        risk
        * config["risk_weight"]
        + duration_hours
        * 8
        * config["time_weight"]
        + distance_km
        * 0.08
        * config["distance_weight"]
    )


# ============================================================
# ROUTE STATUS
# ============================================================

def route_status(risk):

    if risk >= 80:
        return (
            "🔴 BLOCKED / CRITICAL",
            "Avoid if possible",
        )

    if risk >= 60:
        return (
            "🟠 HIGH RISK",
            "Use caution and monitor",
        )

    if risk >= 30:
        return (
            "🟡 PARTIALLY DISRUPTED",
            "Proceed with monitoring",
        )

    return (
        "🟢 NORMAL",
        "Operational conditions",
    )


# ============================================================
# ROUTE STEP SUMMARY
# ============================================================

def get_route_steps(route):

    steps = []

    for leg in route.get(
        "legs",
        [],
    ):

        for step in leg.get(
            "steps",
            [],
        ):

            name = (
                step.get(
                    "name"
                )
                or "Unnamed road"
            )

            distance = (
                step.get(
                    "distance",
                    0,
                )
                / 1000
            )

            duration = (
                step.get(
                    "duration",
                    0,
                )
                / 60
            )

            maneuver = step.get(
                "maneuver",
                {},
            )

            instruction = maneuver.get(
                "type",
                "continue",
            )

            steps.append(
                {
                    "Road": name,
                    "Distance (km)": round(
                        distance,
                        2,
                    ),
                    "Time (min)": round(
                        duration,
                        1,
                    ),
                    "Action": instruction,
                }
            )

    return steps


# ============================================================
# ROUTE MAP
# ============================================================

def build_route_map(
    route_data,
    origin,
    destination,
    origin_name,
    destination_name,
):

    route_map = folium.Map(
        location=[
            (
                origin[0]
                + destination[0]
            )
            / 2,
            (
                origin[1]
                + destination[1]
            )
            / 2,
        ],
        zoom_start=7,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # --------------------------------------------------------
    # Collect every point drawn on the map so we can fit the
    # viewport to them at the end. Without this, the map stays
    # at a fixed zoom/center and long routes can render mostly
    # (or entirely) off-screen.
    # --------------------------------------------------------

    bounds_points = [
        list(origin),
        list(destination),
    ]

    # --------------------------------------------------------
    # Alternate route layers
    # --------------------------------------------------------

    for item in route_data:

        route = item["route"]

        coordinates = (
            route
            .get("geometry", {})
            .get("coordinates", [])
        )

        if not coordinates:
            continue

        latlon = [
            [
                point[1],
                point[0],
            ]
            for point in coordinates
        ]

        bounds_points.extend(latlon)

        index = item["index"]

        if item["recommended"]:

            color = "green"
            weight = 8
            opacity = 0.95

            label = (
                "🧠 AI RECOMMENDED"
            )

        elif item["fastest"]:

            color = "blue"
            weight = 5
            opacity = 0.70

            label = "⚡ FASTEST"

        elif item["safest"]:

            color = "purple"
            weight = 5
            opacity = 0.70

            label = "🛡️ LOWEST RISK"

        else:

            color = "orange"
            weight = 4
            opacity = 0.55

            label = "🔀 ALTERNATIVE"

        tooltip = (
            f"{label} | "
            f"{item['distance_km']:.1f} km | "
            f"{item['duration_hours']:.2f} h | "
            f"Risk {item['risk']:.0f}%"
        )

        folium.PolyLine(
            latlon,
            color=color,
            weight=weight,
            opacity=opacity,
            tooltip=tooltip,
        ).add_to(route_map)

    # --------------------------------------------------------
    # Origin
    # --------------------------------------------------------

    folium.Marker(
        location=origin,
        tooltip="📍 ORIGIN",
        popup=folium.Popup(
            f"""
            <b>ORIGIN</b><br>
            {origin_name}
            """,
            max_width=350,
        ),
        icon=folium.Icon(
            color="green",
            icon="play",
        ),
    ).add_to(route_map)

    # --------------------------------------------------------
    # Destination
    # --------------------------------------------------------

    folium.Marker(
        location=destination,
        tooltip="🎯 DESTINATION",
        popup=folium.Popup(
            f"""
            <b>DESTINATION</b><br>
            {destination_name}
            """,
            max_width=350,
        ),
        icon=folium.Icon(
            color="red",
            icon="flag",
        ),
    ).add_to(route_map)

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend = """
    <div style="
        position: fixed;
        top: 90px;
        right: 12px;
        z-index: 9999;
        background: white;
        padding: 12px;
        border: 2px solid grey;
        border-radius: 8px;
        font-size: 13px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.3);
    ">
        <b>NER SMART ROUTES</b><br>
        🟢 AI Recommended<br>
        🔵 Fastest<br>
        🟣 Lowest Risk<br>
        🟠 Alternative<br>
        📍 Origin<br>
        🎯 Destination
    </div>
    """

    route_map.get_root().html.add_child(
        folium.Element(legend)
    )

    # --------------------------------------------------------
    # Fit the viewport to the actual route(s) instead of relying
    # on a fixed zoom level, so the route is always in view.
    # --------------------------------------------------------

    if len(bounds_points) >= 2:

        route_map.fit_bounds(
            bounds_points,
            padding=(30, 30),
        )

    return route_map


# ============================================================
# LOCATION SELECTION MAP
# ============================================================

st.subheader(
    "🗺️ Select Origin & Destination"
)

st.info(
    "Click a blue NER location or click anywhere on the "
    "map. First click = Origin, second click = Destination."
)


selection_map = folium.Map(
    location=NER_CENTER,
    zoom_start=6,
    tiles="OpenStreetMap",
    control_scale=True,
)


# ------------------------------------------------------------
# Location pins
# ------------------------------------------------------------

for name, location in (
    AVAILABLE_LOCATIONS.items()
):

    folium.CircleMarker(
        location=[
            location["lat"],
            location["lon"],
        ],
        radius=7,
        color="blue",
        fill=True,
        fill_opacity=0.9,
        tooltip=(
            f"{name}, "
            f"{location['state']}"
        ),
        popup=folium.Popup(
            f"""
            <b>{name}</b><br>
            State: {location['state']}<br>
            Latitude: {location['lat']:.5f}<br>
            Longitude: {location['lon']:.5f}
            """,
            max_width=300,
        ),
    ).add_to(selection_map)


# ------------------------------------------------------------
# Existing origin
# ------------------------------------------------------------

if st.session_state.route_origin:

    folium.Marker(
        location=st.session_state.route_origin,
        tooltip="📍 ORIGIN",
        popup=(
            f"<b>ORIGIN</b><br>"
            f"{st.session_state.route_origin_name}"
        ),
        icon=folium.Icon(
            color="green",
            icon="play",
        ),
    ).add_to(selection_map)


# ------------------------------------------------------------
# Existing destination
# ------------------------------------------------------------

if st.session_state.route_destination:

    folium.Marker(
        location=st.session_state.route_destination,
        tooltip="🎯 DESTINATION",
        popup=(
            f"<b>DESTINATION</b><br>"
            f"{st.session_state.route_destination_name}"
        ),
        icon=folium.Icon(
            color="red",
            icon="flag",
        ),
    ).add_to(selection_map)


map_data = st_folium(
    selection_map,
    width=None,
    height=580,
    returned_objects=[
        "last_clicked",
    ],
    key="route_selection_map",
)


# ============================================================
# MAP CLICK
# ============================================================

if map_data:

    clicked = map_data.get(
        "last_clicked"
    )

    if clicked:

        lat = clicked.get("lat")
        lon = clicked.get("lng")

        if (
            lat is not None
            and lon is not None
        ):

            if (
                st.session_state.route_origin
                is None
            ):

                st.session_state.route_origin = (
                    float(lat),
                    float(lon),
                )

                known, _ = find_known_location(
                    float(lat),
                    float(lon),
                )

                if known:
                    st.session_state.route_origin_name = (
                        known
                    )
                else:
                    st.session_state.route_origin_name = (
                        f"Selected location "
                        f"({lat:.5f}, {lon:.5f})"
                    )

                st.session_state.calculated_routes = None

                st.rerun()

            elif (
                st.session_state.route_destination
                is None
            ):

                st.session_state.route_destination = (
                    float(lat),
                    float(lon),
                )

                known, _ = find_known_location(
                    float(lat),
                    float(lon),
                )

                if known:
                    st.session_state.route_destination_name = (
                        known
                    )
                else:
                    st.session_state.route_destination_name = (
                        f"Selected location "
                        f"({lat:.5f}, {lon:.5f})"
                    )

                st.session_state.calculated_routes = None

                st.rerun()


# ============================================================
# SEARCH LOCATIONS
# ============================================================

st.divider()

st.subheader(
    "🔎 Search Locations"
)

search_col1, search_col2 = st.columns(2)


# ============================================================
# ORIGIN SEARCH
# ============================================================

with search_col1:

    st.markdown(
        "### 📍 Origin"
    )

    origin_query = st.text_input(
        "Search origin",
        placeholder="Guwahati",
        key="route_origin_search",
    )

    if origin_query:

        results = search_location(
            origin_query
        )

        if results:

            selected_origin = st.selectbox(
                "Search results",
                results,
                format_func=lambda x: x[
                    "name"
                ],
                key="route_origin_result",
            )

            if st.button(
                "📍 Set Origin",
                key="route_set_origin",
                use_container_width=True,
            ):

                st.session_state.route_origin = (
                    selected_origin["lat"],
                    selected_origin["lon"],
                )

                st.session_state.route_origin_name = (
                    selected_origin["name"]
                )

                st.session_state.calculated_routes = None

                st.rerun()

        else:

            st.warning(
                "No location found."
            )


# ============================================================
# DESTINATION SEARCH
# ============================================================

with search_col2:

    st.markdown(
        "### 🎯 Destination"
    )

    destination_query = st.text_input(
        "Search destination",
        placeholder="Aizawl",
        key="route_destination_search",
    )

    if destination_query:

        results = search_location(
            destination_query
        )

        if results:

            selected_destination = st.selectbox(
                "Search results",
                results,
                format_func=lambda x: x[
                    "name"
                ],
                key="route_destination_result",
            )

            if st.button(
                "🎯 Set Destination",
                key="route_set_destination",
                use_container_width=True,
            ):

                st.session_state.route_destination = (
                    selected_destination["lat"],
                    selected_destination["lon"],
                )

                st.session_state.route_destination_name = (
                    selected_destination["name"]
                )

                st.session_state.calculated_routes = None

                st.rerun()

        else:

            st.warning(
                "No location found."
            )


# ============================================================
# QUICK SELECT
# ============================================================

st.divider()

with st.expander(
    "⚡ Quick Select NER Locations"
):

    q1, q2 = st.columns(2)

    with q1:

        quick_origin = st.selectbox(
            "Origin",
            [
                "Select"
            ]
            + list(
                AVAILABLE_LOCATIONS.keys()
            ),
            key="quick_route_origin",
        )

        if st.button(
            "Set Quick Origin",
            key="quick_route_origin_button",
            use_container_width=True,
        ):

            if quick_origin != "Select":

                data = AVAILABLE_LOCATIONS[
                    quick_origin
                ]

                st.session_state.route_origin = (
                    data["lat"],
                    data["lon"],
                )

                st.session_state.route_origin_name = (
                    f"{quick_origin}, "
                    f"{data['state']}"
                )

                st.session_state.calculated_routes = None

                st.rerun()

    with q2:

        quick_destination = st.selectbox(
            "Destination",
            [
                "Select"
            ]
            + list(
                AVAILABLE_LOCATIONS.keys()
            ),
            key="quick_route_destination",
        )

        if st.button(
            "Set Quick Destination",
            key="quick_route_destination_button",
            use_container_width=True,
        ):

            if quick_destination != "Select":

                data = AVAILABLE_LOCATIONS[
                    quick_destination
                ]

                st.session_state.route_destination = (
                    data["lat"],
                    data["lon"],
                )

                st.session_state.route_destination_name = (
                    f"{quick_destination}, "
                    f"{data['state']}"
                )

                st.session_state.calculated_routes = None

                st.rerun()


# ============================================================
# SELECTED LOCATIONS
# ============================================================

st.divider()

s1, s2 = st.columns(2)

with s1:

    if st.session_state.route_origin:

        st.success(
            f"📍 **Origin:** "
            f"{st.session_state.route_origin_name}"
        )

    else:

        st.warning(
            "📍 Origin not selected"
        )


with s2:

    if st.session_state.route_destination:

        st.error(
            f"🎯 **Destination:** "
            f"{st.session_state.route_destination_name}"
        )

    else:

        st.warning(
            "🎯 Destination not selected"
        )


# ============================================================
# LOGISTICS CONTROLS
# ============================================================

st.divider()

c1, c2, c3 = st.columns(
    [1.4, 1, 1]
)


with c1:

    cargo = st.selectbox(
        "📦 Cargo Type",
        list(
            CARGO_CONFIG.keys()
        ),
        index=list(
            CARGO_CONFIG.keys()
        ).index(
            st.session_state.route_cargo
        )
        if st.session_state.route_cargo
        in CARGO_CONFIG
        else 0,
    )


with c2:

    reset_button = st.button(
        "🔄 Reset",
        use_container_width=True,
    )


with c3:

    calculate_button = st.button(
        "🧠 Calculate AI Route",
        type="primary",
        use_container_width=True,
    )


if reset_button:

    st.session_state.route_origin = None
    st.session_state.route_destination = None
    st.session_state.route_origin_name = ""
    st.session_state.route_destination_name = ""
    st.session_state.calculated_routes = None
    st.session_state.route_timestamp = None

    st.rerun()


# ============================================================
# CALCULATE ROUTE
# ============================================================

if calculate_button:

    origin = st.session_state.route_origin
    destination = st.session_state.route_destination

    if not origin:

        st.error(
            "❌ Please select an origin."
        )

        st.stop()

    if not destination:

        st.error(
            "❌ Please select a destination."
        )

        st.stop()

    same_location = (
        abs(
            origin[0]
            - destination[0]
        )
        < 0.00001
        and abs(
            origin[1]
            - destination[1]
        )
        < 0.00001
    )

    if same_location:

        st.error(
            "❌ Origin and destination cannot be the same."
        )

        st.stop()

    st.session_state.route_cargo = cargo

    with st.spinner(
        "🧠 NER Smart AI is analyzing road routes, "
        "weather and live network risk..."
    ):

        osrm_routes = get_osrm_routes(
            origin,
            destination,
        )

        origin_weather = get_weather(
            origin[0],
            origin[1],
        )

        destination_weather = get_weather(
            destination[0],
            destination[1],
        )

        _, network_df = (
            get_live_network_intelligence()
        )

    if not osrm_routes:

        st.error(
            "❌ No drivable road route was returned by OSRM."
        )

        st.info(
            "Try another origin/destination or check your "
            "internet connection."
        )

        st.stop()

    # --------------------------------------------------------
    # Build route intelligence
    # --------------------------------------------------------

    route_data = []

    for index, route in enumerate(
        osrm_routes
    ):

        distance_km = (
            float(
                route.get(
                    "distance",
                    0,
                )
            )
            / 1000
        )

        duration_hours = (
            float(
                route.get(
                    "duration",
                    0,
                )
            )
            / 3600
        )

        live_risk = calculate_route_live_risk(
            route=route,
            origin=origin,
            destination=destination,
            origin_weather=origin_weather,
            destination_weather=destination_weather,
            network_df=network_df,
        )

        score = calculate_route_score(
            distance_km=distance_km,
            duration_hours=duration_hours,
            risk=live_risk,
            cargo=cargo,
        )

        status, action = route_status(
            live_risk
        )

        route_data.append(
            {
                "index": index + 1,
                "route": route,
                "distance_km": distance_km,
                "duration_hours": duration_hours,
                "risk": live_risk,
                "score": score,
                "status": status,
                "action": action,
                "recommended": False,
                "fastest": False,
                "safest": False,
            }
        )

    # --------------------------------------------------------
    # Select routes
    # --------------------------------------------------------

    recommended = min(
        route_data,
        key=lambda x: x["score"],
    )

    fastest = min(
        route_data,
        key=lambda x: x[
            "duration_hours"
        ],
    )

    safest = min(
        route_data,
        key=lambda x: x["risk"],
    )

    recommended["recommended"] = True
    fastest["fastest"] = True
    safest["safest"] = True

    st.session_state.calculated_routes = (
        route_data
    )

    st.session_state.route_timestamp = (
        datetime.now().strftime(
            "%d %b %Y, %H:%M:%S"
        )
    )

    st.success(
        f"✅ AI analysis completed — "
        f"{len(route_data)} road route(s) analyzed."
    )


# ============================================================
# RESULTS
# ============================================================

if st.session_state.calculated_routes:

    route_data = (
        st.session_state.calculated_routes
    )

    recommended = next(
        x
        for x in route_data
        if x["recommended"]
    )

    fastest = next(
        x
        for x in route_data
        if x["fastest"]
    )

    safest = next(
        x
        for x in route_data
        if x["safest"]
    )

    # --------------------------------------------------------
    # OSRM's public routing server frequently returns only one
    # or two genuinely distinct road paths between two points
    # (especially across the sparser NER highway network). When
    # that happens, the single best path can legitimately win
    # every category at once, so "AI Recommended", "Fastest"
    # and "Lowest Risk" end up pointing at the *same* route and
    # show identical numbers below — that is expected, not a
    # bug, and we flag it clearly instead of implying three
    # different roads exist.
    # --------------------------------------------------------

    routes_converged = (
        recommended["index"] == fastest["index"]
        and recommended["index"] == safest["index"]
    )

    origin = (
        st.session_state.route_origin
    )

    destination = (
        st.session_state.route_destination
    )

    cargo = (
        st.session_state.route_cargo
    )

    # ========================================================
    # MAIN RECOMMENDATION
    # ========================================================

    st.divider()

    st.header(
        "🧠 AI Route Recommendation"
    )

    if recommended["risk"] >= 80:

        st.error(
            "⚠️ AI WARNING: The selected route has "
            "critical environmental/network risk."
        )

    elif recommended["risk"] >= 60:

        st.warning(
            "⚠️ Elevated route risk detected. "
            "Monitor conditions before dispatch."
        )

    else:

        st.success(
            "✅ Recommended route is currently within "
            "operational risk limits."
        )

    # ========================================================
    # ROUTE METRICS
    # ========================================================

    r1, r2, r3, r4 = st.columns(4)

    with r1:

        st.metric(
            "📏 Distance",
            f"{recommended['distance_km']:.1f} km",
        )

    with r2:

        st.metric(
            "⏱️ ETA",
            f"{recommended['duration_hours']:.2f} h",
        )

    with r3:

        st.metric(
            "⚠️ Live Risk",
            f"{recommended['risk']:.0f}%",
        )

    with r4:

        st.metric(
            "📦 Cargo",
            cargo,
        )

    # ========================================================
    # STATUS
    # ========================================================

    st.info(
        f"""
        **Route status:** {recommended['status']}

        **Operational guidance:** {recommended['action']}

        **Analysis timestamp:** {
            st.session_state.route_timestamp
        }
        """
    )

    # ========================================================
    # ROUTE MAP
    # ========================================================

    st.divider()

    st.subheader(
        "🗺️ Live Road-Based Route Map"
    )

    route_map = build_route_map(
        route_data=route_data,
        origin=origin,
        destination=destination,
        origin_name=(
            st.session_state.route_origin_name
        ),
        destination_name=(
            st.session_state.route_destination_name
        ),
    )

    st_folium(
        route_map,
        width=None,
        height=680,
        key="final_route_map",
    )

    # ========================================================
    # ROUTE COMPARISON
    # ========================================================

    st.divider()

    st.subheader(
        "📊 AI Route Comparison"
    )

    comparison_rows = []

    for item in route_data:

        if item["recommended"]:
            route_type = "🧠 AI Recommended"

        elif item["fastest"]:
            route_type = "⚡ Fastest"

        elif item["safest"]:
            route_type = "🛡️ Lowest Risk"

        else:
            route_type = "🔀 Alternative"

        comparison_rows.append(
            {
                "Route": (
                    f"Route {item['index']}"
                ),
                "Type": route_type,
                "Distance (km)": round(
                    item["distance_km"],
                    1,
                ),
                "ETA (hours)": round(
                    item["duration_hours"],
                    2,
                ),
                "Live Risk (%)": round(
                    item["risk"],
                    1,
                ),
                "AI Score": round(
                    item["score"],
                    2,
                ),
                "Status": item["status"],
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # THREE ROUTE CARDS
    # ========================================================

    st.divider()

    if routes_converged:

        st.info(
            "🧭 OSRM found only "
            f"{len(route_data)} distinct road route(s) "
            "between these two points, and Route "
            f"{recommended['index']} scores best on speed, "
            "distance and risk simultaneously — so AI "
            "Recommended, Fastest and Lowest Risk all refer "
            "to the same road below. This is expected when "
            "few alternative roads exist between the chosen "
            "points, not a calculation error."
        )

    a, b, c = st.columns(3)

    # --------------------------------------------------------
    # Recommended
    # --------------------------------------------------------

    with a:

        st.markdown(
            "### 🧠 AI Recommended"
        )

        st.metric(
            "Distance",
            f"{recommended['distance_km']:.1f} km",
        )

        st.metric(
            "ETA",
            f"{recommended['duration_hours']:.2f} h",
        )

        st.metric(
            "Risk",
            f"{recommended['risk']:.0f}%",
        )

        st.caption(
            recommended["status"]
        )

    # --------------------------------------------------------
    # Fastest
    # --------------------------------------------------------

    with b:

        st.markdown(
            "### ⚡ Fastest"
        )

        st.metric(
            "Distance",
            f"{fastest['distance_km']:.1f} km",
        )

        st.metric(
            "ETA",
            f"{fastest['duration_hours']:.2f} h",
        )

        st.metric(
            "Risk",
            f"{fastest['risk']:.0f}%",
        )

        st.caption(
            fastest["status"]
        )

    # --------------------------------------------------------
    # Safest
    # --------------------------------------------------------

    with c:

        st.markdown(
            "### 🛡️ Lowest Risk"
        )

        st.metric(
            "Distance",
            f"{safest['distance_km']:.1f} km",
        )

        st.metric(
            "ETA",
            f"{safest['duration_hours']:.2f} h",
        )

        st.metric(
            "Risk",
            f"{safest['risk']:.0f}%",
        )

        st.caption(
            safest["status"]
        )

    # ========================================================
    # AI EXPLANATION
    # ========================================================

    st.divider()

    st.subheader(
        "🧠 Why Did AI Select This Route?"
    )

    cargo_config = CARGO_CONFIG[
        cargo
    ]

    # NOTE: every line below is left-flush (no leading spaces).
    # Streamlit's markdown renderer treats 4+ leading spaces on
    # a line as a literal code block, which silently breaks HTML
    # rendering and throws off page layout — so nested elements
    # like <li> items must NOT be indented further than their
    # siblings inside an f-string passed to st.markdown().
    ai_decision_html = "\n".join([
        '<div class="route-card">',
        '<div class="route-title">AI Route Decision</div>',
        "<br>",
        f"<b>📍 Origin:</b> {st.session_state.route_origin_name}",
        "<br>",
        f"<b>🎯 Destination:</b> {st.session_state.route_destination_name}",
        "<br>",
        f"<b>📦 Cargo:</b> {cargo}",
        "<br>",
        f"<b>🚨 Cargo Priority:</b> {cargo_config['priority']}",
        "<br><br>",
        "The optimizer evaluates multiple road alternatives using:",
        "<ul>",
        "<li>Real road distance from OSRM</li>",
        "<li>Estimated driving time</li>",
        "<li>Current weather conditions</li>",
        "<li>Live NER Smart network risk</li>",
        "<li>Environmental exposure</li>",
        "<li>Cargo priority and safety requirements</li>",
        "</ul>",
        "The selected route has a current estimated risk of "
        f"<b>{recommended['risk']:.0f}%</b> and an estimated "
        "travel time of "
        f"<b>{recommended['duration_hours']:.2f} hours</b>.",
        "</div>",
    ])

    st.markdown(
        ai_decision_html,
        unsafe_allow_html=True,
    )

    # ========================================================
    # WEATHER INTELLIGENCE
    # ========================================================

    st.divider()

    st.subheader(
        "🌧️ Current Route Weather Intelligence"
    )

    origin_weather = get_weather(
        origin[0],
        origin[1],
    )

    destination_weather = get_weather(
        destination[0],
        destination[1],
    )

    weather1, weather2 = st.columns(2)

    with weather1:

        st.markdown(
            "### 📍 Origin Weather"
        )

        if origin_weather:

            temperature = origin_weather.get(
                "temperature_2m",
                0,
            )

            rain = origin_weather.get(
                "rain",
                0,
            )

            precipitation = origin_weather.get(
                "precipitation",
                0,
            )

            wind = origin_weather.get(
                "wind_speed_10m",
                0,
            )

            risk = weather_risk_score(
                origin_weather
            )

            st.metric(
                "Temperature",
                f"{temperature} °C",
            )

            st.metric(
                "Rain",
                f"{rain} mm",
            )

            st.metric(
                "Precipitation",
                f"{precipitation} mm",
            )

            st.metric(
                "Wind",
                f"{wind} km/h",
            )

            st.info(
                f"Weather risk: "
                f"{risk:.0f}% — "
                f"{weather_status(risk)}"
            )

        else:

            st.warning(
                "Weather data unavailable."
            )

    with weather2:

        st.markdown(
            "### 🎯 Destination Weather"
        )

        if destination_weather:

            temperature = destination_weather.get(
                "temperature_2m",
                0,
            )

            rain = destination_weather.get(
                "rain",
                0,
            )

            precipitation = destination_weather.get(
                "precipitation",
                0,
            )

            wind = destination_weather.get(
                "wind_speed_10m",
                0,
            )

            risk = weather_risk_score(
                destination_weather
            )

            st.metric(
                "Temperature",
                f"{temperature} °C",
            )

            st.metric(
                "Rain",
                f"{rain} mm",
            )

            st.metric(
                "Precipitation",
                f"{precipitation} mm",
            )

            st.metric(
                "Wind",
                f"{wind} km/h",
            )

            st.info(
                f"Weather risk: "
                f"{risk:.0f}% — "
                f"{weather_status(risk)}"
            )

        else:

            st.warning(
                "Weather data unavailable."
            )

    # ========================================================
    # LIVE NETWORK INTELLIGENCE
    # ========================================================

    st.divider()

    st.subheader(
        "🛰️ Live NER Network Intelligence"
    )

    _, network_df = (
        get_live_network_intelligence()
    )

    network_risk, blocked, high_risk = (
        calculate_network_risk(
            network_df
        )
    )

    n1, n2, n3 = st.columns(3)

    with n1:

        st.metric(
            "Network Risk",
            f"{network_risk:.0f}%",
        )

    with n2:

        st.metric(
            "Critical / Blocked",
            blocked,
        )

    with n3:

        st.metric(
            "High Risk Corridors",
            high_risk,
        )

    if (
        network_df is not None
        and not network_df.empty
    ):

        with st.expander(
            "🔍 View Live Network Risk Data"
        ):

            display_df = network_df.copy()

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

    # ========================================================
    # ROUTE DETAILS
    # ========================================================

    st.divider()

    st.subheader(
        "🧭 Road-by-Road Navigation Details"
    )

    for item in route_data:

        label = (
            "🧠 AI Recommended"
            if item["recommended"]
            else
            "⚡ Fastest"
            if item["fastest"]
            else
            "🛡️ Lowest Risk"
            if item["safest"]
            else
            "🔀 Alternative"
        )

        with st.expander(
            f"{label} — Route {item['index']}"
        ):

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            col1.metric(
                "Distance",
                f"{item['distance_km']:.1f} km",
            )

            col2.metric(
                "ETA",
                f"{item['duration_hours']:.2f} h",
            )

            col3.metric(
                "Live Risk",
                f"{item['risk']:.0f}%",
            )

            col4.metric(
                "AI Score",
                f"{item['score']:.1f}",
            )

            st.write(
                f"**Status:** "
                f"{item['status']}"
            )

            st.write(
                f"**Action:** "
                f"{item['action']}"
            )

            steps = get_route_steps(
                item["route"]
            )

            if steps:

                steps_df = pd.DataFrame(
                    steps
                )

                st.dataframe(
                    steps_df,
                    use_container_width=True,
                    hide_index=True,
                )

    # ========================================================
    # TECHNICAL ARCHITECTURE
    # ========================================================

    st.divider()

    with st.expander(
        "🧠 NER Smart Route Optimization Architecture"
    ):

        st.markdown(
            """
            ```text
                    ┌────────────────────────┐
                    │   User Origin/Dest.    │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  OpenStreetMap /       │
                    │  Nominatim Location    │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │        OSRM             │
                    │ Real Road Routing       │
                    │ Alternatives + ETA      │
                    └────────────┬───────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
          ┌──────────────────┐     ┌────────────────────┐
          │ Open-Meteo       │     │ common.py           │
          │ Weather          │     │ Live Risk Engine    │
          └────────┬─────────┘     └──────────┬─────────┘
                   │                          │
                   └────────────┬─────────────┘
                                ▼
                    ┌────────────────────────┐
                    │ Route Risk Intelligence│
                    │                        │
                    │ • Weather              │
                    │ • Road Risk             │
                    │ • Network Conditions    │
                    │ • Exposure              │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Cargo-Aware AI Scoring │
                    │                        │
                    │ Risk + ETA + Distance  │
                    │ + Cargo Priority       │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Recommended Route      │
                    │ Fastest Route          │
                    │ Lowest-Risk Route      │
                    └────────────────────────┘
            ```

            **Important:**

            OSRM supplies the actual road geometry and
            routing distance/ETA.

            Your `common.py` supplies NER Smart's live
            network-risk intelligence.

            Weather is supplied by Open-Meteo.

            The final route score combines these signals
            according to cargo requirements.
            """
        )

    # ========================================================
    # DATA SOURCES
    # ========================================================

    st.divider()

    st.caption(
        "Data sources: OpenStreetMap / Nominatim, OSRM and "
        "Open-Meteo. NER Smart live risk intelligence is "
        "provided by the shared common.py model. External "
        "routing/weather services should be replaced or "
        "hardened with official/production APIs for a "
        "production deployment."
    )