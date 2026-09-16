import math
import time
from datetime import datetime

import folium
import pandas as pd
import requests
import streamlit as st
from folium.plugins import AntPath
from streamlit_folium import st_folium

from common import (
    DISTRICTS,
    init_state,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NER Smart Fleet Tracking",
    page_icon="🚛",
    layout="wide",
)

init_state()


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .vehicle-card {
        padding: 12px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 8px;
    }

    .vehicle-title {
        font-size: 17px;
        font-weight: 700;
    }

    .vehicle-sub {
        font-size: 13px;
        color: #777;
    }

    .status-live {
        color: green;
        font-weight: 700;
    }

    .status-stop {
        color: orange;
        font-weight: 700;
    }

    .status-danger {
        color: red;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "tracking_api_url" not in st.session_state:
    st.session_state.tracking_api_url = ""

if "live_vehicles" not in st.session_state:
    st.session_state.live_vehicles = {}

if "vehicle_history" not in st.session_state:
    st.session_state.vehicle_history = {}


# ============================================================
# DISTANCE
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    radius_km = 6371

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return 2 * radius_km * math.asin(math.sqrt(a))


# ============================================================
# REAL ROAD ROUTING
# ============================================================

OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving"


@st.cache_data(ttl=120, show_spinner=False)
def get_driving_route(points):
    """
    Returns a road-following route from OSRM.

    OSRM returns actual road geometry instead of drawing
    straight lines between district coordinates.
    """
    if len(points) < 2:
        return None

    coordinates = ";".join(
        f"{lon:.6f},{lat:.6f}"
        for lat, lon in points
    )

    try:
        response = requests.get(
            f"{OSRM_ROUTE_URL}/{coordinates}",
            params={
                "overview": "full",
                "geometries": "geojson",
                "steps": "false",
                "alternatives": "false",
            },
            headers={
                "User-Agent": "NER-Fleet-Tracking/1.0",
            },
            timeout=15,
        )

        response.raise_for_status()

        payload = response.json()

        if payload.get("code") != "Ok":
            return None

        if not payload.get("routes"):
            return None

        route = payload["routes"][0]

        return {
            "coordinates": [
                (lat, lon)
                for lon, lat in route["geometry"]["coordinates"]
            ],
            "distance_km": route["distance"] / 1000,
            "duration_min": route["duration"] / 60,
        }

    except (
        requests.RequestException,
        KeyError,
        TypeError,
        ValueError,
    ):
        return None


def upcoming_route_points(vehicle):
    """
    Uses the live vehicle location as the route start point,
    then adds only upcoming district stops.
    """
    current_position = (
        vehicle["lat"],
        vehicle["lon"],
    )

    stops = [
        DISTRICTS[place]
        for place in vehicle.get("route", [])
        if place in DISTRICTS
    ]

    destination = vehicle.get("destination")

    if destination in DISTRICTS:
        destination_point = DISTRICTS[destination]

        if not stops or stops[-1] != destination_point:
            stops.append(destination_point)

    if not stops:
        return []

    nearest_stop_index = min(
        range(len(stops)),
        key=lambda index: haversine(
            *current_position,
            *stops[index],
        ),
    )

    remaining_stops = stops[nearest_stop_index + 1:]

    if not remaining_stops and destination in DISTRICTS:
        destination_point = DISTRICTS[destination]

        if haversine(
            *current_position,
            *destination_point,
        ) > 0.05:
            remaining_stops = [destination_point]

    return [
        current_position,
        *remaining_stops,
    ]


# ============================================================
# GPS API
# ============================================================

def get_gps_data(url):
    if not url:
        return None

    try:
        response = requests.get(
            url,
            timeout=10,
        )

        response.raise_for_status()

        return response.json()

    except Exception as error:
        st.error(f"GPS API Error: {error}")
        return None


# ============================================================
# NORMALIZE GPS
# ============================================================

def normalize_vehicles(data):
    if not data:
        return []

    vehicles = data.get("vehicles", [])
    result = []

    for vehicle in vehicles:
        try:
            route = vehicle.get("route", [])

            if not isinstance(route, list):
                route = []

            result.append(
                {
                    "id": str(vehicle["id"]),
                    "lat": float(vehicle["lat"]),
                    "lon": float(vehicle["lon"]),
                    "speed": float(vehicle.get("speed", 0)),
                    "heading": float(vehicle.get("heading", 0)),
                    "cargo": vehicle.get("cargo", "General"),
                    "status": vehicle.get(
                        "status",
                        "ACTIVE",
                    ).upper(),
                    "origin": vehicle.get("origin", "Unknown"),
                    "destination": vehicle.get(
                        "destination",
                        "Unknown",
                    ),
                    "route": route,
                    "timestamp": vehicle.get(
                        "timestamp",
                        "Unknown",
                    ),
                    "driver": vehicle.get(
                        "driver",
                        "Not Available",
                    ),
                    "vehicle_type": vehicle.get(
                        "vehicle_type",
                        "Truck",
                    ),
                    "fuel": vehicle.get("fuel"),
                }
            )

        except Exception:
            continue

    return result


# ============================================================
# DEMO GPS
# ============================================================

def demo_vehicles():
    return {
        "vehicles": [
            {
                "id": "NER-001",
                "lat": 26.1445,
                "lon": 91.7362,
                "speed": 42,
                "heading": 90,
                "cargo": "Medicines",
                "status": "ACTIVE",
                "origin": "Guwahati",
                "destination": "Aizawl",
                "route": [
                    "Guwahati",
                    "Silchar",
                    "Aizawl",
                ],
                "driver": "Driver-001",
                "vehicle_type": "Medical Supply Truck",
                "fuel": 78,
                "timestamp": datetime.now().isoformat(),
            },
            {
                "id": "NER-002",
                "lat": 25.5788,
                "lon": 91.8933,
                "speed": 35,
                "heading": 120,
                "cargo": "Food",
                "status": "ACTIVE",
                "origin": "Shillong",
                "destination": "Imphal",
                "route": [
                    "Shillong",
                    "Imphal",
                ],
                "driver": "Driver-002",
                "vehicle_type": "Cargo Truck",
                "fuel": 61,
                "timestamp": datetime.now().isoformat(),
            },
            {
                "id": "NER-003",
                "lat": 25.8629,
                "lon": 93.7538,
                "speed": 5,
                "heading": 180,
                "cargo": "Construction Materials",
                "status": "STOPPED",
                "origin": "Dimapur",
                "destination": "Itanagar",
                "route": [
                    "Dimapur",
                    "Itanagar",
                ],
                "driver": "Driver-003",
                "vehicle_type": "Heavy Truck",
                "fuel": 45,
                "timestamp": datetime.now().isoformat(),
            },
        ]
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚛 Fleet Control")

api_url = st.sidebar.text_input(
    "GPS API Endpoint",
    value=st.session_state.tracking_api_url,
    placeholder="https://your-api.com/api/vehicles",
)

st.session_state.tracking_api_url = api_url

demo = st.sidebar.checkbox(
    "Use demo fleet",
    value=True,
)

refresh = st.sidebar.selectbox(
    "Auto refresh",
    [
        "Off",
        "5 seconds",
        "10 seconds",
        "30 seconds",
    ],
    index=2,
)

st.sidebar.divider()

map_style = st.sidebar.selectbox(
    "Map style",
    [
        "OpenStreetMap",
        "OpenTopoMap",
    ],
)

show_roads = st.sidebar.checkbox(
    "Show road route",
    value=True,
)

show_trails = st.sidebar.checkbox(
    "Show vehicle trails",
    value=True,
)


# ============================================================
# GET DATA
# ============================================================

if api_url:
    gps = get_gps_data(api_url)
    vehicles = normalize_vehicles(gps)

elif demo:
    vehicles = normalize_vehicles(
        demo_vehicles()
    )

else:
    vehicles = []


# ============================================================
# STORE VEHICLE HISTORY
# ============================================================

for vehicle in vehicles:
    vehicle_id = vehicle["id"]

    st.session_state.live_vehicles[
        vehicle_id
    ] = vehicle

    if vehicle_id not in st.session_state.vehicle_history:
        st.session_state.vehicle_history[
            vehicle_id
        ] = []

    history = st.session_state.vehicle_history[
        vehicle_id
    ]

    history.append(
        (
            vehicle["lat"],
            vehicle["lon"],
        )
    )

    st.session_state.vehicle_history[
        vehicle_id
    ] = history[-100:]


# ============================================================
# HEADER
# ============================================================

st.title("🚛 NER Smart Fleet Tracking")

st.caption(
    "Real-time logistics monitoring and AI-assisted fleet intelligence"
)


# ============================================================
# METRICS
# ============================================================

total = len(vehicles)

active = sum(
    vehicle["status"] == "ACTIVE"
    for vehicle in vehicles
)

stopped = sum(
    vehicle["speed"] < 3
    for vehicle in vehicles
)

overspeed = sum(
    vehicle["speed"] > 90
    for vehicle in vehicles
)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("🚛 Vehicles", total)
c2.metric("🟢 Moving", active)
c3.metric("🟠 Stopped", stopped)
c4.metric("🔴 Overspeed", overspeed)
c5.metric(
    "📡 Connection",
    "LIVE" if api_url else "DEMO",
)


# ============================================================
# SEARCH VEHICLE
# ============================================================

st.divider()

search = st.text_input(
    "🔎 Search vehicle",
    placeholder="Search NER-001...",
)

filtered = vehicles

if search:
    search_value = search.lower()

    filtered = [
        vehicle
        for vehicle in vehicles
        if search_value in vehicle["id"].lower()
        or search_value in vehicle["cargo"].lower()
        or search_value in vehicle["destination"].lower()
    ]


# ============================================================
# VEHICLE SELECT
# ============================================================

if filtered:
    selected_id = st.selectbox(
        "Select vehicle",
        [
            vehicle["id"]
            for vehicle in filtered
        ],
    )

    selected = next(
        (
            vehicle
            for vehicle in vehicles
            if vehicle["id"] == selected_id
        ),
        None,
    )

else:
    selected = None


# ============================================================
# VEHICLE INFORMATION
# ============================================================

if selected:
    st.divider()

    st.subheader(f"🚛 {selected['id']}")

    a, b, c, d, e = st.columns(5)

    a.metric(
        "Speed",
        f"{selected['speed']:.1f} km/h",
    )

    b.metric(
        "Heading",
        f"{selected['heading']:.0f}°",
    )

    c.metric("Cargo", selected["cargo"])
    d.metric("Driver", selected["driver"])

    e.metric(
        "Fuel",
        (
            f"{selected['fuel']}%"
            if selected["fuel"] is not None
            else "N/A"
        ),
    )


# ============================================================
# MAP
# ============================================================

st.divider()

st.subheader("🗺️ Live Fleet Map")

if selected:
    center = [
        selected["lat"],
        selected["lon"],
    ]
    zoom = 10

else:
    center = [25.7, 92.5]
    zoom = 6

tiles = (
    "OpenTopoMap"
    if map_style == "OpenTopoMap"
    else "OpenStreetMap"
)

m = folium.Map(
    location=center,
    zoom_start=zoom,
    tiles=tiles,
    control_scale=True,
    prefer_canvas=True,
)


# ============================================================
# DISTRICT MARKERS
# ============================================================

for name, coordinate in DISTRICTS.items():
    folium.Marker(
        location=[
            coordinate[0],
            coordinate[1],
        ],
        tooltip=name,
        popup=f"""
        <b>{name}</b><br>
        NER Smart Monitoring Point
        """,
        icon=folium.Icon(
            icon="info-sign",
        ),
    ).add_to(m)


# ============================================================
# VEHICLE TRAILS
# ============================================================

if show_trails:
    for vehicle in vehicles:
        history = st.session_state.vehicle_history.get(
            vehicle["id"],
            [],
        )

        if len(history) > 1:
            folium.PolyLine(
                locations=history,
                color="#777777",
                weight=3,
                opacity=0.5,
                dash_array="6, 8",
                tooltip=f"{vehicle['id']} GPS trail",
            ).add_to(m)


# ============================================================
# VEHICLE MARKERS
# ============================================================

for vehicle in filtered:
    if vehicle["status"] == "ACTIVE":
        icon_color = "green"

    elif vehicle["speed"] < 3:
        icon_color = "orange"

    else:
        icon_color = "red"

    popup_html = f"""
    <div style="width:260px">
        <h4>🚛 {vehicle['id']}</h4>
        <b>Status:</b> {vehicle['status']}<br>
        <b>Speed:</b> {vehicle['speed']:.1f} km/h<br>
        <b>Heading:</b> {vehicle['heading']:.0f}°<br>
        <b>Cargo:</b> {vehicle['cargo']}<br>
        <b>Vehicle:</b> {vehicle['vehicle_type']}<br>
        <b>Driver:</b> {vehicle['driver']}<br>
        <b>Route:</b> {vehicle['origin']} → {vehicle['destination']}<br>
        <b>GPS:</b> {vehicle['lat']:.5f}, {vehicle['lon']:.5f}<br>
        <b>Updated:</b> {vehicle['timestamp']}
    </div>
    """

    folium.Marker(
        location=[
            vehicle["lat"],
            vehicle["lon"],
        ],
        tooltip=(
            f"🚛 {vehicle['id']} | "
            f"{vehicle['speed']:.0f} km/h"
        ),
        popup=folium.Popup(
            popup_html,
            max_width=350,
        ),
        icon=folium.Icon(
            color=icon_color,
            icon="truck",
            prefix="fa",
        ),
    ).add_to(m)


# ============================================================
# SELECTED VEHICLE ROAD ROUTE
# ============================================================

road_route = None
road_route_status = None

if selected and show_roads:
    route_points = upcoming_route_points(selected)

    if len(route_points) < 2:
        road_route_status = "Vehicle is already at its final destination."

    else:
        road_route = get_driving_route(
            tuple(route_points)
        )

        if road_route:
            # White route border.
            folium.PolyLine(
                locations=road_route["coordinates"],
                color="#FFFFFF",
                weight=11,
                opacity=0.9,
            ).add_to(m)

            # Moving blue direction route.
            AntPath(
                locations=road_route["coordinates"],
                color="#1976D2",
                pulse_color="#FFFFFF",
                weight=7,
                opacity=0.95,
                delay=800,
                dash_array="12, 22",
            ).add_to(m)

            # Destination marker.
            folium.Marker(
                location=road_route["coordinates"][-1],
                tooltip=(
                    f"Destination: "
                    f"{selected['destination']}"
                ),
                icon=folium.Icon(
                    color="red",
                    icon="flag-checkered",
                    prefix="fa",
                ),
            ).add_to(m)

            # Fit map to the complete driving route.
            m.fit_bounds(
                road_route["coordinates"],
                padding=(35, 35),
            )

        else:
            road_route_status = (
                "Road route is temporarily unavailable. "
                "No incorrect straight line is shown."
            )

elif selected:
    road_route_status = (
        "Enable 'Show road route' to display navigation."
    )


# ============================================================
# MAP CONTROLS
# ============================================================

folium.LayerControl().add_to(m)


# ============================================================
# RENDER MAP
# ============================================================

st_folium(
    m,
    width=None,
    height=700,
    returned_objects=[],
)

if road_route_status:
    st.caption(road_route_status)


# ============================================================
# SELECTED VEHICLE DETAILS
# ============================================================

if selected:
    st.divider()

    st.subheader("📍 Vehicle Details")

    destination = selected["destination"]

    remaining = (
        road_route["distance_km"]
        if road_route
        else None
    )

    eta_text = (
        f"{road_route['duration_min']:.0f} min"
        if road_route
        else "Road route unavailable"
    )

    x1, x2, x3, x4 = st.columns(4)

    x1.metric(
        "📍 Current Position",
        f"{selected['lat']:.4f}, "
        f"{selected['lon']:.4f}",
    )

    x2.metric(
        "🎯 Destination",
        destination,
    )

    x3.metric(
        "🛣️ Road Remaining",
        (
            f"{remaining:.1f} km"
            if remaining is not None
            else "Unknown"
        ),
    )

    x4.metric(
        "⏱️ Route ETA",
        eta_text,
    )


# ============================================================
# ALERT CENTER
# ============================================================

st.divider()

st.subheader("🚨 Fleet Alert Center")

alerts = []

for vehicle in vehicles:
    if vehicle["speed"] > 90:
        alerts.append(
            f"🔴 {vehicle['id']} — "
            f"Overspeed: "
            f"{vehicle['speed']:.1f} km/h"
        )

    if vehicle["speed"] < 3:
        alerts.append(
            f"🟠 {vehicle['id']} — "
            "Vehicle stopped"
        )

    if vehicle["status"] == "OFFLINE":
        alerts.append(
            f"🔴 {vehicle['id']} — "
            "GPS offline"
        )

if alerts:
    for alert in alerts:
        st.warning(alert)

else:
    st.success("🟢 Fleet operating normally")


# ============================================================
# FLEET TABLE
# ============================================================

st.divider()

st.subheader("📋 Fleet Operations")

rows = []

for vehicle in filtered:
    rows.append(
        {
            "Vehicle": vehicle["id"],
            "Status": vehicle["status"],
            "Speed km/h": round(
                vehicle["speed"],
                1,
            ),
            "Heading": round(
                vehicle["heading"],
                0,
            ),
            "Cargo": vehicle["cargo"],
            "Origin": vehicle["origin"],
            "Destination": vehicle["destination"],
            "Driver": vehicle["driver"],
            "Latitude": round(
                vehicle["lat"],
                5,
            ),
            "Longitude": round(
                vehicle["lon"],
                5,
            ),
        }
    )

if rows:
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# GPS API DOCUMENTATION
# ============================================================

st.divider()

st.subheader("📡 Connect Your GPS Device")

st.info(
    "Your GPS/vehicle backend should send JSON in this format."
)

gps_example = {
    "vehicles": [
        {
            "id": "NER-001",
            "lat": 26.1445,
            "lon": 91.7362,
            "speed": 42,
            "heading": 90,
            "cargo": "Medicines",
            "status": "ACTIVE",
            "origin": "Guwahati",
            "destination": "Aizawl",
            "route": [
                "Guwahati",
                "Silchar",
                "Aizawl",
            ],
            "driver": "Driver-001",
            "vehicle_type": "Medical Supply Truck",
            "fuel": 78,
            "timestamp": "2026-09-17T10:30:00Z",
        }
    ]
}

st.json(gps_example)


# ============================================================
# AUTO REFRESH
# ============================================================

if refresh != "Off":
    seconds = int(
        refresh.split()[0]
    )

    time.sleep(seconds)
    st.rerun()