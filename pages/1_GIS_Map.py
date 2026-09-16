# ============================================================
# NER-SMART
# GIS ACCESSIBILITY & LOGISTICS INTELLIGENCE MAP
# ============================================================

import math
from datetime import datetime, timezone

import folium
import networkx as nx
import pandas as pd
import requests
import streamlit as st

from folium.plugins import (
    AntPath,
    Fullscreen,
    MarkerCluster,
    MiniMap,
    MousePosition,
)
from streamlit_folium import st_folium

from common import (
    init_state,
    build_live_graph,
    DISTRICTS,
    ROADS,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NER-SMART GIS",
    page_icon="🗺️",
    layout="wide",
)

init_state()


# ============================================================
# CONFIGURATION
# ============================================================

OSRM_URL = "https://router.project-osrm.org"

NER_CENTER = [25.8, 93.0]


# ============================================================
# ADDITIONAL IMPORTANT NER LOCATIONS
# ============================================================

LOCATIONS = {
    "Guwahati": [26.1445, 91.7362],
    "Shillong": [25.5788, 91.8933],
    "Itanagar": [27.0844, 93.6053],
    "Imphal": [24.8170, 93.9368],
    "Aizawl": [23.7271, 92.7176],
    "Kohima": [25.6751, 94.1086],
    "Agartala": [23.8315, 91.2868],
    "Gangtok": [27.3389, 88.6065],
    "Dibrugarh": [27.4728, 94.9120],
    "Tinsukia": [27.4922, 95.3468],
    "Silchar": [24.8333, 92.7789],
    "Tezpur": [26.6528, 92.7926],
    "Dimapur": [25.8629, 93.7538],
    "Dima Hasao": [25.5000, 93.0000],
}


# ============================================================
# STATE COORDINATES
# ============================================================

STATES = {
    "Assam": [26.20, 92.94],
    "Arunachal Pradesh": [28.20, 94.70],
    "Manipur": [24.80, 93.94],
    "Meghalaya": [25.47, 91.36],
    "Mizoram": [23.16, 92.94],
    "Nagaland": [26.16, 94.56],
    "Tripura": [23.94, 91.99],
    "Sikkim": [27.53, 88.51],
}


# ============================================================
# DISTRICT COORDINATE HELPER
# ============================================================

def get_location(name):
    """
    Supports both:
        DISTRICTS[name] = [lat, lon]
    and
        DISTRICTS[name] = {"lat": ..., "lon": ...}
    """

    value = DISTRICTS.get(name)

    if value is None:
        value = LOCATIONS.get(name)

    if value is None:
        return None

    if isinstance(value, dict):
        return [
            float(value["lat"]),
            float(value["lon"]),
        ]

    return [
        float(value[0]),
        float(value[1]),
    ]


# ============================================================
# OSRM REAL ROAD ROUTING
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def get_osrm_route(points):
    """
    Uses OSRM + OpenStreetMap to obtain
    actual drivable road geometry.
    """

    if not points or len(points) < 2:
        return [], 0, 0

    try:

        coordinates = ";".join(
            f"{lon},{lat}"
            for lat, lon in points
        )

        url = (
            f"{OSRM_URL}/route/v1/driving/"
            f"{coordinates}"
        )

        params = {
            "overview": "full",
            "geometries": "geojson",
            "steps": "false",
            "alternatives": "false",
        }

        response = requests.get(
            url,
            params=params,
            headers={
                "User-Agent": "NER-SMART/1.0"
            },
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        if (
            data.get("code") != "Ok"
            or not data.get("routes")
        ):
            return [], 0, 0

        route = data["routes"][0]

        geometry = route[
            "geometry"
        ]["coordinates"]

        latlon = [
            [point[1], point[0]]
            for point in geometry
        ]

        distance = float(
            route.get("distance", 0)
        )

        duration = float(
            route.get("duration", 0)
        )

        return (
            latlon,
            distance,
            duration,
        )

    except Exception:
        return [], 0, 0


# ============================================================
# ROAD GEOMETRY CACHE
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def get_corridor_geometry(a, b):

    point_a = get_location(a)
    point_b = get_location(b)

    if not point_a or not point_b:
        return []

    geometry, _, _ = get_osrm_route(
        (
            tuple(point_a),
            tuple(point_b),
        )
    )

    return geometry


# ============================================================
# BEARING / VEHICLE DIRECTION
# ============================================================

def calculate_bearing(point_a, point_b):

    lat1 = math.radians(point_a[0])
    lon1 = math.radians(point_a[1])

    lat2 = math.radians(point_b[0])
    lon2 = math.radians(point_b[1])

    dlon = lon2 - lon1

    x = (
        math.sin(dlon)
        * math.cos(lat2)
    )

    y = (
        math.cos(lat1)
        * math.sin(lat2)
        -
        math.sin(lat1)
        * math.cos(lat2)
        * math.cos(dlon)
    )

    bearing = math.degrees(
        math.atan2(x, y)
    )

    return (
        bearing + 360
    ) % 360


# ============================================================
# RISK COLOR
# ============================================================

def risk_color(risk):

    risk = float(risk)

    if risk >= 80:
        return "#dc2626"

    if risk >= 60:
        return "#f97316"

    if risk >= 30:
        return "#eab308"

    return "#16a34a"


def risk_status(risk):

    risk = float(risk)

    if risk >= 80:
        return "BLOCKED / CRITICAL"

    if risk >= 60:
        return "HIGH RISK"

    if risk >= 30:
        return "PARTIALLY DISRUPTED"

    return "NORMAL"


# ============================================================
# VEHICLE DATA
# ============================================================

def default_vehicles():

    return [
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
            "driver": "Driver-001",
            "route": [
                "Guwahati",
                "Silchar",
                "Aizawl",
            ],
            "last_update": datetime.now(
                timezone.utc
            ).isoformat(),
        },

        {
            "id": "NER-002",
            "lat": 25.5788,
            "lon": 91.8933,
            "speed": 35,
            "heading": 120,
            "cargo": "Food Supplies",
            "status": "ACTIVE",
            "origin": "Shillong",
            "destination": "Imphal",
            "driver": "Driver-002",
            "route": [
                "Shillong",
                "Dimapur",
                "Imphal",
            ],
            "last_update": datetime.now(
                timezone.utc
            ).isoformat(),
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
            "driver": "Driver-003",
            "route": [
                "Dimapur",
                "Itanagar",
            ],
            "last_update": datetime.now(
                timezone.utc
            ).isoformat(),
        },

        {
            "id": "NER-004",
            "lat": 24.8333,
            "lon": 92.7789,
            "speed": 48,
            "heading": 45,
            "cargo": "Agricultural Produce",
            "status": "ACTIVE",
            "origin": "Silchar",
            "destination": "Guwahati",
            "driver": "Driver-004",
            "route": [
                "Silchar",
                "Guwahati",
            ],
            "last_update": datetime.now(
                timezone.utc
            ).isoformat(),
        },
    ]


if (
    "vehicles" not in st.session_state
    or not st.session_state.vehicles
):

    st.session_state.vehicles = (
        default_vehicles()
    )


vehicles = st.session_state.vehicles


# ============================================================
# VEHICLE ROUTE POINTS
# ============================================================

def get_vehicle_points(vehicle):

    points = [
        (
            float(vehicle["lat"]),
            float(vehicle["lon"]),
        )
    ]

    route = vehicle.get(
        "route",
        [],
    )

    for location in route:

        point = get_location(
            location
        )

        if point:

            last = points[-1]

            if (
                abs(last[0] - point[0])
                > 0.0001
                or
                abs(last[1] - point[1])
                > 0.0001
            ):
                points.append(
                    tuple(point)
                )

    destination = vehicle.get(
        "destination"
    )

    if destination:

        destination_point = get_location(
            destination
        )

        if destination_point:

            last = points[-1]

            if (
                abs(
                    last[0]
                    - destination_point[0]
                )
                > 0.0001
                or
                abs(
                    last[1]
                    - destination_point[1]
                )
                > 0.0001
            ):

                points.append(
                    tuple(
                        destination_point
                    )
                )

    return points


# ============================================================
# VEHICLE ICON
# ============================================================

def vehicle_icon(vehicle, heading):

    status = str(
        vehicle.get(
            "status",
            "ACTIVE"
        )
    ).upper()

    if status == "ACTIVE":
        background = "#16a34a"

    elif status == "STOPPED":
        background = "#f59e0b"

    else:
        background = "#dc2626"

    html = f"""
    <div style="
        width:42px;
        height:42px;
        border-radius:50%;
        background:{background};
        border:3px solid white;
        box-shadow:
            0 3px 12px rgba(0,0,0,.45);
        display:flex;
        align-items:center;
        justify-content:center;
        transform:rotate({heading}deg);
        font-size:22px;
        z-index:9999;
    ">
        🚚
    </div>
    """

    return folium.DivIcon(
        html=html,
        icon_size=(42, 42),
        icon_anchor=(21, 21),
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "🗺️ NER-SMART — GIS Command Map"
)

st.caption(
    "Real-world OpenStreetMap roads • "
    "AI accessibility intelligence • "
    "GPS fleet tracking • "
    "weather risk • field incidents"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🗺️ Map Controls")

    show_roads = st.checkbox(
        "🛣️ Real Road Network",
        True,
    )

    show_risk = st.checkbox(
        "⚠️ AI Risk Corridors",
        True,
    )

    show_vehicles = st.checkbox(
        "🚚 Fleet Vehicles",
        True,
    )

    show_routes = st.checkbox(
        "📍 Vehicle Routes",
        True,
    )

    show_incidents = st.checkbox(
        "🚧 Field Incidents",
        True,
    )

    show_states = st.checkbox(
        "🏛️ States",
        False,
    )

    show_locations = st.checkbox(
        "📍 Logistics Locations",
        True,
    )

    st.divider()

    selected_vehicle = st.selectbox(
        "🚚 Focus Vehicle",
        ["All Vehicles"]
        + [
            v["id"]
            for v in vehicles
        ],
    )


# ============================================================
# MAP
# ============================================================

m = folium.Map(
    location=NER_CENTER,
    zoom_start=6,
    tiles=None,
    control_scale=True,
)


# ============================================================
# BASE MAP
# ============================================================

folium.TileLayer(
    "OpenStreetMap",
    name="🗺️ OpenStreetMap",
).add_to(m)


folium.TileLayer(
    tiles=(
        "https://{s}.tile.opentopomap.org/"
        "{z}/{x}/{y}.png"
    ),
    attr="OpenTopoMap",
    name="⛰️ Terrain",
).add_to(m)


# ============================================================
# MAP CONTROLS
# ============================================================

Fullscreen(
    position="topleft"
).add_to(m)

MiniMap(
    toggle_display=True
).add_to(m)

MousePosition(
    position="bottomleft",
    separator=" | ",
    prefix="GPS: ",
).add_to(m)


# ============================================================
# STATES
# ============================================================

if show_states:

    state_group = folium.FeatureGroup(
        name="🏛️ NER States"
    )

    for state, coord in STATES.items():

        folium.Marker(
            location=coord,
            tooltip=state,
            popup=(
                f"<b>{state}</b><br>"
                "North Eastern Region"
            ),
        ).add_to(
            state_group
        )

    state_group.add_to(m)


# ============================================================
# MAJOR LOGISTICS LOCATIONS
# ============================================================

if show_locations:

    location_group = (
        folium.FeatureGroup(
            name="📍 Logistics Locations"
        )
    )

    for city, coord in LOCATIONS.items():

        folium.CircleMarker(
            location=coord,
            radius=5,
            color="#1e293b",
            fill=True,
            fill_opacity=0.9,
            tooltip=city,
            popup=(
                f"<b>{city}</b><br>"
                "NER logistics monitoring location"
            ),
        ).add_to(
            location_group
        )

    location_group.add_to(m)


# ============================================================
# REAL OSM ROAD NETWORK
# ============================================================

if show_roads:

    road_group = folium.FeatureGroup(
        name="🛣️ Real OSM Roads"
    )

    with st.spinner(
        "Loading real road routes..."
    ):

        for road in ROADS:

            a = road.get("a")
            b = road.get("b")

            if not a or not b:
                continue

            geometry = (
                get_corridor_geometry(
                    a,
                    b,
                )
            )

            if not geometry:

                pa = get_location(a)
                pb = get_location(b)

                if pa and pb:
                    geometry = [
                        pa,
                        pb,
                    ]

            if geometry:

                folium.PolyLine(
                    geometry,
                    color="#64748b",
                    weight=4,
                    opacity=0.65,
                    tooltip=(
                        f"{road.get('name', 'Road')} | "
                        f"{a} → {b}"
                    ),
                ).add_to(
                    road_group
                )

    road_group.add_to(m)


# ============================================================
# AI RISK ROAD LAYER
# ============================================================

if show_risk:

    risk_group = folium.FeatureGroup(
        name="⚠️ AI Accessibility Risk"
    )

    try:

        graph = build_live_graph()

        for a, b, data in graph.edges(
            data=True
        ):

            geometry = (
                get_corridor_geometry(
                    a,
                    b,
                )
            )

            if not geometry:

                pa = get_location(a)
                pb = get_location(b)

                if pa and pb:
                    geometry = [
                        pa,
                        pb,
                    ]

            if not geometry:
                continue

            risk = float(
                data.get(
                    "risk_pct",
                    data.get(
                        "risk",
                        0
                    ),
                )
            )

            color = risk_color(
                risk
            )

            status = risk_status(
                risk
            )

            folium.PolyLine(
                geometry,
                color=color,
                weight=8,
                opacity=0.65,
                tooltip=(
                    f"{a} → {b} | "
                    f"Risk: {risk:.0f}% | "
                    f"{status}"
                ),
                popup=folium.Popup(
                    f"""
                    <b>🛣️ Road Accessibility</b><br><br>

                    <b>From:</b> {a}<br>
                    <b>To:</b> {b}<br>
                    <b>AI Risk:</b> {risk:.0f}%<br>
                    <b>Status:</b> {status}
                    """,
                    max_width=320,
                ),
            ).add_to(
                risk_group
            )

    except Exception as e:

        st.warning(
            "AI risk layer unavailable."
        )

    risk_group.add_to(m)


# ============================================================
# FLEET ROUTES
# ============================================================

if show_routes:

    fleet_route_group = (
        folium.FeatureGroup(
            name="🚚 Fleet Routes"
        )
    )

    for vehicle in vehicles:

        if (
            selected_vehicle
            != "All Vehicles"
            and vehicle["id"]
            != selected_vehicle
        ):
            continue

        points = (
            get_vehicle_points(
                vehicle
            )
        )

        geometry, distance, duration = (
            get_osrm_route(
                tuple(points)
            )
        )

        if not geometry:
            continue

        is_selected = (
            selected_vehicle
            == vehicle["id"]
        )

        if is_selected:

            # Route shadow
            folium.PolyLine(
                geometry,
                color="#2563eb",
                weight=12,
                opacity=0.20,
            ).add_to(
                fleet_route_group
            )

            # Main route
            folium.PolyLine(
                geometry,
                color="#2563eb",
                weight=5,
                opacity=0.95,
                tooltip=(
                    f"{vehicle['id']} | "
                    f"{vehicle['origin']} → "
                    f"{vehicle['destination']}"
                ),
            ).add_to(
                fleet_route_group
            )

            # Animated direction
            AntPath(
                geometry,
                color="#2563eb",
                weight=4,
                opacity=0.9,
                delay=700,
                dash_array=[
                    12,
                    22
                ],
                pulse_color="#93c5fd",
            ).add_to(
                fleet_route_group
            )

        else:

            folium.PolyLine(
                geometry,
                color="#94a3b8",
                weight=3,
                opacity=0.45,
                dash_array="8 8",
                tooltip=vehicle["id"],
            ).add_to(
                fleet_route_group
            )

        # Destination
        destination = vehicle.get(
            "destination"
        )

        destination_point = (
            get_location(
                destination
            )
            if destination
            else None
        )

        if (
            destination_point
            and is_selected
        ):

            folium.Marker(
                destination_point,
                tooltip=(
                    f"🏁 {destination}"
                ),
                popup=(
                    f"<b>🏁 Destination</b><br>"
                    f"{destination}"
                ),
                icon=folium.Icon(
                    color="red",
                    icon="flag",
                    prefix="fa",
                ),
            ).add_to(
                fleet_route_group
            )

    fleet_route_group.add_to(m)


# ============================================================
# VEHICLE MARKERS
# ============================================================

if show_vehicles:

    vehicle_group = (
        folium.FeatureGroup(
            name="🚚 Live Fleet"
        )
    )

    cluster = MarkerCluster(
        name="Fleet Vehicles"
    )

    cluster.add_to(
        vehicle_group
    )

    for vehicle in vehicles:

        if (
            selected_vehicle
            != "All Vehicles"
            and vehicle["id"]
            != selected_vehicle
        ):
            continue

        points = (
            get_vehicle_points(
                vehicle
            )
        )

        geometry, _, _ = (
            get_osrm_route(
                tuple(points)
            )
        )

        if (
            geometry
            and len(geometry) > 2
        ):

            heading = calculate_bearing(
                geometry[0],
                geometry[
                    min(
                        12,
                        len(geometry) - 1
                    )
                ],
            )

        else:

            heading = float(
                vehicle.get(
                    "heading",
                    0
                )
            )

        popup = f"""
        <div style="min-width:250px">

        <h3>🚚 {vehicle['id']}</h3>

        <b>Status:</b>
        {vehicle.get('status', 'UNKNOWN')}
        <br>

        <b>Driver:</b>
        {vehicle.get('driver', '—')}
        <br>

        <b>Cargo:</b>
        {vehicle.get('cargo', '—')}
        <br>

        <b>Origin:</b>
        {vehicle.get('origin', '—')}
        <br>

        <b>Destination:</b>
        {vehicle.get('destination', '—')}
        <br>

        <b>Speed:</b>
        {vehicle.get('speed', 0)} km/h
        <br>

        <b>Heading:</b>
        {heading:.0f}°
        <br><br>

        <b>GPS:</b><br>
        {vehicle['lat']:.6f},
        {vehicle['lon']:.6f}

        </div>
        """

        folium.Marker(
            location=[
                float(vehicle["lat"]),
                float(vehicle["lon"]),
            ],
            tooltip=(
                f"🚚 {vehicle['id']} | "
                f"{vehicle.get('speed', 0)} km/h"
            ),
            popup=folium.Popup(
                popup,
                max_width=350,
            ),
            icon=vehicle_icon(
                vehicle,
                heading,
            ),
        ).add_to(
            cluster
        )

    vehicle_group.add_to(m)


# ============================================================
# FIELD INCIDENTS
# ============================================================

if show_incidents:

    incident_group = (
        folium.FeatureGroup(
            name="🚧 Field Incidents"
        )
    )

    incidents = (
        st.session_state.incidents
    )

    if (
        hasattr(incidents, "empty")
        and not incidents.empty
    ):

        for _, incident in incidents.iterrows():

            location_name = str(
                incident.get(
                    "location",
                    ""
                )
            )

            point = get_location(
                location_name
            )

            if not point:
                continue

            severity = str(
                incident.get(
                    "severity",
                    "MEDIUM"
                )
            ).upper()

            if severity in [
                "CRITICAL",
                "HIGH",
            ]:
                color = "red"

            elif severity == "MEDIUM":
                color = "orange"

            else:
                color = "green"

            folium.Marker(
                point,
                tooltip=(
                    f"🚧 {incident.get('type', 'Incident')}"
                ),
                popup=folium.Popup(
                    f"""
                    <b>🚧 Field Incident</b><br><br>

                    <b>Type:</b>
                    {incident.get('type', '—')}<br>

                    <b>Location:</b>
                    {location_name}<br>

                    <b>Severity:</b>
                    {severity}<br>

                    <b>Status:</b>
                    {incident.get('status', 'OPEN')}<br>

                    <b>Description:</b>
                    {incident.get('description', '—')}
                    """,
                    max_width=350,
                ),
                icon=folium.Icon(
                    color=color,
                    icon="warning-sign",
                    prefix="glyphicon",
                ),
            ).add_to(
                incident_group
            )

    incident_group.add_to(m)


# ============================================================
# MAP LEGEND
# ============================================================

legend_html = """
<div style="
position: fixed;
bottom: 30px;
left: 30px;
z-index: 9999;
background: white;
padding: 14px;
border-radius: 8px;
box-shadow: 0 2px 10px rgba(0,0,0,.25);
font-size: 13px;
">

<b>NER-SMART Road Status</b><br><br>

<span style="
display:inline-block;
width:25px;
height:5px;
background:#16a34a;
margin-right:6px;
"></span>
Normal<br>

<span style="
display:inline-block;
width:25px;
height:5px;
background:#eab308;
margin-right:6px;
"></span>
Caution<br>

<span style="
display:inline-block;
width:25px;
height:5px;
background:#f97316;
margin-right:6px;
"></span>
High Risk<br>

<span style="
display:inline-block;
width:25px;
height:5px;
background:#dc2626;
margin-right:6px;
"></span>
Blocked / Critical<br><br>

🚚 Live Fleet<br>
🏁 Destination<br>
🚧 Field Incident

</div>
"""

m.get_root().html.add_child(
    folium.Element(
        legend_html
    )
)


# ============================================================
# LAYER CONTROL
# ============================================================

folium.LayerControl(
    collapsed=False
).add_to(m)


# ============================================================
# DISPLAY MAP
# ============================================================

st_folium(
    m,
    height=700,
    use_container_width=True,
    returned_objects=[],
)


# ============================================================
# SELECTED VEHICLE INFORMATION
# ============================================================

if selected_vehicle != "All Vehicles":

    selected = next(
        (
            v
            for v in vehicles
            if v["id"]
            == selected_vehicle
        ),
        None,
    )

    if selected:

        st.divider()

        st.subheader(
            f"🚚 {selected['id']} — Live Vehicle Intelligence"
        )

        points = get_vehicle_points(
            selected
        )

        geometry, distance, duration = (
            get_osrm_route(
                tuple(points)
            )
        )

        hours = int(
            duration // 3600
        )

        minutes = int(
            (duration % 3600)
            // 60
        )

        eta = (
            f"{hours}h {minutes}m"
            if hours
            else f"{minutes} min"
        )

        a, b, c, d, e = st.columns(5)

        a.metric(
            "Status",
            selected.get(
                "status",
                "UNKNOWN"
            ),
        )

        b.metric(
            "Speed",
            f"{selected.get('speed', 0)} km/h",
        )

        c.metric(
            "Road Distance",
            (
                f"{distance / 1000:.1f} km"
                if distance
                else "—"
            ),
        )

        d.metric(
            "Routing ETA",
            eta,
        )

        e.metric(
            "Destination",
            selected.get(
                "destination",
                "—"
            ),
        )

        st.code(
            f"""
Vehicle ID : {selected['id']}
Latitude   : {selected['lat']:.6f}
Longitude  : {selected['lon']:.6f}
Speed      : {selected.get('speed', 0)} km/h
Heading    : {selected.get('heading', 0)}°
Cargo      : {selected.get('cargo', '—')}
Driver     : {selected.get('driver', '—')}
Origin     : {selected.get('origin', '—')}
Destination: {selected.get('destination', '—')}
""",
            language="text",
        )


# ============================================================
# GIS SUMMARY
# ============================================================

st.divider()

st.subheader(
    "📊 GIS Accessibility Intelligence"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "NER States",
    len(STATES),
)

c2.metric(
    "Monitoring Locations",
    len(LOCATIONS),
)

c3.metric(
    "Fleet Vehicles",
    len(vehicles),
)

c4.metric(
    "Map Data",
    "OpenStreetMap",
)


# ============================================================
# ROAD STATUS TABLE
# ============================================================

st.subheader(
    "🛣️ Corridor Accessibility"
)

try:

    G = build_live_graph()

    rows = []

    for a, b, data in G.edges(
        data=True
    ):

        risk = float(
            data.get(
                "risk_pct",
                data.get(
                    "risk",
                    0
                ),
            )
        )

        rows.append(
            {
                "From": a,
                "To": b,
                "AI Risk %": round(
                    risk,
                    1,
                ),
                "Status": risk_status(
                    risk
                ),
            }
        )

    if rows:

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

except Exception:

    st.info(
        "Road intelligence table unavailable."
    )