# NER-SMART — AI Logistics & Accessibility Intelligence (Prototype)

A runnable Streamlit prototype of the NER-SMART concept: an AI-powered
platform for monitoring road accessibility, predicting disruptions, and
optimizing routes for essential-goods logistics across the North Eastern
Region.

This is a **working MVP**, not a mockup — the map, risk model, routing,
vehicle simulation, incident reports, and disaster simulator are all wired
together through one shared, live data model (`common.py`), so an action on
one page (e.g. reporting a landslide) genuinely changes what every other
page shows (route recommendations, vehicle risk, the map, the copilot).

## Stack

| Layer | Choice | Why |
|---|---|---|
| UI | Streamlit (multipage) | Fastest way to a real interactive app, matches the request |
| Map | Plotly `Scattermapbox` (`open-street-map` style) | No API key/token required |
| Graph & routing | NetworkX (Dijkstra / k-shortest-paths) | Simple, well-tested, easy to swap edge weights |
| Risk model | scikit-learn `RandomForestRegressor` trained on synthetic data | Real trained model with an explainable factor breakdown; swappable for a real dataset later |
| Data | In-memory synthetic districts/roads + `st.session_state` | Zero setup — no database needed to try it |

## Project layout

```
ner-smart/
├── Home.py                        # Command Center (landing page)
├── common.py                      # Shared data model, ML model, graph, session state
├── requirements.txt
└── pages/
    ├── 1_GIS_Map.py                # Interactive map with road/district/vehicle layers
    ├── 2_AI_Risk_Prediction.py     # Risk model with sliders + explainability chart
    ├── 3_Route_Optimizer.py        # Best/fastest/alternative route comparison
    ├── 4_Fleet_Tracking.py         # Simulated GPS vehicles moving on the map
    ├── 5_Field_Reports.py          # Submit incidents → feeds the live risk graph
    ├── 6_Disaster_Simulator.py     # What-if sliders (rainfall, landslide bias, etc.)
    └── 7_AI_Copilot.py             # Rule-based chat over live conditions
```

## How to run it

1. Unzip the project and open a terminal in the `ner-smart` folder.
2. Create a virtual environment (recommended) and install dependencies:

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux

   pip install -r requirements.txt
   ```

3. Run it:

   ```bash
   streamlit run Home.py
   ```

4. Your browser opens automatically at `http://localhost:8501`. Use the
   sidebar to move between pages — Streamlit auto-discovers everything in
   `pages/`.

No API keys, database, or external services are required to run this.

## Suggested demo flow (the "killer workflow")

1. **Home** — note the network is mostly green, few blocked roads.
2. **Field Reports** — submit a `CRITICAL` **Landslide** on `NH-306
   (Silchar → Aizawl)`.
3. **GIS Map** — that road is now red; click it to see the updated risk.
4. **Route Optimizer** — search Guwahati → Aizawl; the AI-recommended
   route now avoids the blocked segment and explains why.
5. **Fleet Tracking** — vehicle `NER-1023` (medicines, same corridor) now
   shows "AT RISK".
6. **AI Copilot** — ask "show blocked roads" or "how many vehicles are at
   risk?" and get an instant, correct answer, because it reads the same
   live state.
7. **Disaster Simulator** — crank up rainfall/landslide sliders for a
   regional stress test instead of one road at a time.

## What I'd extend first

Roughly in priority order:

1. **Real data instead of synthetic.** Replace `DISTRICTS`/`ROADS` in
   `common.py` with an actual road network (start with OpenStreetMap via
   `osmnx`, or a government road dataset) and back it with PostGIS instead
   of an in-memory graph. The rest of the app (routing, map, risk model
   inputs) barely has to change since it's already isolated in
   `common.py`.
2. **Real weather + a real training set.** Wire `_init_weather()` to a
   weather API (IMD/OpenWeather) and retrain `train_risk_model()` on
   historical incident + weather data instead of the synthetic generator —
   the feature schema (`FEATURES` in `common.py`) is designed to make that
   swap drop-in.
3. **Persistence.** Everything currently lives in `st.session_state` and
   resets on refresh. Move `incidents`, `vehicles`, and `weather` into
   Postgres (or even SQLite to start) so reports and vehicle state survive
   restarts and multiple users see the same data.
4. **Real vehicle GPS.** Replace the simulated "advance simulation" button
   with an actual ingestion endpoint (FastAPI) that vehicles/phones POST
   GPS pings to, and have Streamlit poll that store.
5. **Authentication & roles.** Add `streamlit-authenticator` (or a proper
   auth service) and gate pages by role (Admin / District Official / Field
   Officer / Driver) as described in the original spec.
6. **Image classification for field reports.** The photo upload on Field
   Reports currently just displays the image — the natural next step is a
   small fine-tuned CNN (or a hosted vision API) that classifies it into
   Landslide/Flood/Damaged Road/etc. and pre-fills severity.
7. **A real LLM copilot.** The AI Copilot page is deliberately rule-based
   so the prototype needs zero API keys. `pages/7_AI_Copilot.py` has a
   note at the bottom on wiring it to the Anthropic API for open-ended
   questions, using the rule-based version as a fast/free fallback.
8. **Multi-user support.** Streamlit's `session_state` is per-browser-tab;
   moving shared state (incidents, vehicles) to a database is also what
   unlocks multiple officials seeing the same live picture at once.

## Notes on the risk model

`train_risk_model()` generates ~4,000 synthetic examples from a known
formula (rainfall, slope, historical incidents, road condition, traffic →
risk score) and fits a `RandomForestRegressor` on them — so it's a *real*
trained model, not hard-coded if/else logic, but its "ground truth" is
synthetic rather than historical. `explain_risk()` deliberately mirrors
that same formula so the on-screen explanation stays consistent with the
prediction; replace both once real historical data is available.
