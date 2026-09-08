import pandas as pd
import plotly.express as px
import streamlit as st

from common import init_state, train_risk_model, predict_risk, explain_risk, ROADS

init_state()
st.title("\U0001F916 AI Disruption Risk Prediction")
st.caption("A RandomForest model trained on synthetic weather/terrain/traffic examples "
           "predicts a 0-100 disruption score, with a transparent breakdown of *why*.")

model = train_risk_model()

st.subheader("Predict risk for a road")
road_options = {f"{r['name']} ({r['a']} \u2192 {r['b']})": r for r in ROADS}
choice = st.selectbox("Road", list(road_options.keys()))
r = road_options[choice]

c1, c2 = st.columns(2)
with c1:
    rainfall = st.slider("Rainfall (mm, next 24h)", 0, 250, 40)
    slope = st.slider("Terrain slope (degrees)", 0, 60, int(r["slope"]))
    historical = st.slider("Recent historical incidents", 0, 10, 2)
with c2:
    road_condition = st.slider("Road condition (0=good, 1=poor)", 0.0, 1.0, round(r["base_risk"], 2))
    traffic = st.slider("Traffic level (0=light, 1=heavy)", 0.0, 1.0, 0.4)

risk = predict_risk(model, rainfall, slope, historical, road_condition, traffic)

if risk >= 80:
    label, color = "CRITICAL", "red"
elif risk >= 60:
    label, color = "HIGH", "orange"
elif risk >= 30:
    label, color = "MEDIUM", "orange"
else:
    label, color = "LOW", "green"

st.metric("Predicted Disruption Probability", f"{risk:.0f}%", label)
st.progress(min(int(risk), 100) / 100)

st.subheader("Why is this road at this risk level?")
contributions = explain_risk(rainfall, slope, historical, road_condition, traffic)
contrib_df = pd.DataFrame(
    {"Factor": list(contributions.keys()), "Contribution": list(contributions.values())}
).sort_values("Contribution", ascending=True)
fig = px.bar(contrib_df, x="Contribution", y="Factor", orientation="h",
             color="Contribution", color_continuous_scale="OrRd")
fig.update_layout(height=320, showlegend=False, coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "These contributions use the same weighted formula that generated the model's "
    "training labels, so they stay consistent with the prediction above — a simple "
    "but honest stand-in for a full SHAP explanation."
)

with st.expander("How would this become a real model?"):
    st.markdown(
        "- Swap the synthetic training generator for **historical incident + weather "
        "+ IMD/road-authority data** joined by road segment and time.\n"
        "- Keep the same feature schema so the rest of the app (routing, alerts) "
        "doesn't need to change.\n"
        "- Add a held-out evaluation set and track precision/recall on actual road "
        "closures, not just R².\n"
        "- Replace the hand-written `explain_risk` with real SHAP values once the "
        "model is no longer hand-designed."
    )
