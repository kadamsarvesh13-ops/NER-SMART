import streamlit as st

from common import init_state, ROADS, INCIDENT_TYPES, add_incident

init_state()
st.title("\U0001F4F1 Field Reporting")
st.caption("Field officers submit geo-tagged incident reports. Open incidents immediately "
           "feed the live risk model on every other page.")

with st.form("incident_form", clear_on_submit=True):
    road_options = {f"{r['name']} ({r['a']} \u2192 {r['b']})": r for r in ROADS}
    road_choice = st.selectbox("Road", list(road_options.keys()))
    r = road_options[road_choice]

    c1, c2 = st.columns(2)
    itype = c1.selectbox("Incident type", INCIDENT_TYPES)
    severity = c2.selectbox("Severity", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    location = st.text_input("Specific location / landmark", value=f"Near {r['a']}")
    description = st.text_area("Description", placeholder="Road blocked due to major landslide...")
    photo = st.file_uploader("Attach photo (optional)", type=["png", "jpg", "jpeg"])
    reported_by = st.text_input("Reported by", value="Field Officer")
    submitted = st.form_submit_button("\U0001F6A8 Submit Incident", use_container_width=True)

if submitted:
    add_incident(r["id"], r["name"], location, itype, severity, description, reported_by)
    st.success(f"Incident logged on {r['name']} ({r['a']} \u2192 {r['b']}). "
               f"Risk scores updated across the app.")
    if photo is not None:
        st.image(photo, caption="Attached field photo", width=320)
        st.info(
            "**AI image analysis (simulated):** In production this frame would run through "
            "a fine-tuned CNN/YOLO classifier (Landslide / Flood / Damaged Road / Bridge "
            "Damage / Normal) and auto-suggest the severity above."
        )

st.divider()
st.subheader("Open Incidents")
incidents = st.session_state.incidents
if incidents.empty:
    st.caption("No incidents reported yet.")
else:
    for i, row in incidents[::-1].iterrows():
        cols = st.columns([5, 1])
        with cols[0]:
            st.markdown(
                f"**{row['type']}** on **{row['road_name']}** \u2014 {row['location']} "
                f"| Severity: **{row['severity']}** | Status: **{row['status']}**  \n"
                f"<span style='color:gray'>{row['timestamp']} \u00b7 reported by {row['reported_by']}</span>  \n"
                f"{row['description']}",
                unsafe_allow_html=True,
            )
        with cols[1]:
            if row["status"] == "OPEN" and st.button("Resolve", key=f"resolve_{i}"):
                st.session_state.incidents.loc[i, "status"] = "RESOLVED"
                st.rerun()
        st.divider()
