import streamlit as st

from common import init_state, build_live_graph, edges_dataframe, DISTRICTS, route_summary
import networkx as nx

init_state()
st.title("\U0001F9E0 NER AI Copilot")
st.caption("Ask about current conditions in plain English. This is a rule-based copilot over "
           "the live data (no external API key needed) — see the note below for upgrading to a real LLM.")

G = build_live_graph()
edges_df = edges_dataframe(G)

SAMPLE_QUESTIONS = [
    "Which roads are highest risk right now?",
    "Which districts are at highest risk today?",
    "Show blocked roads",
    "Safest route to Aizawl",
    "How many vehicles are at risk?",
    "Any open incidents?",
]


def answer(question: str) -> str:
    q = question.lower()

    if "highest risk" in q and "road" in q:
        top = edges_df.head(3)
        lines = [f"- **{r['name']}** ({r['a']} \u2192 {r['b']}): {r['risk_pct']:.0f}% ({r['status']})"
                 for _, r in top.iterrows()]
        return "The highest-risk roads right now are:\n" + "\n".join(lines)

    if "district" in q and "risk" in q:
        district_risk = {}
        for d in DISTRICTS:
            related = edges_df[(edges_df.a == d) | (edges_df.b == d)]
            district_risk[d] = related["risk_pct"].mean() if not related.empty else 0
        ranked = sorted(district_risk.items(), key=lambda x: -x[1])[:3]
        lines = [f"- **{d}**: avg connecting-road risk {r:.0f}%" for d, r in ranked]
        return "Highest-risk districts by connecting-road risk:\n" + "\n".join(lines)

    if "blocked" in q:
        blocked = edges_df[edges_df.status == "BLOCKED"]
        if blocked.empty:
            return "No roads are currently BLOCKED."
        lines = [f"- **{r['name']}** ({r['a']} \u2192 {r['b']})" for _, r in blocked.iterrows()]
        return "Currently blocked roads:\n" + "\n".join(lines)

    if "safest route to" in q or "route to" in q:
        target = None
        for d in DISTRICTS:
            if d.lower() in q:
                target = d
                break
        if not target:
            return "Tell me a destination district, e.g. 'safest route to Aizawl'."
        origin = "Guwahati"
        try:
            path = nx.shortest_path(G, origin, target, weight="weight")
            s = route_summary(G, path)
            return (f"From {origin} to {target}: " + " \u2192 ".join(path) +
                    f"\n\nDistance {s['distance_km']} km, max segment risk {s['max_risk']}%, "
                    f"ETA ~{s['eta_hours']} h.")
        except nx.NetworkXNoPath:
            return f"No path currently exists from {origin} to {target}."

    if "vehicle" in q and "risk" in q:
        at_risk = 0
        names = []
        for v in st.session_state.vehicles:
            edges = list(zip(v["path"][:-1], v["path"][1:]))
            if any(G.edges[a, b]["status"] in ("HIGH RISK", "BLOCKED") for a, b in edges):
                at_risk += 1
                names.append(v["id"])
        if at_risk == 0:
            return "No vehicles are currently on a high-risk or blocked segment."
        return f"{at_risk} vehicle(s) are on a high-risk or blocked segment: {', '.join(names)}."

    if "incident" in q:
        open_incidents = st.session_state.incidents[st.session_state.incidents.status == "OPEN"]
        if open_incidents.empty:
            return "No open incidents right now."
        lines = [f"- {r['type']} on {r['road_name']} ({r['severity']})" for _, r in open_incidents.iterrows()]
        return f"{len(open_incidents)} open incident(s):\n" + "\n".join(lines)

    return ("I can answer questions about road risk, blocked roads, district risk, "
            "routes, vehicles at risk, and open incidents. Try one of the sample "
            "questions in the sidebar.")


with st.sidebar:
    st.subheader("Try asking")
    for sq in SAMPLE_QUESTIONS:
        if st.button(sq, use_container_width=True):
            st.session_state.copilot_q = sq

if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.chat_input("Ask about NER logistics...", key="copilot_input")
if "copilot_q" in st.session_state:
    query = st.session_state.pop("copilot_q")

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)

if query:
    st.session_state.chat.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)
    reply = answer(query)
    st.session_state.chat.append(("assistant", reply))
    with st.chat_message("assistant"):
        st.markdown(reply)

with st.expander("Upgrading this to a real LLM copilot"):
    st.markdown(
        "This copilot is intentionally rule-based so the prototype runs with zero API "
        "keys. To upgrade it:\n"
        "1. Build a short context string from `edges_df`, `st.session_state.incidents`, "
        "and `st.session_state.vehicles`.\n"
        "2. Send that context plus the user's question to the Anthropic Messages API "
        "(`claude-sonnet-4-6` or similar) with a system prompt describing the data.\n"
        "3. Keep the rule-based `answer()` function as a fast, free fallback for common "
        "queries, and only call the API for anything it doesn't match."
    )
