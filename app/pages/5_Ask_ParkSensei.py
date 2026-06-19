"""Ask ParkSensei — natural-language enforcement co-pilot (Anthropic tool-use over core.py)."""
import os
import streamlit as st
import ui, core, copilot

ui.page("Ask ParkSensei", "Q")
ui.brand_sidebar()

st.markdown("## Ask ParkSensei")
st.caption("Your enforcement co-pilot — ask in plain English, Hindi or Kannada. Queries run against the live model, "
           "not canned text.")

EXAMPLES = [
    "Where should I send 6 teams on Friday evening?",
    "Plan 8 teams for Saturday morning around KR Market",
    "What are the 5 worst hotspots?",
    "How bad is our evening coverage?",
]

def _get_secret(name, default=""):
    try:
        v = st.secrets.get(name)
        if v:
            return v
    except Exception:
        pass
    return os.environ.get(name, default)

key   = _get_secret("GEMINI_API_KEY")
model = _get_secret("COPILOT_MODEL", "gemini-2.5-flash")

# ---- not configured: show a friendly setup card and stop (keeps the app crash-free) ----
if not key:
    st.info("**Enable the co-pilot in 1 minute** — add a free Google Gemini API key "
            "(get one at aistudio.google.com/apikey).")
    st.markdown("**Local:** create `.streamlit/secrets.toml`:")
    st.code('GEMINI_API_KEY = "AIza..."', language="toml")
    st.markdown("**Streamlit Cloud:** *Manage app → Settings → Secrets* → paste the same line. "
                "Then reload this page.")
    st.markdown("Once enabled, try asking:")
    for e in EXAMPLES:
        st.markdown(f"- *{e}*")
    st.stop()

try:
    from google import genai
except ImportError:
    st.error("The `google-genai` package isn't installed. Run `pip install google-genai` (it's in requirements.txt).")
    st.stop()

client = genai.Client(api_key=key)
ctx = {"df": ui.load_data(), "zones": ui.get_zones(), "fc": ui.get_forecaster()}

st.markdown("**Try:** " + "  ·  ".join(f"`{e}`" for e in EXAMPLES))

def render_plan(plan):
    if plan is None or not len(plan):
        return
    tbl = plan[["team", "label", "pred_load", "impact_score"]].copy()
    tbl.columns = ["Team", "Deploy to", "Exp. catches", "Impact"]
    st.dataframe(tbl, hide_index=True, width="stretch")
    c = plan.iloc[0]
    st.pydeck_chart(ui.deck(ui.plan_layers(plan), ui.view(c["lat"], c["lon"], zoom=11, pitch=35),
                            ui.TIP_PLAN), width="stretch")

if "chat" not in st.session_state:
    st.session_state.chat = []

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_plan(m.get("plan"))

q = st.chat_input("e.g. Where should I send 6 teams on Friday evening?")
if q:
    st.session_state.chat.append({"role": "user", "content": q, "plan": None})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer, plan = copilot.run_agent(client, q, ctx, model=model)
            except Exception as e:
                answer, plan = f"Error — {type(e).__name__}: {e}", None
        st.markdown(answer)
        render_plan(plan)
    st.session_state.chat.append({"role": "assistant", "content": answer, "plan": plan})
