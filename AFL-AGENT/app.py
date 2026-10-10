"""Streamlit frontend.  Run:  streamlit run app.py"""
import os
import uuid

import streamlit as st

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="AFL Agent", page_icon="🏉")


@st.cache_resource(show_spinner="Loading models and knowledge base...")
def load_agent():
    from Day3 import DB_DIR, ingest
    if not os.path.isdir(DB_DIR):          # first run on a fresh server: build the vector store
        ingest()
    from Day4 import build_agent, safe_ask
    return build_agent(), safe_ask


agent, safe_ask = load_agent()

EXAMPLES = [
    "How many points is a goal worth?",
    "Who will win Collingwood vs Carlton?",
    "Who are the likely top players in Sydney vs Hawthorn?",
    "Who will win Geelong vs Richmond, and when was Geelong founded?",
]

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []

st.title("🏉 AFL Agent")
st.caption("Ask about AFL rules and history (answers cite sources) or request a match / top-player "
           "prediction from the trained models. Off-topic questions are politely declined.")

clicked = None
with st.sidebar:
    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            clicked = ex
    if st.button("New conversation", type="primary", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Limits: 500-character questions, 8 messages/minute, 10 predictions per 10 minutes.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("tools"):
            with st.expander("Model output (raw tool results)"):
                st.json(m["tools"])

prompt = st.chat_input("Ask about AFL rules, history, or a match prediction...") or clicked
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            r = safe_ask(agent, st.session_state.thread_id, prompt)
        st.markdown(r["answer"])
        st.caption(f"route: {r['route']}")
        if r["tools"]:
            with st.expander("Model output (raw tool results)"):
                st.json(r["tools"])
    st.session_state.messages.append(
        {"role": "assistant", "content": r["answer"], "tools": r["tools"]})