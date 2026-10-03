import os
import sys
from pathlib import Path

# Streamlit apps need to be launched by Streamlit so its script runner can
# provide the ScriptRunContext used by st.session_state, caching, and widgets.
# Keep `python query_agent_web.py` convenient by handing off to Streamlit.
if __name__ == "__main__" and "streamlit" not in sys.modules:
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
        ],
    )

import streamlit as st
from rag_engine import HRAgentEngine

st.set_page_config(page_title="HR Agent RAG Portal", page_icon="🤖", layout="centered")
st.title("🤖 Enterprise HR Agent Portal")

# Setup layout environment variables in sidebar framework
st.sidebar.header("⚙️ Engine Configuration")
engine_mode = st.sidebar.selectbox(
    "Select Backend Engine Target:",
    ["Groq Cloud (Blazing Fast)", "Local Ollama (llama3.1:8b)"]
)
use_api = (engine_mode == "Groq Cloud (Blazing Fast)")

@st.cache_resource
def get_cached_agent(use_api):
    """Caches engine objects so swapping selector inputs handles memory resets beautifully."""
    try:
        return HRAgentEngine(use_api=use_api), None
    except Exception as e:
        return None, str(e)

agent, err = get_cached_agent(use_api)
if err:
    st.error(f"Initialization Exception: {err}")
    st.stop()
else:
    st.sidebar.success("Engine Pipeline successfully attached!")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "token_metrics" not in st.session_state:
    st.session_state.token_metrics = None

st.sidebar.divider()
st.sidebar.subheader("Token usage")
token_usage = st.sidebar.empty()

def show_token_usage(metrics):
    if not metrics:
        token_usage.caption("Token usage will appear after the first query.")
        return
    token_usage.table([
        {"Metric": "Current question", "Tokens": metrics["current_total"]},
        {"Metric": "Used today", "Tokens": metrics["today_cumulative"]},
        {"Metric": "Remaining today", "Tokens": metrics["today_remaining"]},
    ])

show_token_usage(st.session_state.token_metrics)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask an HR policy query..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Processing architectural modules..."):
            # Direct call to the exact same engine backend!
            result = agent.ask_question(user_query)
            
            st.markdown(result["answer"])
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            
            st.session_state.token_metrics = result.get("token_metrics")
            show_token_usage(st.session_state.token_metrics)
