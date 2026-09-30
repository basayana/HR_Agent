import os
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
use_groq = (engine_mode == "Groq Cloud (Blazing Fast)")

@st.cache_resource
def get_cached_agent(use_groq):
    """Caches engine objects so swapping selector inputs handles memory resets beautifully."""
    try:
        return HRAgentEngine(use_groq=use_groq), None
    except Exception as e:
        return None, str(e)

agent, err = get_cached_agent(use_groq)
if err:
    st.error(f"Initialization Exception: {err}")
    st.stop()
else:
    st.sidebar.success("Engine Pipeline successfully attached!")

if "messages" not in st.session_state:
    st.session_state.messages = []

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
            
            st.markdown("---")
            st.caption(f"⏱️ **Total latency pipeline runtime:** {result['total_time']:.2f} seconds (Retrieval: {result['retrieval_time']:.2f}s | Generation: {result['generation_time']:.2f}s)")
            
            if result["token_metrics"]:
                m = result["token_metrics"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Question", f"{m['current_total']} tkn")
                col2.metric("Used Today", f"{m['today_cumulative']} tkn")
                col3.metric("Remaining Today", f"{m['today_remaining']} tkn")
