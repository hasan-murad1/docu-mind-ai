"""DocuMind AI - Streamlit frontend. Talks to the FastAPI backend over HTTP."""

import html

import requests
import streamlit as st

from ui_helpers import escape_markdown, sanitize_markdown, timeline_lines, trace_summary

API_BASE_URL = "http://127.0.0.1:8000"
UPLOAD_TIMEOUT_SECONDS = 120
ASK_TIMEOUT_SECONDS = 300  # the agent may make several LLM calls per question


def get_error_detail(response) -> str:
    """Safely read an error message even if the server didn't return JSON."""
    try:
        return response.json().get("detail", "Unknown error")
    except ValueError:
        return f"Server error ({response.status_code})"


def ask_backend(question: str):
    """Send a question to the API. Returns (True, result) or (False, error message)."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"question": question},
            timeout=ASK_TIMEOUT_SECONDS,
        )
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to the backend server. Is it running?"
    except requests.exceptions.Timeout:
        return False, "The agent took too long to answer. Please try again."

    if response.status_code != 200:
        return False, f"Error: {get_error_detail(response)}"
    return True, response.json()


def render_answer_extras(message: dict, show_details: bool) -> None:
    """Show the sources under an answer, and the agent timeline if enabled."""
    sources = message.get("sources") or []
    if sources:
        st.caption("📎 " + " · ".join(escape_markdown(s) for s in sources))

    trace = message.get("trace") or []
    if show_details and trace:
        with st.expander(f"🔍 Agent details ({trace_summary(trace)})", expanded=False):
            st.markdown("  \n".join(timeline_lines(trace)))


st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
    }
    .block-container { padding-top: 2rem; }

    section[data-testid="stSidebar"] {
        background-color: #0D0D0D;
        border-right: 1px solid #1F1F1F;
    }
    section[data-testid="stSidebar"] * {
        color: #E5E7EB !important;
    }
    section[data-testid="stSidebar"] .stButton button {
        background-color: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        color: #FFFFFF !important;
        text-align: left;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background-color: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.3);
    }

    .app-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #F3F4F6;
        margin-top: 10px;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: #9CA3AF;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }
    .doc-item {
        background-color: rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
        font-size: 0.8rem;
    }
    .user-profile {
        background-color: rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        margin-top: 1rem;
        font-size: 0.8rem;
    }

    /* Native chat messages on the dark background */
    [data-testid="stChatMessage"] {
        background-color: #0D0D0D;
        border: 1px solid #1F1F1F;
        border-radius: 12px;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] strong {
        color: #E5E7EB;
    }

    /* Agent details expander */
    [data-testid="stExpander"] details {
        background-color: #0A0A0A;
        border: 1px solid #2D2D2D;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] p {
        color: #E5E7EB;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# --- Session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📄 DocuMind AI")
    st.caption("Your AI Knowledge Assistant")
    st.markdown("&nbsp;", unsafe_allow_html=True)

    if st.button("➕  New Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("**📤 Upload a Document**")
    uploaded_file = st.file_uploader(
        "Choose a PDF or DOCX file",
        type=["pdf", "docx"],
        label_visibility="collapsed",
    )
    st.caption("📁 Drag and drop file here • Limit 20MB • PDF, DOCX")

    if uploaded_file is not None:
        if st.button("Process Document", use_container_width=True):
            with st.spinner("Processing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(
                        f"{API_BASE_URL}/upload", files=files, timeout=UPLOAD_TIMEOUT_SECONDS
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['chunks_stored']} chunks stored")
                        if uploaded_file.name not in st.session_state.uploaded_docs:
                            st.session_state.uploaded_docs.append(uploaded_file.name)
                    else:
                        st.error(f"Error: {get_error_detail(response)}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the backend server. Is it running?")
                except requests.exceptions.Timeout:
                    st.error("Processing took too long. Please try again.")

    st.markdown("---")
    show_details = st.toggle("🧠 Show agent details", value=False, key="show_agent_details")
    st.caption("See how the agent planned, searched and checked each answer.")

    if st.session_state.uploaded_docs:
        st.markdown("---")
        st.markdown("**Recent Documents**")
        for doc_name in st.session_state.uploaded_docs:
            # Filenames come from users: escape them before putting them in HTML
            st.markdown(
                f'<div class="doc-item">📄 {html.escape(doc_name)}</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="user-profile">👤 <b>Your Session</b><br>Agentic RAG Assistant</div>',
        unsafe_allow_html=True,
    )

# --- Header ---
st.markdown('<p class="app-title">📄 DocuMind AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Upload a document and ask questions. An AI agent plans, '
    'searches, and checks every answer against your documents.</p>',
    unsafe_allow_html=True,
)

# --- Chat history (native chat: markdown works, raw HTML is not rendered) ---
if not st.session_state.chat_history:
    st.info("👋 Upload a document from the sidebar, then ask a question below to get started.")

for message in st.session_state.chat_history:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(sanitize_markdown(message["content"]))
        if message["role"] == "assistant":
            render_answer_extras(message, show_details)

# --- Chat input: pinned to the bottom of the page ---
user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(sanitize_markdown(user_question))

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("The agent is planning, searching and checking..."):
            ok, payload = ask_backend(user_question)
        if not ok:
            st.error(payload)

    if ok:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": payload.get("answer", ""),
            "sources": payload.get("sources", []),
            "trace": payload.get("trace", []),
        })
        st.rerun()