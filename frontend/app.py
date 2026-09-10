import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

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
        margin-bottom: 0;
        line-height: 1.2;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: #9CA3AF;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }
    .source-card {
        background-color: #0D0D0D;
        border: 1px solid #1F1F1F;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #E5E7EB;
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
</style>
""", unsafe_allow_html=True)

# --- Session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📄 DocuMind AI")
    st.caption("Your AI Knowledge Assistant")
    st.markdown("&nbsp;", unsafe_allow_html=True)

    if st.button("➕  New Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_sources = []
        st.rerun()

    st.markdown("---")
    st.markdown("**📤 Upload a Document**")
    uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"], label_visibility="collapsed")

    if uploaded_file is not None:
        if st.button("Process Document", use_container_width=True):
            with st.spinner("Processing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=120)
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['chunks_stored']} chunks stored")
                        st.session_state.uploaded_docs.append(uploaded_file.name)
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the backend server. Is it running?")

    if st.session_state.uploaded_docs:
        st.markdown("---")
        st.markdown("**Recent Documents**")
        for doc_name in st.session_state.uploaded_docs:
            st.markdown(f'<div class="doc-item">📄 {doc_name}</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="user-profile">👤 <b>Your Session</b><br>Local RAG Assistant</div>',
        unsafe_allow_html=True,
    )

# --- Header ---
st.markdown('<p class="app-title" style="margin-top: 10px;">📄 DocuMind AI</p>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Upload a document and ask questions about its content, powered by a local RAG pipeline.</p>', unsafe_allow_html=True)

# --- Main layout: chat (left) + sources (right) ---
chat_col, side_col = st.columns([3, 1])

with chat_col:
    if not st.session_state.chat_history:
        st.info("👋 Upload a document from the sidebar, then ask a question below to get started.")

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            spacer, bubble_col = st.columns([1, 3])
            with bubble_col:
                st.markdown(
                    f'<div style="background-color:#1F1F1F; color:white; padding:0.7rem 1rem; '
                    f'border-radius:14px 14px 2px 14px; text-align:left; margin-bottom:0.6rem; '
                    f'display:inline-block; float:right;">🧑 {message["content"]}</div>'
                    f'<div style="clear:both;"></div>',
                    unsafe_allow_html=True,
                )
        else:
            bubble_col, spacer = st.columns([3, 1])
            with bubble_col:
                st.markdown(
                    f'<div style="background-color:#0A0A0A; color:white; padding:0.7rem 1rem; '
                    f'border-radius:14px 14px 14px 2px; text-align:left; margin-bottom:0.6rem; '
                    f'display:inline-block; float:left; border:1px solid #2D2D2D;">🤖 {message["content"]}</div>'
                    f'<div style="clear:both;"></div>',
                    unsafe_allow_html=True,
                )

with side_col:
    st.markdown('<p style="color:#F3F4F6; font-weight:700;">📎 Sources</p>', unsafe_allow_html=True)
    if st.session_state.last_sources:
        for s in st.session_state.last_sources:
            st.markdown(f'<div class="source-card">📄 {s}</div>', unsafe_allow_html=True)
    else:
        st.caption("Sources for the latest answer will appear here.")

# --- Chat input: kept outside columns so Streamlit pins it to the bottom of the page ---
user_question = st.chat_input("Type your message...")

if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/ask",
                json={"question": user_question},
                timeout=120,
            )
            if response.status_code == 200:
                result = response.json()
                answer = result["answer"]
                sources = result.get("sources", [])
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.last_sources = sources
                st.rerun()
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend server. Is it running?")