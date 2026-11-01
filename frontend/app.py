"""DocuMind AI - Streamlit frontend. Talks to the FastAPI backend over HTTP."""

import html
import uuid

import requests
import streamlit as st

from ui_helpers import (
    css_variables,
    escape_markdown,
    find_chat,
    is_no_info,
    sanitize_markdown,
    sources_html,
    timeline_html,
    trace_summary,
    upsert_chat,
)

API_BASE_URL = "http://127.0.0.1:8000"
UPLOAD_TIMEOUT_SECONDS = 120
ASK_TIMEOUT_SECONDS = 300  # the agent may make several LLM calls per question
ASSISTANT_AVATAR = ":material/auto_awesome:"

# Static styles. Colors come from --dm-* variables, so one stylesheet serves both themes.
BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

.stApp { background: var(--dm-bg); color: var(--dm-text); }
.stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input,
.stApp textarea, .stApp h1, .stApp summary { font-family: 'Inter Tight', system-ui, sans-serif; }
.stApp p, .stApp li, .stApp label, .stApp h1, .stApp summary, .stApp strong { color: var(--dm-text); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 900px; padding-top: 1.5rem; }
hr { border-color: var(--dm-border); }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--dm-side); border-right: 1px solid var(--dm-border); }
.dm-logo { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.dm-logo-mark { width: 32px; height: 32px; border-radius: 8px; background: var(--dm-accent);
    display: flex; align-items: center; justify-content: center; }
.dm-logo-name { font-weight: 700; font-size: 16px; line-height: 1.2; color: var(--dm-text); }
.stApp .dm-logo-sub, .stApp .dm-label, .stApp .dm-hint, .stApp .dm-muted { color: var(--dm-muted); }
.dm-logo-sub { font-size: 12px; }
.stApp .dm-label { font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; margin: 12px 0 4px; }
.stApp .dm-hint { font-size: 12px; text-align: center; margin: -6px 0 4px; }
.dm-doc { display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-radius: 10px;
    background: var(--dm-surface); border: 1px solid var(--dm-border); font-size: 13px;
    margin-bottom: 6px; color: var(--dm-text); }
.dm-doc span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

section[data-testid="stSidebar"] .stButton button { width: 100%; justify-content: flex-start;
    min-height: 40px; border-radius: 10px; font-size: 14px; }
section[data-testid="stSidebar"] .stButton button[kind="secondary"] { background: transparent;
    border: 1px solid transparent; color: var(--dm-muted); }
section[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover { background: var(--dm-surface);
    color: var(--dm-text); }
section[data-testid="stSidebar"] .stButton button[kind="primary"] { background: var(--dm-bubble);
    border: 1px solid var(--dm-border2); color: var(--dm-text); }
section[data-testid="stSidebar"] .stButton button p { color: inherit; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; }

/* Compact uploader: one pill button, no big drop box */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { padding: 0; border: 0;
    background: transparent; min-height: 0; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderPagination"] { display: none; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button { width: 100%; height: 40px;
    border-radius: 999px; border: 1px solid var(--dm-border2); background: var(--dm-surface); font-size: 0; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button::after {
    content: "Upload PDF or DOCX"; font-size: 14px; font-weight: 500; color: var(--dm-text); }

/* Header buttons in the main area */
[data-testid="stMain"] .stButton button, section.main .stButton button { background: transparent;
    border: 1px solid var(--dm-border2); color: var(--dm-text); border-radius: 10px; }
.stApp h1.dm-title { font-size: 20px; font-weight: 600; margin: 0; padding: 0; }
.dm-empty { border: 1px dashed var(--dm-border2); border-radius: 12px; padding: 18px;
    color: var(--dm-muted); font-size: 14px; margin-top: 16px; }

/* Chat: user messages are right-aligned bubbles without an avatar */
[data-testid="stChatMessage"] { background: transparent; padding: 0; gap: 14px; margin-bottom: 12px; }
.stElementContainer:has(.dm-user), .element-container:has(.dm-user) { display: none; }
[data-testid="stChatMessage"]:has(.dm-user) { flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has(.dm-user) [data-testid^="stChatMessageAvatar"] { display: none; }
[data-testid="stChatMessage"]:has(.dm-user) [data-testid="stChatMessageContent"] { flex: 0 1 auto;
    max-width: 70%; background: var(--dm-bubble); border-radius: 16px 16px 4px 16px; padding: 12px 16px; }
[data-testid="stChatMessage"]:has(.dm-user) p { margin: 0; }
[data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] { background: transparent;
    border: 1px solid var(--dm-border2); border-radius: 8px; color: var(--dm-accent); }
.stApp p.dm-noinfo { color: var(--dm-muted); }

/* Sources and agent timeline */
.dm-sources { display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 8px; }
.dm-chip { font-size: 12px; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--dm-border2);
    color: var(--dm-muted); }
[data-testid="stExpander"] details { background: var(--dm-side); border: 1px solid var(--dm-border);
    border-radius: 12px; }
[data-testid="stExpander"] summary p { font-size: 13px; font-weight: 600; }
.dm-steps { list-style: none; margin: 0; padding: 4px 0; display: flex; flex-direction: column; gap: 10px; }
.dm-steps li { display: grid; grid-template-columns: 14px 120px 1fr auto; gap: 10px; align-items: center;
    margin: 0; font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: var(--dm-text); }
.dm-dot { width: 8px; height: 8px; border-radius: 4px; background: var(--dm-accent); }
.dm-dot-ok { background: var(--dm-ok); }
.dm-dot-warn { background: var(--dm-warn); }

/* Chat input */
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: var(--dm-bg); }
[data-testid="stChatInput"] { background: var(--dm-input); border: 1px solid var(--dm-border2); border-radius: 14px; }
[data-testid="stChatInput"] textarea { color: var(--dm-text); background: transparent; }
[data-testid="stChatInputSubmitButton"] { background: var(--dm-accent); border-radius: 10px; }
[data-testid="stChatInputSubmitButton"] svg { color: #0B0B0C; fill: #0B0B0C; }
</style>
"""

LOGO_HTML = """
<div class="dm-logo">
  <div class="dm-logo-mark">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0B0B0C" stroke-width="2.2" aria-hidden="true">
      <path d="M6 3h9l4 4v14H6z"/><path d="M14 3v5h5"/><path d="M9 13h6M9 17h4"/>
    </svg>
  </div>
  <div><div class="dm-logo-name">DocuMind AI</div><div class="dm-logo-sub">Agentic RAG</div></div>
</div>
"""


# ---------- Backend calls ----------

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
            f"{API_BASE_URL}/ask", json={"question": question}, timeout=ASK_TIMEOUT_SECONDS
        )
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to the backend server. Is it running?"
    except requests.exceptions.Timeout:
        return False, "The agent took too long to answer. Please try again."
    if response.status_code != 200:
        return False, f"Error: {get_error_detail(response)}"
    return True, response.json()


def upload_backend(uploaded_file):
    """Send a file to the API. Returns (True, message) or (False, error message)."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        response = requests.post(
            f"{API_BASE_URL}/upload", files=files, timeout=UPLOAD_TIMEOUT_SECONDS
        )
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to the backend server. Is it running?"
    except requests.exceptions.Timeout:
        return False, "Processing took too long. Please try again."
    if response.status_code != 200:
        return False, f"Error: {get_error_detail(response)}"
    return True, f"{response.json().get('chunks_stored', 0)} chunks stored"


# ---------- Session state ----------

def new_chat_id() -> str:
    return uuid.uuid4().hex[:8]


def init_state() -> None:
    defaults = {
        "theme": "dark",
        "chats": [],               # saved chats, newest first (this browser session only)
        "current_chat_id": new_chat_id(),
        "chat_history": [],        # messages of the open chat
        "uploaded_docs": [],
        "processed_uploads": [],   # files already sent, so reruns don't upload twice
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_current_chat() -> None:
    st.session_state.chats = upsert_chat(
        st.session_state.chats, st.session_state.current_chat_id, st.session_state.chat_history
    )


def start_new_chat() -> None:
    save_current_chat()
    st.session_state.current_chat_id = new_chat_id()
    st.session_state.chat_history = []


def open_chat(chat_id: str) -> None:
    save_current_chat()
    chat = find_chat(st.session_state.chats, chat_id)
    if chat is not None:
        st.session_state.current_chat_id = chat_id
        st.session_state.chat_history = list(chat["messages"])


def toggle_theme() -> None:
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"


def handle_upload(uploaded_file) -> None:
    """Upload a newly selected file once, then list it under Documents."""
    upload_key = f"{uploaded_file.name}:{uploaded_file.size}"
    if upload_key in st.session_state.processed_uploads:
        return
    st.session_state.processed_uploads.append(upload_key)
    with st.spinner("Processing document..."):
        ok, message = upload_backend(uploaded_file)
    if ok:
        if uploaded_file.name not in st.session_state.uploaded_docs:
            st.session_state.uploaded_docs.append(uploaded_file.name)
        st.toast(f"{uploaded_file.name}: {message}")
    else:
        st.error(message)


# ---------- Rendering ----------

def theme_button(key: str) -> None:
    is_dark = st.session_state.theme == "dark"
    st.button(
        "Light mode" if is_dark else "Dark mode",
        icon=":material/light_mode:" if is_dark else ":material/dark_mode:",
        key=key,
        on_click=toggle_theme,
        use_container_width=True,
    )


def render_sidebar() -> bool:
    """Draw the sidebar. Returns whether agent details should be shown."""
    with st.sidebar:
        st.markdown(LOGO_HTML, unsafe_allow_html=True)
        st.button("New chat", icon=":material/add:", type="primary",
                  on_click=start_new_chat, use_container_width=True)

        uploaded_file = st.file_uploader(
            "Upload PDF or DOCX", type=["pdf", "docx"], label_visibility="collapsed"
        )
        st.markdown('<p class="dm-hint">Max 20 MB</p>', unsafe_allow_html=True)
        if uploaded_file is not None:
            handle_upload(uploaded_file)

        if st.session_state.uploaded_docs:
            st.markdown('<p class="dm-label">Documents</p>', unsafe_allow_html=True)
            for name in st.session_state.uploaded_docs:
                # Filenames come from users: escape them before putting them in HTML
                st.markdown(f'<div class="dm-doc"><span>{html.escape(name)}</span></div>',
                            unsafe_allow_html=True)

        st.markdown('<p class="dm-label">History</p>', unsafe_allow_html=True)
        if not st.session_state.chats:
            st.markdown('<p class="dm-hint" style="text-align:left">No chats yet</p>',
                        unsafe_allow_html=True)
        for chat in st.session_state.chats:
            is_open = chat["id"] == st.session_state.current_chat_id
            st.button(
                escape_markdown(chat["title"]),
                key=f"chat_{chat['id']}",
                type="primary" if is_open else "secondary",
                on_click=open_chat,
                args=(chat["id"],),
                use_container_width=True,
            )

        st.markdown("---")
        show_details = st.toggle("Show agent details", value=False, key="show_agent_details")
        theme_button("theme_sidebar")
    return show_details


def render_message(message: dict, show_details: bool) -> None:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown('<span class="dm-user"></span>', unsafe_allow_html=True)
            st.markdown(sanitize_markdown(message["content"]))
        return

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        answer = message.get("content", "")
        if is_no_info(answer):
            st.markdown(f'<p class="dm-noinfo">{html.escape(answer)}</p>', unsafe_allow_html=True)
        else:
            st.markdown(sanitize_markdown(answer))

        chips = sources_html(message.get("sources"))
        if chips:
            st.markdown(chips, unsafe_allow_html=True)

        trace = message.get("trace") or []
        if show_details and trace:
            with st.expander(f"Agent details · {trace_summary(trace)}", expanded=False):
                st.markdown(timeline_html(trace), unsafe_allow_html=True)


# ---------- Page ----------

st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide")
init_state()

st.markdown(f"<style>{css_variables(st.session_state.theme)}</style>", unsafe_allow_html=True)
st.markdown(BASE_CSS, unsafe_allow_html=True)

show_details = render_sidebar()

title_col, theme_col = st.columns([5, 1], vertical_alignment="center")
with title_col:
    st.markdown('<h1 class="dm-title">Ask your documents</h1>', unsafe_allow_html=True)
with theme_col:
    theme_button("theme_header")

if not st.session_state.chat_history:
    st.markdown(
        '<div class="dm-empty">Upload a document from the sidebar, then ask a question below.</div>',
        unsafe_allow_html=True,
    )

for message in st.session_state.chat_history:
    render_message(message, show_details)

user_question = st.chat_input("Ask about your documents…")

if user_question:
    render_message({"role": "user", "content": user_question}, show_details)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
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
        save_current_chat()
        st.rerun()