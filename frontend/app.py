"""DocuMind AI - Streamlit frontend.

Direct mode: the app calls the shared service in the same process, so no
separate API server is needed. This is how it runs on Streamlit Cloud.
"""

import html
import logging
import sys
import uuid
from pathlib import Path

import streamlit as st

# Allow importing the backend modules from app/core
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "core"))

from service import DocuMindService, ServiceError  # noqa: E402
from session_helpers import (  # noqa: E402
    PRIVACY_NOTE,
    SAMPLE_QUESTIONS,
    client_ip_from_headers,
    usage_line,
)
from ui_helpers import (  # noqa: E402
    css_variables,
    doc_item_html,
    escape_markdown,
    find_chat,
    is_no_info,
    sanitize_markdown,
    sources_html,
    timeline_html,
    trace_summary,
    upsert_chat,
    user_bubble_html,
)

logger = logging.getLogger(__name__)

# Inline SVG mask for the upload icon (black = visible part of the mask)
_UPLOAD_MASK = (
    "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
    "fill='none' stroke='black' stroke-width='2'><path d='M12 16V4M7 9l5-5 5 5'/>"
    "<path d='M4 16v4h16v-4'/></svg>\")"
)

# Static styles. Colors come from --dm-* variables, so one stylesheet serves both themes.
# Widgets are targeted by their key (Streamlit adds a "st-key-<key>" class).
BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

.stApp { background: var(--dm-bg); color: var(--dm-text); }
.stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input,
.stApp textarea, .stApp h1, .stApp summary { font-family: 'Inter Tight', system-ui, sans-serif; }
.stApp p, .stApp li, .stApp label, .stApp h1, .stApp summary, .stApp strong { color: var(--dm-text); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 900px; padding-top: 1.5rem; padding-bottom: 7rem; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] { background: var(--dm-side); border-right: 1px solid var(--dm-border);
    min-width: 300px; }
[data-testid="stSidebarHeader"] { height: 0; min-height: 0; padding: 0; }
[data-testid="stSidebarUserContent"] { padding-top: 22px; }
[data-testid="stSidebarCollapseButton"] { display: flex !important; visibility: visible !important;
    opacity: 1 !important; position: absolute; top: 22px; right: 16px; z-index: 100; }
[data-testid="stSidebarCollapsedControl"] { display: flex !important; visibility: visible !important;
    opacity: 1 !important; }
[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapsedControl"] button {
    width: 36px; height: 36px; border-radius: 8px; border: 1px solid var(--dm-border2) !important;
    background: var(--dm-surface) !important; }
[data-testid="stSidebarCollapseButton"] *, [data-testid="stSidebarCollapsedControl"] * {
    color: var(--dm-text) !important; }

.dm-logo { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.dm-logo-mark { width: 32px; height: 32px; border-radius: 8px; background: var(--dm-accent);
    display: flex; align-items: center; justify-content: center; }
.dm-logo-name { font-weight: 700; font-size: 16px; line-height: 1.2; color: var(--dm-text); }
.stApp .dm-logo-sub, .stApp .dm-label, .stApp .dm-hint, .stApp .dm-muted, .dm-doc-sub,
.stApp .dm-usage, .stApp .dm-privacy { color: var(--dm-muted); }
.dm-logo-sub { font-size: 12px; }
.stApp .dm-label { font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; margin: 14px 0 6px; }
.stApp .dm-hint { font-size: 12px; text-align: center; margin: -6px 0 4px; }
.stApp .dm-usage { font-size: 12px; margin: 2px 0 8px; }
.stApp .dm-privacy { font-size: 12px; line-height: 1.45; margin: 8px 0 4px; }
.dm-doc { display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-radius: 10px;
    background: var(--dm-surface); border: 1px solid var(--dm-border); margin-bottom: 6px;
    color: var(--dm-accent); }
.dm-doc-text { min-width: 0; }
.dm-doc-name { font-size: 13px; color: var(--dm-text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.dm-doc-sub { font-size: 11px; }

/* Buttons */
.stApp .stButton button { border-radius: 10px; min-height: 40px; box-shadow: none; }
.stApp .stButton button p { color: inherit; font-size: 14px; }
.st-key-new_chat button { width: 100%; justify-content: flex-start; background: var(--dm-surface) !important;
    border: 1px solid var(--dm-border2) !important; color: var(--dm-text) !important; }
[class*="st-key-chat_"] button { width: 100%; justify-content: flex-start; background: transparent !important;
    border: 1px solid transparent !important; color: var(--dm-muted) !important; }
[class*="st-key-chat_"] button:hover { background: var(--dm-surface) !important; color: var(--dm-text) !important; }
[class*="st-key-chat_"] button[kind="primary"] { background: var(--dm-bubble) !important; color: var(--dm-text) !important; }
[class*="st-key-chat_"] button div, [class*="st-key-chat_"] button p { min-width: 0; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }
[class*="st-key-sample_"] button { width: 100%; justify-content: flex-start; text-align: left;
    background: var(--dm-surface) !important; border: 1px solid var(--dm-border2) !important;
    color: var(--dm-text) !important; }
.st-key-delete_files button { width: 100%; justify-content: flex-start; background: transparent !important;
    border: 1px solid var(--dm-border) !important; color: var(--dm-muted) !important; }
.st-key-theme_sidebar button { width: 100%; justify-content: flex-start; min-height: 44px;
    background: transparent !important; border: 1px solid var(--dm-border) !important; color: var(--dm-text) !important; }
.st-key-theme_header { display: flex; justify-content: flex-end; }
.st-key-theme_header button { width: 44px; height: 44px; padding: 0; background: transparent !important;
    border: 1px solid var(--dm-border2) !important; color: var(--dm-text) !important; }
.st-key-show_agent_details { border: 1px solid var(--dm-border); border-radius: 10px; padding: 10px 12px; }

/* Compact uploader: one pill button with an icon */
[class*="st-key-uploader"] [data-testid="stFileUploaderDropzone"] { padding: 0 !important; border: 0 !important;
    background: transparent !important; min-height: 0; }
[class*="st-key-uploader"] [data-testid="stFileUploaderDropzoneInstructions"],
[class*="st-key-uploader"] [data-testid="stFileUploaderFile"],
[class*="st-key-uploader"] [data-testid="stFileUploaderPagination"] { display: none !important; }
[class*="st-key-uploader"] [data-testid="stFileUploaderDropzone"] button { width: 100%; height: 42px; border-radius: 999px;
    border: 1px solid var(--dm-border2) !important; background: var(--dm-surface) !important; font-size: 0;
    display: flex; align-items: center; justify-content: center; gap: 8px; }
[class*="st-key-uploader"] [data-testid="stFileUploaderDropzone"] button::before { content: ""; width: 15px; height: 15px;
    background-color: var(--dm-text); -webkit-mask: UPLOAD_MASK center / contain no-repeat;
    mask: UPLOAD_MASK center / contain no-repeat; }
[class*="st-key-uploader"] [data-testid="stFileUploaderDropzone"] button::after { content: "Upload PDF or DOCX";
    font-size: 14px; font-weight: 500; color: var(--dm-text); }

/* ---------- Main area ---------- */
.stApp h1.dm-title { font-size: 20px; font-weight: 600; margin: 0; padding: 0; }
.dm-divider { height: 1px; background: var(--dm-border); margin: 8px 0 24px; }
.dm-empty { border: 1px dashed var(--dm-border2); border-radius: 12px; padding: 18px;
    color: var(--dm-muted); font-size: 14px; margin-bottom: 8px; }
.dm-user-row { display: flex; justify-content: flex-end; margin: 4px 0 18px; }
.dm-user-bubble { max-width: 70%; background: var(--dm-bubble); color: var(--dm-text);
    border-radius: 16px 16px 4px 16px; padding: 12px 16px; font-size: 15px; line-height: 1.5;
    overflow-wrap: anywhere; }
.dm-avatar { width: 32px; height: 32px; border-radius: 8px; border: 1px solid var(--dm-border2);
    display: flex; align-items: center; justify-content: center; color: var(--dm-accent); margin-top: 2px; }
.stApp p.dm-noinfo { color: var(--dm-muted); font-size: 15px; margin: 4px 0 18px; }

/* Sources and agent timeline */
.dm-sources { display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 8px; }
.dm-chip { font-size: 12px; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--dm-border2);
    color: var(--dm-muted); }
[data-testid="stExpander"] details { background: var(--dm-side) !important; border: 1px solid var(--dm-border) !important;
    border-radius: 12px; }
[data-testid="stExpander"] summary p { font-size: 13px; font-weight: 600; }
[data-testid="stExpander"] summary svg { color: var(--dm-text); }
.dm-steps { list-style: none; margin: 0; padding: 4px 0; display: flex; flex-direction: column; gap: 10px; }
.dm-steps li { display: grid; grid-template-columns: 14px 120px 1fr auto; gap: 10px; align-items: center;
    margin: 0; font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: var(--dm-text); }
.dm-dot { width: 8px; height: 8px; border-radius: 4px; background: var(--dm-accent); }
.dm-dot-ok { background: var(--dm-ok); }
.dm-dot-warn { background: var(--dm-warn); }

/* Chat input */
[data-testid="stBottom"], [data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] { background: var(--dm-bg) !important; }
[data-testid="stBottomBlockContainer"] { max-width: 900px; margin: 0 auto; padding-bottom: 24px; }
[data-testid="stChatInput"], [data-testid="stChatInput"] > div { background: var(--dm-input) !important;
    border-color: var(--dm-border2) !important; border-radius: 14px !important; }
[data-testid="stChatInput"] textarea { color: var(--dm-text) !important; background: transparent !important;
    caret-color: var(--dm-text); }
[data-testid="stChatInput"] textarea::placeholder { color: var(--dm-muted) !important; opacity: 1; }
[data-testid="stChatInputSubmitButton"] { background: var(--dm-accent) !important; border-radius: 10px !important; }
[data-testid="stChatInputSubmitButton"] svg { color: #0B0B0C !important; fill: #0B0B0C !important; }
</style>
""".replace("UPLOAD_MASK", _UPLOAD_MASK)

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

SPARKLE_SVG = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'aria-hidden="true"><path d="M12 3l2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5z"/></svg>'
)

GENERIC_ERROR = "Something went wrong. Please try again."


# ---------- Service (one per server process, shared by all visitors) ----------

@st.cache_resource(show_spinner="Loading the demo document...")
def get_service() -> DocuMindService:
    service = DocuMindService()
    service.load_demo_documents()
    return service


# ---------- Session state ----------

def new_chat_id() -> str:
    return uuid.uuid4().hex[:8]


def init_state() -> None:
    defaults = {
        "theme": "dark",
        "chats": [],                       # saved chats, newest first (this browser session only)
        "current_chat_id": new_chat_id(),
        "chat_history": [],                # messages of the open chat
        "processed_uploads": [],           # files already sent, so reruns don't upload twice
        "uploader_key": 0,                 # changing it resets the file picker
        "visitor_session": uuid.uuid4().hex,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def visitor_ids() -> tuple[str, str]:
    """(document id, quota id): documents follow the browser session, quotas follow the IP."""
    doc_id = st.session_state.visitor_session
    headers = getattr(getattr(st, "context", None), "headers", None)
    return doc_id, client_ip_from_headers(headers) or doc_id


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


def queue_question(question: str) -> None:
    st.session_state.pending_question = question


def delete_my_files() -> None:
    get_service().end_session(st.session_state.visitor_session)
    st.session_state.processed_uploads = []
    st.session_state.uploader_key += 1      # clear the file picker so nothing is re-uploaded
    st.session_state.files_deleted = True


def handle_upload(service: DocuMindService, uploaded_file, doc_id: str, quota_id: str) -> None:
    """Upload a newly selected file once."""
    upload_key = f"{uploaded_file.name}:{uploaded_file.size}"
    if upload_key in st.session_state.processed_uploads:
        return
    st.session_state.processed_uploads.append(upload_key)
    with st.spinner("Checking and processing your document..."):
        try:
            result = service.upload(doc_id, uploaded_file.name, uploaded_file.getvalue(), quota_id=quota_id)
        except ServiceError as error:
            st.error(error.message)
            return
        except Exception:
            logger.exception("Upload failed")
            st.error(GENERIC_ERROR)
            return
    st.toast(f"{result['filename']}: {result['chunks_stored']} chunks stored")


# ---------- Rendering ----------

def render_sidebar(service: DocuMindService, doc_id: str, quota_id: str) -> bool:
    """Draw the sidebar. Returns whether agent details should be shown."""
    with st.sidebar:
        st.markdown(LOGO_HTML, unsafe_allow_html=True)
        st.button("New chat", icon=":material/add:", key="new_chat",
                  on_click=start_new_chat, use_container_width=True)

        uploaded_file = st.file_uploader(
            "Upload PDF or DOCX", type=["pdf", "docx"],
            key=f"uploader_{st.session_state.uploader_key}", label_visibility="collapsed",
        )
        st.markdown('<p class="dm-hint">PDF or DOCX · max 5 MB · 30 pages</p>', unsafe_allow_html=True)
        if uploaded_file is not None:
            handle_upload(service, uploaded_file, doc_id, quota_id)

        st.markdown('<p class="dm-label">Documents</p>', unsafe_allow_html=True)
        for name in service.documents(doc_id):
            st.markdown(doc_item_html(name), unsafe_allow_html=True)
        usage = usage_line(service.usage(doc_id, quota_id=quota_id))
        st.markdown(f'<p class="dm-usage">{html.escape(usage)}</p>', unsafe_allow_html=True)

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

        st.markdown('<div style="height: 24px"></div>', unsafe_allow_html=True)
        show_details = st.toggle("Show agent details", value=False, key="show_agent_details")
        is_dark = st.session_state.theme == "dark"
        st.button(
            "Light mode" if is_dark else "Dark mode",
            icon=":material/light_mode:" if is_dark else ":material/dark_mode:",
            key="theme_sidebar",
            on_click=toggle_theme,
            use_container_width=True,
        )
        st.button("Delete my files now", icon=":material/delete:", key="delete_files",
                  on_click=delete_my_files, use_container_width=True)
        st.markdown(f'<p class="dm-privacy">{html.escape(PRIVACY_NOTE)}</p>', unsafe_allow_html=True)
    return show_details


def render_header() -> None:
    is_dark = st.session_state.theme == "dark"
    title_col, theme_col = st.columns([12, 1], vertical_alignment="center")
    with title_col:
        st.markdown('<h1 class="dm-title">Ask your documents</h1>', unsafe_allow_html=True)
    with theme_col:
        st.button(
            "",
            icon=":material/light_mode:" if is_dark else ":material/dark_mode:",
            key="theme_header",
            help="Switch to light mode" if is_dark else "Switch to dark mode",
            on_click=toggle_theme,
        )
    st.markdown('<div class="dm-divider"></div>', unsafe_allow_html=True)


def render_empty_state() -> None:
    st.markdown(
        '<div class="dm-empty">Ask about the Velora Travel demo handbook (a fictional company), '
        'or upload your own PDF or DOCX from the sidebar.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="dm-label">Try a sample question</p>', unsafe_allow_html=True)
    for i, question in enumerate(SAMPLE_QUESTIONS):
        st.button(question, key=f"sample_{i}", on_click=queue_question, args=(question,),
                  use_container_width=True)


def render_user(text: str) -> None:
    st.markdown(user_bubble_html(text), unsafe_allow_html=True)


def render_assistant(message: dict, show_details: bool) -> None:
    answer = message.get("content", "")
    no_info = is_no_info(answer)
    avatar_col, body_col = st.columns([1, 15], gap="small")
    with avatar_col:
        st.markdown(f'<div class="dm-avatar">{"" if no_info else SPARKLE_SVG}</div>',
                    unsafe_allow_html=True)
    with body_col:
        if no_info:
            st.markdown(f'<p class="dm-noinfo">{html.escape(answer)}</p>', unsafe_allow_html=True)
            return
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

service = get_service()
doc_id, quota_id = visitor_ids()

if st.session_state.pop("files_deleted", False):
    st.toast("Your files were deleted. The demo handbook is still available.")

show_details = render_sidebar(service, doc_id, quota_id)
render_header()

if not st.session_state.chat_history:
    render_empty_state()

for message in st.session_state.chat_history:
    if message["role"] == "user":
        render_user(message["content"])
    else:
        render_assistant(message, show_details)

typed_question = st.chat_input("Ask about your documents…")
user_question = typed_question or st.session_state.pop("pending_question", None)

if user_question:
    render_user(user_question)
    payload = None
    avatar_col, body_col = st.columns([1, 15], gap="small")
    with avatar_col:
        st.markdown(f'<div class="dm-avatar">{SPARKLE_SVG}</div>', unsafe_allow_html=True)
    with body_col:
        with st.spinner("The agent is planning, searching and checking..."):
            try:
                payload = service.ask(doc_id, user_question, quota_id=quota_id)
            except ServiceError as error:
                st.error(error.message)
            except Exception:
                logger.exception("Question failed")
                st.error(GENERIC_ERROR)

    if payload is not None:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": payload.get("answer", ""),
            "sources": payload.get("sources", []),
            "trace": payload.get("trace", []),
        })
        save_current_chat()
        st.rerun()