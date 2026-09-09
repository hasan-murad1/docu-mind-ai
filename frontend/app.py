import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="DocuMind AI", page_icon="📄")
st.title("📄 DocuMind AI")
st.caption("Upload a document and ask questions about it.")

# --- Session state to keep chat history while the app is open ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: Document Upload ---
with st.sidebar:
    st.header("Upload a Document")
    uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])

    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=120)

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['message']} ({result['chunks_stored']} chunks stored)")
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the backend server. Is it running?")

# --- Main area: Chat interface ---
st.header("Ask a Question")

# Display past chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_question = st.chat_input("Ask something about your uploaded documents...")

if user_question:
    # Show user's question immediately
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    # Get answer from backend
    with st.chat_message("assistant"):
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

                    st.write(answer)
                    if sources:
                        st.caption(f"📎 Sources: {', '.join(sources)}")

                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                else:
                    error_msg = f"Error: {response.json().get('detail', 'Unknown error')}"
                    st.error(error_msg)

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend server. Is it running?")