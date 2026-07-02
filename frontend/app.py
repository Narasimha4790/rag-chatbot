import streamlit as st
import requests
import os

# Backend API URL config
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Modular RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom premium CSS styling (Rich Aesthetics)
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
    }
    
    /* Background and container gradients */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(98, 114, 240, 0.04) 0%, rgba(189, 137, 244, 0.04) 90.2%);
    }
    
    /* Hide Deploy button and default header */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Hide sidebar element completely */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    
    /* Hide file uploader size and format instructions */
    [data-testid*="DropzoneInstructions"] {
        display: none !important;
    }
    
    /* Premium Header styling */
    .header-container {
        padding: 0.5rem 0;
        background: linear-gradient(135deg, #6272F0 0%, #BD89F4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Interactive Button transitions */
    .stButton>button {
        background: linear-gradient(135deg, #6272F0 0%, #BD89F4 100%);
        color: white !important;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(98, 114, 240, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(98, 114, 240, 0.4);
    }
    
    /* Source highlight cards */
    .source-card {
        padding: 1rem;
        background-color: rgba(255, 255, 255, 0.6);
        border-radius: 8px;
        border-left: 4px solid #6272F0;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
        font-size: 0.9rem;
        color: #1F2937;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# Main Title Header and Controls on main screen
col1, col2 = st.columns([7, 3])
with col1:
    st.markdown('<div class="header-container"><h1>🤖 Upload your documents and find answers for your queries</h1></div>', unsafe_allow_html=True)
with col2:
    st.write("")  # Vertical spacers to align with header
    st.write("")
    if st.button("🗑️ Clear Vector Index", use_container_width=True):
        with st.spinner("Clearing collection..."):
            try:
                res = requests.post(f"{BACKEND_URL}/api/clear")
                if res.status_code == 200:
                    st.success("Vector database successfully cleared.")
                    st.session_state.messages = []  # Reset session chat
                    st.session_state.processed_file = None  # Reset active file status
                else:
                    st.error(f"Error: {res.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")

# Document Ingestion on main screen
uploaded_file = st.file_uploader(
    "Choose a file", 
    type=["pdf", "docx", "txt", "csv"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    if st.session_state.processed_file != uploaded_file.name:
        with st.spinner("Parsing and embedding document..."):
            # Forward file stream to FastAPI endpoint
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                res = requests.post(f"{BACKEND_URL}/api/upload", files=files)
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.processed_file = uploaded_file.name
                    st.success(f"Success! '{data['filename']}' indexed into {data['chunks_count']} chunks.")
                else:
                    st.error(f"Error: {res.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
    else:
        st.success(f"Active Document: '{uploaded_file.name}' is indexed.")
else:
    st.session_state.processed_file = None

st.markdown("---")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Grounded Sources Used"):
                for src in msg["sources"]:
                    st.markdown(
                        f"<div class='source-card'><b>File:</b> {src['file']}<br/><b>Excerpt:</b><br/>{src['text']}</div>", 
                        unsafe_allow_html=True
                    )

# Input for new question
if user_question := st.chat_input("Ask a question about your documents..."):
    # Append to state and display question
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)
        
    # Generate RAG response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Searching vector store and generating answer..."):
            try:
                res = requests.post(
                    f"{BACKEND_URL}/api/ask",
                    json={"question": user_question, "n_results": 4}
                )
                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    
                    # Display response
                    message_placeholder.write(answer)
                    
                    # Display sources inside expander
                    if sources:
                        with st.expander("📚 Grounded Sources Used"):
                            for src in sources:
                                st.markdown(
                                    f"<div class='source-card'><b>File:</b> {src['file']}<br/><b>Excerpt:</b><br/>{src['text']}</div>", 
                                    unsafe_allow_html=True
                                )
                    
                    # Store in session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    err_msg = res.json().get("detail", "Error generating response.")
                    message_placeholder.error(err_msg)
            except Exception as e:
                message_placeholder.error(f"Failed to connect to backend: {e}")
