"""
VidGenius: YouTube RAG Agent
----------------------------
Main Streamlit interface for transcript analysis, note generation,
and conversational RAG-based chat from YouTube videos.
"""

import streamlit as st
from src.core.rag_pipeline import VidGeniusAgent

# -------------------------------------------------------------------
# Page Config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="VidGenius | YouTube RAG Agent",
    page_icon="🎬",
    layout="wide",
)

# -------------------------------------------------------------------
# Sidebar – User Inputs
# -------------------------------------------------------------------
with st.sidebar:
    st.title("🎬 VidGenius")
    st.markdown("#### Transform YouTube videos into knowledge.")
    st.markdown("---")

    youtube_url = st.text_input(
        "🔗 YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )
    language_code = st.text_input(
        "🌐 Video Language Code",
        placeholder="e.g., en, hi, es, fr",
        value="en",
    )

    task_option = st.radio(
        "🧠 What would you like to do?",
        ["Chat with Video", "Generate Notes"],
    )

    start_button = st.button("✨ Start Processing")

    st.markdown("---")
    st.caption("Built with ❤️ using Gemini & LangChain")

# -------------------------------------------------------------------
# Main Page – Header
# -------------------------------------------------------------------
st.title("🎥 VidGenius: AI-Powered YouTube Understanding")
st.markdown(
    """
    Paste a YouTube video link, and VidGenius will extract its transcript,
    summarize it into structured notes, or let you **chat with the video** itself using RAG.
    """
)
st.divider()

# -------------------------------------------------------------------
# Session Initialization
# -------------------------------------------------------------------
if "agent" not in st.session_state:
    st.session_state.agent = None
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------------------------
# Processing Flow
# -------------------------------------------------------------------
if start_button:
    if not youtube_url:
        st.error("⚠️ Please enter a valid YouTube URL.")
        st.stop()

    # Initialize agent
    st.session_state.agent = VidGeniusAgent(language=language_code)
    agent = st.session_state.agent

    with st.spinner("Step 1/3 🕓 Fetching transcript..."):
        transcript = agent.get_transcript(youtube_url)
        if not transcript:
            st.error("❌ Unable to fetch transcript.")
            st.stop()

    # Translate if needed
    if language_code.lower() != "en":
        with st.spinner("Step 2/3 🌍 Translating transcript to English..."):
            transcript = agent.translate_transcript(transcript)

    # Save transcript in session for later use
    st.session_state.transcript = transcript

    # ---------------------- NOTES MODE ----------------------
    if task_option == "Generate Notes":
        with st.spinner("Step 3/3 🧩 Extracting key topics..."):
            topics = agent.get_important_topics(transcript)
        st.subheader("📘 Key Topics")
        st.markdown(topics)
        st.markdown("---")

        with st.spinner("📝 Generating concise notes..."):
            notes = agent.generate_notes(transcript)
        st.subheader("🧾 Notes Summary")
        st.markdown(notes)

        st.success("✅ Notes generated successfully!")
        st.download_button(
            label="💾 Download Notes (Markdown)",
            data=notes,
            file_name="vidgenius_notes.md",
        )

    # ---------------------- CHAT MODE ----------------------
    elif task_option == "Chat with Video":
        with st.spinner("Step 3/3 🧠 Creating vector store for RAG..."):
            vector_store = agent.create_vector_store(transcript)
            if vector_store:
                st.session_state.vectorstore_ready = True
                st.success("✅ Vector store ready! Start chatting below 👇")
            else:
                st.error("❌ Failed to create vector store.")

# -------------------------------------------------------------------
# Chat Interface
# -------------------------------------------------------------------
if (
    task_option == "Chat with Video"
    and st.session_state.get("vectorstore_ready")
    and st.session_state.get("agent")
):
    st.divider()
    st.subheader("💬 Chat with the Video")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_query = st.chat_input("Ask something about the video...")
    if user_query:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.agent.rag_answer(user_query)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})