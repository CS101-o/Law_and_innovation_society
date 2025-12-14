import base64
import os

import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from streamlit_mic_recorder import mic_recorder

from refiner import QueryRefiner

from src.tts import text_to_speech_bytes, speech_to_text_from_audio_bytes

# --- CONFIGURATION ---
MODEL_NAME = "uk-lawyer"
DATA_FOLDER = "./legal_docs"


def _render_audio_player(audio_bytes: bytes):
    """Render an autoplaying audio player for cached speech."""
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    audio_html = f"""
        <audio autoplay controls>
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


def _handle_read_aloud(idx: int):
    st.session_state.play_audio_idx = idx

@st.cache_resource
def load_refiner():
    return QueryRefiner()

refiner = load_refiner()

# --- PAGE SETUP ---
st.set_page_config(page_title="AI UK Lawyer", page_icon="⚖️")
st.title("⚖️ AI UK Contract Law Advisor")

# --- 1. CACHED RESOURCE LOADING ---
# We use @st.cache_resource so we only read the PDFs ONCE, not every time you ask a question.
@st.cache_resource
def load_rag_system():
    # A. Setup Embeddings & Model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(
        model=MODEL_NAME, 
        temperature=0.1, 
        num_ctx=8192,
        stop=["<|eot_id|>", "<|start_header_id|>", "📝", "Client Scenario:", "User:", "----------------"]
    )

    # B. Load Documents
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        return None, None, "⚠️ Data folder created. Please add PDFs."

    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.pdf')]
    if not pdf_files:
        return None, None, "❌ No PDFs found. Add files to 'legal_docs'."

    docs = []
    for file in pdf_files:
        loader = PyPDFLoader(os.path.join(DATA_FOLDER, file))
        docs.extend(loader.load())

    # C. Build Database
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, collection_name="uk_law_gui")
    retriever = vectorstore.as_retriever()

    # D. Define Prompt
    template = """<|start_header_id|>system<|end_header_id|>

    You are an expert UK Contract Law consultant. Answer using the IRAC method.

    CRITICAL RULES:
    1. SERVICES (Labor/Installation) -> Use s.49 (Reasonable Care). Liability cannot be excluded for negligence (s.65).
    2. GOODS (Hardware/Physical) -> Use s.9-24. 
       - < 30 Days: Right to Reject (Refund). 
       - > 30 Days: Repair/Replace (s.23) BEFORE Refund.
    3. DIGITAL (Software/Apps) -> Use s.34-44. 
       - Contract Law: "Lifetime" features generally cannot be removed unilaterally (Tesco v USDAW).

    CONTEXT FROM DOCUMENTS:
    {context}
    <|eot_id|><|start_header_id|>user<|end_header_id|>

    {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    prompt = ChatPromptTemplate.from_template(template)

    # E. Build Chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, pdf_files, "✅ System Ready"

# --- 2. INITIALIZATION ---
rag_chain, loaded_files, status_msg = load_rag_system()

# Sidebar Info
with st.sidebar:
    st.header("📂 Case Files")
    if loaded_files:
        for f in loaded_files:
            st.text(f"📄 {f}")
    else:
        st.warning("No PDFs loaded.")
    st.divider()
    st.caption("Status: " + status_msg)

# Chat History Memory
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tts_audio" not in st.session_state:
    st.session_state.tts_audio = {}
if "play_audio_idx" not in st.session_state:
    st.session_state.play_audio_idx = None
if "pending_voice_prompt" not in st.session_state:
    st.session_state.pending_voice_prompt = None


def process_prompt(prompt_text: str, source: str = "user"):
    """Handle chat submission, model response, and speech rendering."""
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    
   

    if not rag_chain:
        st.error("System not initialized. Check PDF folder.")
        return
    

     # Refine the Query
    refined_text = prompt_text # Default to original if refiner fails

    with st.chat_message("assistant"):
        # Create a status container to show the "thinking" process
        with st.status("👨‍💼 Receptionist is reviewing...", expanded=False) as status:
            st.write("Refining query for legal context...")
            try:
                refined_text = refiner.refine(prompt_text)
                st.markdown(f"**Refined Query:** `{refined_text}`")
                status.update(label="Query Refined", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Refinement failed: {e}")
                status.update(label="Refinement Skipped", state="error")

        response_placeholder = st.empty()
        full_response = ""       

        try:
            for chunk in rag_chain.stream(refined_text):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            try:
                audio_bytes = text_to_speech_bytes(full_response)
                st.session_state.tts_audio[len(st.session_state.messages) - 1] = audio_bytes
            except Exception as audio_error:
                st.warning(f"Audio playback unavailable: {audio_error}")

        except Exception as e:
            st.error(f"Error: {e}")


# Display History
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.button(
                "🔊 Read Aloud",
                key=f"read_{idx}",
                help="Play this answer with text-to-speech.",
                on_click=_handle_read_aloud,
                args=(idx,),
            )
            if st.session_state.play_audio_idx == idx:
                audio_bytes = st.session_state.tts_audio.get(idx)
                if not audio_bytes:
                    try:
                        audio_bytes = text_to_speech_bytes(message["content"])
                        st.session_state.tts_audio[idx] = audio_bytes
                    except Exception as audio_error:
                        st.warning(f"Audio playback unavailable: {audio_error}")
                        st.session_state.play_audio_idx = None
                        continue

                _render_audio_player(audio_bytes)
                st.session_state.play_audio_idx = None


st.divider()
st.subheader("🎙️ Ask with Your Voice")
voice_col, hint_col = st.columns([3, 2])
with hint_col:
    st.caption("Use the mic to record a question instead of typing.")

with voice_col:
    audio_data = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=True,
        use_container_width=True,
        format="wav",
        key="ai_lawyer_mic",
    )

if audio_data and audio_data.get("bytes"):
    st.audio(audio_data["bytes"], format=f"audio/{audio_data.get('format', 'wav')}")
    with st.spinner("Transcribing voice input..."):
        transcript = speech_to_text_from_audio_bytes(
            audio_data["bytes"],
            fmt=audio_data.get("format", "wav"),
        )
    if transcript:
        st.success(f"Transcribed question: {transcript}")
        st.session_state.pending_voice_prompt = transcript
    else:
        st.error("Could not understand that recording. Please try again.")

voice_prompt = None
if st.session_state.pending_voice_prompt:
    voice_prompt = st.session_state.pending_voice_prompt
    st.session_state.pending_voice_prompt = None

# --- 3. MAIN CHAT INTERFACE ---
prompt = st.chat_input("Describe the client's legal scenario...")

if voice_prompt:
    process_prompt(voice_prompt, source="voice")
elif prompt:
    process_prompt(prompt)
