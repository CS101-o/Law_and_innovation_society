import base64
import json
import os

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit_mic_recorder import mic_recorder

from src.pdf_classifier import (
    classify_pdf_documents,
    format_category_list,
    guess_question_categories,
)
from src.tts import speech_to_text_from_audio_bytes, text_to_speech_bytes
from src.waiting_music import get_waiting_tune_bytes

# --- CONFIGURATION ---
MODEL_NAME = "uk-lawyer"
DATA_FOLDER = "./legal_docs"
VECTOR_DB_DIR = "./chroma_db"
VECTOR_METADATA_PATH = os.path.join(VECTOR_DB_DIR, "index_metadata.json")
CHROMA_COLLECTION = "uk_law_gui"

CHUNK_SIZE = 1100
CHUNK_OVERLAP = 120

RUNTIME_MODE_OPTIONS = [
    ("Full RAG (Ollama)", "full"),
    ("Demo (no Ollama)", "demo"),
]
MODE_LOOKUP = {label: value for label, value in RUNTIME_MODE_OPTIONS}
VALUE_TO_LABEL = {value: label for label, value in RUNTIME_MODE_OPTIONS}
DEFAULT_RUNTIME_MODE = os.environ.get("AI_LAWYER_MODE", "full").lower()
DEFAULT_RUNTIME_LABEL = VALUE_TO_LABEL.get(
    DEFAULT_RUNTIME_MODE, RUNTIME_MODE_OPTIONS[0][0]
)


class DemoChain:
    """Ultra-light responder so the UI can be tested without heavy local models."""

    def __init__(self, pdf_files):
        self._pdf_files = pdf_files or []

    def stream(self, question: str):
        docs_hint = ", ".join(self._pdf_files[:2]) if self._pdf_files else "demo references"
        answer = f"""IRAC Demo Response

Issue: {question or "General consumer law query"}.
Rule: Rely on CRA 2015 (Goods s.9-24, Services s.49, Digital s.34-44). Note negligence cannot be excluded (s.65).
Application: Without the full RAG system we use cached demo materials ({docs_hint}) to outline the reasoning only.
Conclusion: Provide a concise recommendation and advise the user to verify against the real PDFs when Full RAG mode is enabled."""
        yield answer


def _load_vector_metadata():
    try:
        with open(VECTOR_METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _persist_vector_metadata(metadata: dict):
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    with open(VECTOR_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _collect_pdf_state(pdf_files):
    state = {}
    for file in pdf_files:
        path = os.path.join(DATA_FOLDER, file)
        try:
            state[file] = os.path.getmtime(path)
        except FileNotFoundError:
            continue
    return state


def _warm_llm(llm: ChatOllama):
    """Send a light ping so Ollama keeps the model loaded."""
    try:
        llm.invoke("ping")
    except Exception:
        # If warmup fails we simply proceed—the main request will try again.
        pass


def _maybe_persist_vectorstore(vectorstore):
    """Persist the vectorstore if the backend exposes a persist method."""
    persist_fn = getattr(vectorstore, "persist", None)
    if callable(persist_fn):
        persist_fn()


def _apply_custom_theme():
    """Inject a lightweight theme to make the interface feel more polished."""
    st.markdown(
        """
        <style>
            :root {
                --app-bg: #081027;
                --panel-bg: #fdfefe;
                --panel-border: rgba(15, 23, 42, 0.08);
                --accent: #7a6afc;
                --accent-dark: #4c3dbf;
                --accent-soft: #eef1ff;
                --text-primary: #0f172a;
                --text-muted: #475569;
                --sunset: linear-gradient(135deg, #f9d976, #f39f86);
                --sea: linear-gradient(135deg, #7dd3fc, #818cf8);
            }
            body {
                color: var(--text-primary);
            }
            .stApp {
                background: radial-gradient(circle at 15% 10%, rgba(125, 211, 252, 0.25), transparent 45%),
                            radial-gradient(circle at 80% 0%, rgba(248, 113, 113, 0.2), transparent 40%),
                            var(--app-bg);
            }
            .main .block-container {
                padding: 2rem 2.5rem 4rem;
                max-width: 950px;
                margin-top: 1.25rem;
                background: var(--panel-bg);
                border-radius: 30px;
                box-shadow: 0 25px 70px rgba(6, 12, 24, 0.6);
                border: 1px solid var(--panel-border);
            }
            .stSidebar {
                background-color: #101522 !important;
            }
            .stSidebar p, .stSidebar span, .stSidebar .stCaption {
                color: #e2e8f0 !important;
            }
            [data-testid="stSidebar"] {
                border-right: 1px solid rgba(255, 255, 255, 0.08);
            }
            .stSidebar .stMarkdown {
                color: #e2e8f0;
            }
            [data-testid="stChatMessage"] {
                border-radius: 22px;
                padding: 1rem 1.2rem;
                margin-bottom: 1rem;
                border: 1px solid rgba(148, 163, 184, 0.4);
                background: white;
                box-shadow: 0px 12px 24px rgba(15, 23, 42, 0.12);
            }
            .assistant-message {
                background: #f5f7ff;
                border-radius: 16px;
                padding: 0.4rem 0.2rem;
                border-left: 4px solid rgba(99, 102, 241, 0.45);
                color: var(--text-primary);
            }
            .user-message {
                background: #fff7ed;
                border-radius: 16px;
                padding: 0.4rem 0.2rem;
                border-left: 4px solid rgba(248, 146, 60, 0.55);
                color: var(--text-primary);
            }
            .stButton button {
                border-radius: 999px;
                padding: 0.45rem 1.5rem;
                font-weight: 600;
            }
            button[kind="primary"] {
                background: linear-gradient(135deg, var(--accent), #a855f7) !important;
                border: none;
                color: white !important;
                box-shadow: 0 15px 30px rgba(105, 99, 255, 0.35);
            }
            button[kind="secondary"] {
                background: #101522;
                border: 1px solid rgba(148, 163, 184, 0.4);
                color: #f8fafc;
            }
            .voice-card {
                border-radius: 18px;
                padding: 1.4rem 1.6rem;
                background: #f9fafb;
                border: 1px solid rgba(15, 23, 42, 0.08);
                color: var(--text-muted);
                margin-bottom: 0.75rem;
            }
            .voice-card strong {
                color: var(--accent-dark);
                font-size: 1.05rem;
            }
            .voice-helper {
                font-size: 1rem;
                color: #d7ddeb;
                margin-top: 0.4rem;
                display: inline-block;
                padding-left: 0.55rem;
            }
            .voice-helper strong {
                font-size: 1.02rem;
                color: #f8fafc;
            }
            div[data-testid="stChatInput"] {
                background: rgba(15, 23, 42, 0.9);
                border-radius: 16px;
                padding: 0.2rem 1rem 0.8rem;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            div[data-testid="stChatInput"] textarea {
                background: transparent;
                color: #e2e8f0;
            }
            div[data-testid="stChatInput"] label {
                color: #94a3b8;
            }
            audio {
                width: 100%;
                margin-top: 0.75rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_audio_player(audio_bytes: bytes):
    """Render an autoplaying audio player for cached speech."""
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    audio_html = f"""
        <audio autoplay controls>
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


def _render_waiting_tune(placeholder):
    """Display a looping audio element used while the model is thinking."""
    audio_b64 = base64.b64encode(get_waiting_tune_bytes()).decode("utf-8")
    placeholder.markdown(
        f"""
        <audio autoplay loop controls style="width: 100%;">
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
        </audio>
        """,
        unsafe_allow_html=True,
    )


def _handle_read_aloud(idx: int):
    st.session_state.play_audio_idx = idx


# --- PAGE SETUP ---
st.set_page_config(page_title="AI UK Lawyer", page_icon="⚖️")
_apply_custom_theme()
st.title("⚖️ AI UK Contract Law Advisor")
st.caption("Reliable IRAC-style answers with instant voice playback and speech input.")

if "runtime_mode_label" not in st.session_state:
    st.session_state.runtime_mode_label = DEFAULT_RUNTIME_LABEL

with st.sidebar:
    st.subheader("⚙️ Runtime")
    selected_label = st.selectbox(
        "Choose mode",
        [label for label, _ in RUNTIME_MODE_OPTIONS],
        key="runtime_mode_label",
        help="Demo mode skips Ollama and embeddings so you can test UI features quietly.",
    )
runtime_mode = MODE_LOOKUP[selected_label]


# --- 1. RESOURCE LOADING ---
def load_rag_system(mode: str):
    created_data_folder = False
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        created_data_folder = True

    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]
    file_catalog = {file: {"categories": ["general"]} for file in pdf_files}

    if mode == "demo":
        status_message = "🧪 Demo mode: lightweight mock answers ({} PDF(s) detected).".format(
            len(pdf_files)
        )
        if created_data_folder and not pdf_files:
            status_message += " Add PDFs to exit demo later."
        return DemoChain(pdf_files), file_catalog, status_message

    if not pdf_files:
        if created_data_folder:
            return None, None, "⚠️ Data folder created. Please add PDFs."
        return None, None, "❌ No PDFs found. Add files to 'legal_docs'."

    # A. Setup Embeddings & Model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.1,
        num_ctx=8192,
        stop=["<|eot_id|>", "<|start_header_id|>", "📝", "Client Scenario:", "User:", "----------------"],
    )
    _warm_llm(llm)

    pdf_state = _collect_pdf_state(pdf_files)
    stored_state = _load_vector_metadata()

    needs_reindex = stored_state != pdf_state

    docs = []
    for file in pdf_files:
        loader = PyPDFLoader(os.path.join(DATA_FOLDER, file))
        file_docs = loader.load()
        categories = classify_pdf_documents(file_docs, source=file)
        file_catalog[file] = {"categories": categories}
        docs.extend(file_docs)

    if needs_reindex:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=CHROMA_COLLECTION,
            persist_directory=VECTOR_DB_DIR,
        )
        _maybe_persist_vectorstore(vectorstore)
        _persist_vector_metadata(pdf_state)
        status_message = "✅ Indexed PDFs, cached embeddings, and labeled document categories."
    else:
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION,
        )
        status_message = "✅ Loaded cached embeddings with stored document categories."

    def _fetch_relevant_docs(question: str):
        categories = guess_question_categories(question or "")
        filter_kwargs = {}
        if categories:
            filter_kwargs["category"] = {"$in": categories}
        docs = vectorstore.similarity_search(
            question,
            k=4,
            filter=filter_kwargs or None,
        )
        if not docs and categories:
            docs = vectorstore.similarity_search(question, k=4)
        return docs

    context_retriever = RunnableLambda(_fetch_relevant_docs)

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
        {"context": context_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, file_catalog, status_message

# --- 2. INITIALIZATION ---
rag_chain, loaded_files, status_msg = load_rag_system(runtime_mode)

# Sidebar Info
with st.sidebar:
    st.header("📂 Case Files")
    if loaded_files:
        for filename, info in loaded_files.items():
            categories_display = format_category_list(info.get("categories", []))
            if categories_display:
                st.text(f"📄 {filename} [{categories_display}]")
            else:
                st.text(f"📄 {filename}")
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
        st.markdown(f"<div class='user-message'>{prompt_text}</div>", unsafe_allow_html=True)

    if not rag_chain:
        st.error("System not initialized. Check PDF folder.")
        return

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        waiting_audio_placeholder = st.empty()
        waiting_note_placeholder = st.empty()
        _render_waiting_tune(waiting_audio_placeholder)
        waiting_note_placeholder.caption("🎵 Holding music while we craft your answer...")
        full_response = ""

        try:
            for chunk in rag_chain.stream(prompt_text):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(
                f"<div class='assistant-message'>{full_response}</div>",
                unsafe_allow_html=True,
            )
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            try:
                with st.spinner("Generating read-aloud audio..."):
                    audio_bytes = text_to_speech_bytes(full_response)
                st.session_state.tts_audio[len(st.session_state.messages) - 1] = audio_bytes
            except Exception as audio_error:
                st.warning(f"Audio playback unavailable: {audio_error}")

        except Exception as e:
            st.error(f"Error: {e}")
            return
        finally:
            waiting_audio_placeholder.empty()
            waiting_note_placeholder.empty()

    st.rerun()


def render_history():
    """Render chat history with Read Aloud controls."""
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(
                    f"<div class='assistant-message'>{message['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='user-message'>{message['content']}</div>",
                    unsafe_allow_html=True,
                )

            if message["role"] == "assistant":
                st.button(
                    "🔊 Read Aloud",
                    key=f"read_{idx}",
                    help="Play this answer with text-to-speech.",
                    on_click=_handle_read_aloud,
                    args=(idx,),
                    type="primary",
                    use_container_width=False,
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


with st.container():
    render_history()

st.divider()
st.subheader("🎙️ Ask with Your Voice")
with st.container():
    st.markdown(
        "<div class='voice-card'><strong>Hands-free mode</strong><br/>Capture any query in seconds and we'll transcribe it into the chat.</div>",
        unsafe_allow_html=True,
    )
    voice_col, hint_col = st.columns([3, 2], gap="large")
    with voice_col:
        audio_data = mic_recorder(
            start_prompt="🎙️ Start recording",
            stop_prompt="✅ Stop & transcribe",
            just_once=True,
            use_container_width=True,
            format="wav",
            key="ai_lawyer_mic",
        )
    with hint_col:
        st.markdown(
            "<span class='voice-helper'><strong>How it works:</strong> Hold the mic button, speak naturally, then release to send the transcript into the chat.</span>",
            unsafe_allow_html=True,
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
