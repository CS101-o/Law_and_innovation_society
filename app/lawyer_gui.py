import base64
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit_mic_recorder import mic_recorder

from core.graph_builder import LegalKnowledgeGraph
from core.hybrid_retriever import HybridGraphRetriever
from core.refiner import QueryRefiner

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
GRAPH_FILE = "legal_knowledge_graph.pkl"
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

# --- HELPER CLASSES ---
class DemoChain:
    def __init__(self, pdf_files):
        self._pdf_files = pdf_files or []

    def stream(self, question: str):
        docs_hint = ", ".join(self._pdf_files[:2]) if self._pdf_files else "demo references"
        answer = f"""**IRAC Demo Response**

**Issue:** {question or "General contract law query"}.

**Rule:** Rely on common law principles and statutory protections. Contract law governs the formation, interpretation, and enforcement of agreements.

**Application:** Using cached demo materials ({docs_hint}) to outline reasoning.

**Conclusion:** Recommendation provided. Verify against real documents in Full RAG mode."""
        yield answer

# --- HELPER FUNCTIONS ---
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
    try:
        llm.invoke("ping")
    except Exception:
        pass

def _maybe_persist_vectorstore(vectorstore):
    persist_fn = getattr(vectorstore, "persist", None)
    if callable(persist_fn):
        persist_fn()

def _render_audio_player(audio_bytes: bytes):
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    audio_html = f'<audio autoplay controls style="width: 100%; margin: 1rem 0;"><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

def _render_waiting_tune(placeholder):
    audio_b64 = base64.b64encode(get_waiting_tune_bytes()).decode("utf-8")
    placeholder.markdown(
        f'<audio autoplay loop controls style="width: 100%;"><source src="data:audio/wav;base64,{audio_b64}" type="audio/wav"></audio>',
        unsafe_allow_html=True,
    )

def _handle_read_aloud(idx: int):
    st.session_state.play_audio_idx = idx

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="UK Contract Law Advisor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME STYLING ---
def apply_theme():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

            :root {
                --sidebar-bg: #4a5a6a;
                --main-bg: #5a6d7e;
                --chat-bg: #f5f5f5;
                --text-dark: #2c3e50;
                --text-light: #ffffff;
                --text-muted: #ffffff99;
                --accent: #2c3e50;
                --border: #00000020;
            }

            * {
                font-family: 'Inter', sans-serif;
            }

            /* Hide Streamlit defaults */
            #MainMenu, footer, header {visibility: hidden;}
            .stApp > header {display: none;}
            .stChatInputContainer {display: none !important;}

            /* Hide the default text input wrapper but keep the element functional */
            .stTextInput {
                position: absolute;
                visibility: hidden;
                width: 0;
                height: 0;
                padding: 0;
                margin: 0;
                border: none;
            }

            /* Remove all extra vertical spacing */
            .stVerticalBlock {
                gap: 0 !important;
            }

            [data-testid="stVerticalBlock"] {
                gap: 0 !important;
            }

            /* Hide empty vertical blocks */
            [data-testid="stVerticalBlock"] > div:empty {
                display: none !important;
            }

            /* Main app background */
            .stApp {
                background: var(--main-bg);
            }

            .main .block-container {
                padding: 0 !important;
                margin: 0 !important;
                max-width: 100% !important;
                gap: 0 !important;
            }

            .main {
                padding: 0 !important;
            }

            /* Sidebar */
            section[data-testid="stSidebar"] {
                background: var(--sidebar-bg) !important;
                border-right: none;
            }

            section[data-testid="stSidebar"] > div {
                padding: 2rem 1.5rem;
                background: var(--sidebar-bg);
            }

            .sidebar-title {
                color: var(--text-light);
                font-size: 0.875rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 1rem;
                opacity: 0.8;
            }

            /* Welcome screen */
            .welcome-screen {
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                background: linear-gradient(135deg, #5a6d7e 0%, #4a5a6a 100%);
                padding: 2rem;
                text-align: center;
            }

            .sphere {
                width: 150px;
                height: 150px;
                background: radial-gradient(circle at 35% 35%, #556575 0%, #3a4a5a 40%, #2a3a4a 100%);
                border-radius: 50%;
                box-shadow: 0 20px 60px rgba(0,0,0,0.4), 0 10px 30px rgba(0,0,0,0.3);
                animation: float 6s ease-in-out infinite;
                margin: 0 auto 1.5rem;
            }

            @keyframes float {
                0%, 100% { transform: translateY(0px) rotate(0deg); }
                25% { transform: translateY(-10px) rotate(3deg); }
                50% { transform: translateY(-15px) rotate(0deg); }
                75% { transform: translateY(-10px) rotate(-3deg); }
            }

            .welcome-title {
                font-family: 'Playfair Display', serif;
                font-size: 2.2rem;
                font-weight: 600;
                color: var(--text-light);
                margin-bottom: 1rem;
                line-height: 1.2;
            }

            .welcome-disclaimer {
                max-width: 550px;
                color: var(--text-muted);
                font-size: 0.9rem;
                line-height: 1.6;
                margin-bottom: 1.5rem;
            }

            /* Chat interface */
            .chat-container {
                background: var(--main-bg);
                width: 100%;
                padding: 0 !important;
                margin: 0 !important;
            }

            .chat-messages {
                max-width: 900px;
                margin: 0 auto;
                padding: 0 2rem;
                display: flex;
                flex-direction: column;
                width: 100%;
            }

            .empty-state {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: flex-start;
                padding-top: 40px;
                gap: 1.5rem;
                margin: 0;
                width: 100%;
                padding-bottom: 250px;
            }

            .decorator-shapes {
                display: flex;
                gap: 3rem;
                align-items: center;
                justify-content: center;
                margin-top: 0 !important;
                padding-top: 0 !important;
            }

            .shape-circle {
                width: 80px;
                height: 80px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                opacity: 0.3;
                animation: float-shape 6s ease-in-out infinite;
            }

            .shape-square {
                width: 70px;
                height: 70px;
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                opacity: 0.3;
                animation: float-shape 7s ease-in-out infinite;
                border-radius: 8px;
            }

            .shape-triangle {
                width: 0;
                height: 0;
                border-left: 40px solid transparent;
                border-right: 40px solid transparent;
                border-bottom: 70px solid #4facfe;
                opacity: 0.3;
                animation: float-shape 8s ease-in-out infinite;
            }

            @keyframes float-shape {
                0%, 100% { transform: translateY(0px) rotate(0deg); }
                25% { transform: translateY(-15px) rotate(10deg); }
                50% { transform: translateY(-30px) rotate(20deg); }
                75% { transform: translateY(-15px) rotate(10deg); }
            }

            .empty-text {
                text-align: center;
                color: var(--text-muted);
                font-size: 1.1rem;
            }

            .empty-text strong {
                color: var(--text-light);
                font-size: 1.3rem;
            }

            .message-bubble {
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                line-height: 1.7;
                color: var(--text-dark);
            }

            .message-user {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }

            /* Hide Streamlit column gaps for input area */
            [data-testid="stHorizontalBlock"] {
                gap: 0.75rem !important;
            }

            /* Style text input */
            .stTextInput > div {
                background: white;
                border-radius: 12px;
                padding: 0.5rem 1rem;
                border: 1px solid var(--border);
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }

            .stTextInput input {
                background: transparent !important;
                border: none !important;
            }


            /* Streamlit overrides */
            .stButton > button {
                background: transparent;
                color: var(--text-light);
                border: none;
                padding: 0.75rem 0;
                font-size: 0.95rem;
                text-align: left;
                width: 100%;
                border-bottom: 1px solid rgba(255,255,255,0.1);
                border-radius: 0;
            }

            .stStatus {
                background: white !important;
                border-radius: 12px !important;
                padding: 1rem !important;
            }
        </style>
    """, unsafe_allow_html=True)

apply_theme()

# --- CACHED LOADERS ---
@st.cache_resource
def load_refiner():
    return QueryRefiner()

@st.cache_resource
def load_scope_checker():
    return ChatOllama(model="qwen2.5:0.5b", temperature=0)

@st.cache_resource
def load_knowledge_graph():
    kg = LegalKnowledgeGraph()
    if os.path.exists(GRAPH_FILE):
        kg.load(GRAPH_FILE)
    else:
        kg.build_from_pdfs(DATA_FOLDER)
        kg.save(GRAPH_FILE)
    return kg

@st.cache_resource
def load_rag_system(mode: str):
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    pdf_files = []
    for root, _, files in os.walk(DATA_FOLDER):
        for file in files:
            if file.endswith(".pdf"):
                rel_path = os.path.relpath(os.path.join(root, file), DATA_FOLDER)
                pdf_files.append(rel_path)

    file_catalog = {file: {"categories": ["general"]} for file in pdf_files}

    if mode == "demo":
        return DemoChain(pdf_files), None, file_catalog, f"Demo mode. Found {len(pdf_files)} PDFs."

    if not pdf_files:
        return None, None, None, "No PDFs found in legal_docs or subfolders."

    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        llm = ChatOllama(model=MODEL_NAME, temperature=0.1, num_ctx=8192)
        _warm_llm(llm)
    except Exception as e:
        return None, None, file_catalog, f"Failed to initialize LLM/embeddings: {str(e)}"

    try:
        pdf_state = _collect_pdf_state(pdf_files)
        stored_state = _load_vector_metadata()
        needs_reindex = stored_state != pdf_state

        docs = []
        for rel_file_path in pdf_files:
            abs_path = os.path.join(DATA_FOLDER, rel_file_path)
            loader = PyPDFLoader(abs_path)
            file_docs = loader.load()

            folder_name = os.path.dirname(rel_file_path)
            if folder_name:
                for doc in file_docs:
                    doc.metadata["domain"] = os.path.basename(folder_name)

            categories = classify_pdf_documents(file_docs, source=rel_file_path)
            file_catalog[rel_file_path] = {"categories": categories}
            docs.extend(file_docs)

        if needs_reindex:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            splits = text_splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                collection_name=CHROMA_COLLECTION,
                persist_directory=VECTOR_DB_DIR,
            )
            _maybe_persist_vectorstore(vectorstore)
            _persist_vector_metadata(pdf_state)
            status_message = "Indexed new documents."
        else:
            vectorstore = Chroma(
                persist_directory=VECTOR_DB_DIR,
                embedding_function=embeddings,
                collection_name=CHROMA_COLLECTION,
            )
            status_message = "Loaded index from cache."

        knowledge_graph = load_knowledge_graph()
        return vectorstore, knowledge_graph, file_catalog, status_message
    except Exception as e:
        return None, None, file_catalog, f"Error loading RAG system: {str(e)}"

refiner = load_refiner()
scope_checker = load_scope_checker()

# --- SCOPE CHECKING (IMPROVED) ---
def is_out_of_scope(text: str) -> bool:
    """Check if query is about UK Contract Law. Very lenient to avoid false positives."""
    contract_keywords = [
        'contract', 'agreement', 'promise', 'rent', 'lease', 'tenant', 'landlord',
        'breach', 'terms', 'clause', 'offer', 'acceptance', 'misrepresentation',
        'fraud', 'mistake', 'estoppel', 'liability', 'damages', 'remedies',
        'seller', 'buyer', 'goods', 'service', 'money', 'payment', 'lied', 'false',
        'broken', 'broken promise', 'legal', 'law', 'lawsuit', 'court'
    ]

    text_lower = text.lower()
    # If contains ANY contract keyword, it's in scope
    if any(keyword in text_lower for keyword in contract_keywords):
        return False

    # Otherwise check with LLM as fallback
    prompt = f"""Is this about UK Contract Law? Answer SCOPE or OUTSCOPE.
Query: "{text}" """
    try:
        response = scope_checker.invoke(prompt)
        result = response.content.strip().upper()
        return "OUTSCOPE" in result and "SCOPE" not in result
    except Exception:
        return False  # If error, assume in scope

def get_welcome_message() -> str:
    return """Welcome to the AI UK Contract Law Advisor.

I specialize in UK Contract Law, including:
- Promissory Estoppel
- Misrepresentation
- Contractual Terms and Breaches
- Mutual Mistake
- Offer and Acceptance

Ask a question about a legal scenario to begin."""

# --- SESSION STATE ---
if "view" not in st.session_state:
    st.session_state.view = "chat"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tts_audio" not in st.session_state:
    st.session_state.tts_audio = {}
if "play_audio_idx" not in st.session_state:
    st.session_state.play_audio_idx = None
if "pending_voice_prompt" not in st.session_state:
    st.session_state.pending_voice_prompt = None
if "runtime_mode_label" not in st.session_state:
    st.session_state.runtime_mode_label = DEFAULT_RUNTIME_LABEL
if "last_processed_input" not in st.session_state:
    st.session_state.last_processed_input = None

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="sidebar-title">Settings</div>', unsafe_allow_html=True)
    selected_label = st.selectbox(
        "Mode",
        [label for label, _ in RUNTIME_MODE_OPTIONS],
        key="runtime_mode_label",
        label_visibility="collapsed"
    )

runtime_mode = MODE_LOOKUP[selected_label]
vectorstore, knowledge_graph, loaded_files, status_msg = load_rag_system(runtime_mode)

with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">System Status</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color: white; font-size: 0.85rem; padding: 0.5rem 0;">📚 {len(loaded_files) if loaded_files else 0} documents</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color: white; font-size: 0.85rem; padding: 0.5rem 0;">✅ {status_msg}</div>', unsafe_allow_html=True)

    if knowledge_graph:
        st.markdown(f'<div style="color: white; font-size: 0.85rem; padding: 0.5rem 0;">🔗 {knowledge_graph.graph.number_of_nodes()} entities</div>', unsafe_allow_html=True)
        if st.button("🔄 Rebuild Graph"):
            if os.path.exists(GRAPH_FILE):
                os.remove(GRAPH_FILE)
            st.cache_resource.clear()
            st.rerun()

# --- MAIN CONTENT ---
st.markdown('<div class="chat-container"><div class="chat-messages">', unsafe_allow_html=True)

# Show empty state if no messages
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="decorator-shapes">
            <div class="shape-circle"></div>
            <div class="shape-square"></div>
            <div class="shape-triangle"></div>
        </div>
        <div class="empty-text">
            <strong>Welcome to Contract Law Advisor</strong><br>
            Ask a question about UK contract law to get started
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Show messages
    for idx, message in enumerate(st.session_state.messages):
        bubble_class = "message-bubble message-user" if message["role"] == "user" else "message-bubble"
        st.markdown(f'<div class="{bubble_class}">{message["content"]}</div>', unsafe_allow_html=True)

        if message["role"] == "assistant":
            col1, col2 = st.columns([1, 10])
            with col1:
                pass
            with col2:
                if st.button("🔊 Read Aloud", key=f"read_{idx}"):
                    _handle_read_aloud(idx)
                    st.rerun()

            if st.session_state.play_audio_idx == idx:
                audio_bytes = st.session_state.tts_audio.get(idx) or text_to_speech_bytes(message["content"])
                st.session_state.tts_audio[idx] = audio_bytes
                _render_audio_player(audio_bytes)
                st.session_state.play_audio_idx = None

st.markdown('</div></div>', unsafe_allow_html=True)

# --- INPUT AREA (ALWAYS VISIBLE) ---
col_input, col_mic = st.columns([5, 1])

with col_input:
    user_input = st.text_input(
        "Message",
        placeholder="Ask about contract law...",
        label_visibility="collapsed",
        key="chat_input"
    )

with col_mic:
    audio_data = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", just_once=True, key="mic")

if audio_data and audio_data.get("bytes"):
    with st.spinner("Transcribing voice input..."):
        # mic_recorder returns webm format by default
        audio_format = audio_data.get("format", "webm")
        transcript = speech_to_text_from_audio_bytes(
            audio_data["bytes"],
            fmt=audio_format,
        )
    if transcript:
        st.success(f"Transcribed question: {transcript}")
        st.session_state.pending_voice_prompt = transcript
    else:
        st.error("Could not understand that recording. Please try again.")

# --- PROCESS INPUT ---
def process_prompt(prompt_text: str):
    if st.session_state.view != "chat":
        st.session_state.view = "chat"

    # Mark this input as processed
    st.session_state.last_processed_input = prompt_text
    st.session_state.messages.append({"role": "user", "content": prompt_text})

    if runtime_mode == "demo":
        full_response = ""
        for chunk in vectorstore.stream(prompt_text):
            full_response += chunk
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()
        return

    if not vectorstore or not knowledge_graph:
        st.error(f"System not ready: {status_msg}")
        return

    if is_out_of_scope(prompt_text):
        msg = get_welcome_message()
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.rerun()
        return

    # Show analysis
    with st.status("🔍 Analyzing Legal Context...", expanded=True) as status:
        st.write("Step 1: Classifying legal domain...")
        domain_tags, refined_text = refiner.refine(prompt_text)

        st.write("Step 2: Identifying content type...")
        content_categories = guess_question_categories(prompt_text)
        content_display = format_category_list(content_categories)

        if content_display:
            st.markdown(f"**Content Type:** `{content_display}`")
        st.markdown(f"**Legal Domain:** {', '.join([f'`{d}`' for d in domain_tags])}")
        st.markdown(f"**Refined Query:** `{refined_text}`")

        st.write("Step 3: Retrieving with GraphRAG...")
        hybrid_retriever = HybridGraphRetriever(vectorstore, knowledge_graph)
        domain_str = domain_tags[0] if domain_tags else "General"
        retrieved_docs = hybrid_retriever.get_relevant_documents(refined_text, domain=domain_str, k=4)

        st.write(f"Retrieved {len(retrieved_docs)} documents")

        graph_context = ""
        for doc in retrieved_docs:
            if "graph_context" in doc.metadata:
                graph_context = doc.metadata["graph_context"]
                break

        if graph_context:
            with st.expander("📊 Knowledge Graph Context"):
                st.markdown(graph_context)

        status.update(label="✅ Analysis Complete", state="complete", expanded=False)

    # Generate response with waiting music
    waiting_audio_placeholder = st.empty()
    _render_waiting_tune(waiting_audio_placeholder)

    llm_engine = ChatOllama(model=MODEL_NAME, temperature=0.1)
    domains_str = ", ".join(domain_tags)
    prompt_template = ChatPromptTemplate.from_template(
        f"""You are an expert UK Contract Law consultant specializing in {domains_str}.

Use IRAC method (Issue, Rule, Analysis, Conclusion).
Cite cases with [Year] citations.

RETRIEVED CONTEXT: {{context}}

KNOWLEDGE GRAPH: {{graph}}

User Query: {{question}}"""
    )

    rag_chain = (
        {
            "context": lambda _: "\n\n".join([d.page_content for d in retrieved_docs]),
            "graph": lambda _: graph_context if graph_context else "No graph relationships found",
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm_engine
        | StrOutputParser()
    )

    full_response = ""
    try:
        for chunk in rag_chain.stream(refined_text):
            full_response += chunk

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Generate TTS
        try:
            audio_bytes = text_to_speech_bytes(full_response)
            st.session_state.tts_audio[len(st.session_state.messages) - 1] = audio_bytes
        except Exception:
            pass

    except Exception as e:
        st.error(f"Response generation error: {e}")
    finally:
        waiting_audio_placeholder.empty()

    st.rerun()

# --- PROCESS PENDING INPUTS ---
if st.session_state.pending_voice_prompt:
    prompt_to_process = st.session_state.pending_voice_prompt
    st.session_state.pending_voice_prompt = None
    process_prompt(prompt_to_process)
elif user_input and user_input != st.session_state.last_processed_input:
    # Only process if input is new (different from last processed)
    process_prompt(user_input)
