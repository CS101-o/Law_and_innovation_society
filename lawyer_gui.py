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

from graph_builder import LegalKnowledgeGraph
from hybrid_retriever import HybridGraphRetriever
from refiner import QueryRefiner

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

class DemoChain:
    def __init__(self, pdf_files):
        self._pdf_files = pdf_files or []

    def stream(self, question: str):
        docs_hint = ", ".join(self._pdf_files[:2]) if self._pdf_files else "demo references"
        answer = f"""IRAC Demo Response

Issue: {question or "General consumer law query"}.
Rule: Rely on CRA 2015 (Goods s.9-24, Services s.49, Digital s.34-44). Note negligence cannot be excluded (s.65).
Application: Using cached demo materials ({docs_hint}) to outline reasoning.
Conclusion: Recommendation provided. Verify against real documents in Full RAG mode."""
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
    try:
        llm.invoke("ping")
    except Exception:
        pass

def _maybe_persist_vectorstore(vectorstore):
    persist_fn = getattr(vectorstore, "persist", None)
    if callable(persist_fn):
        persist_fn()

def _apply_custom_theme():
    st.markdown(
        """
        <style>
            :root {
                --app-bg: #081027;
                --panel-bg: #fdfefe;
                --panel-border: rgba(15, 23, 42, 0.08);
                --accent: #7a6afc;
                --accent-dark: #4c3dbf;
                --text-primary: #0f172a;
            }
            .stApp {
                background: var(--app-bg);
            }
            .main .block-container {
                padding: 2rem;
                max-width: 950px;
                background: var(--panel-bg);
                border-radius: 20px;
            }
            .assistant-message {
                background: #f5f7ff;
                border-radius: 12px;
                padding: 1rem;
                border-left: 4px solid var(--accent);
            }
            .user-message {
                background: #fff7ed;
                border-radius: 12px;
                padding: 1rem;
                border-left: 4px solid #f8923c;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _render_audio_player(audio_bytes: bytes):
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    audio_html = f'<audio autoplay controls><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

def _render_waiting_tune(placeholder):
    audio_b64 = base64.b64encode(get_waiting_tune_bytes()).decode("utf-8")
    placeholder.markdown(
        f'<audio autoplay loop controls style="width: 100%;"><source src="data:audio/wav;base64,{audio_b64}" type="audio/wav"></audio>',
        unsafe_allow_html=True,
    )

def _handle_read_aloud(idx: int):
    st.session_state.play_audio_idx = idx

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
        st.info("Building knowledge graph. Please wait.")
        kg.build_from_pdfs(DATA_FOLDER)
        kg.save(GRAPH_FILE)
    return kg

refiner = load_refiner()
scope_checker = load_scope_checker()

def is_out_of_scope(text: str) -> bool:
    prompt = f"""Identify if the following query is related to UK Contract Law.
    Answer only SCOPE or OUTSCOPE.
    Query: "{text}" """
    try:
        response = scope_checker.invoke(prompt)
        result = response.content.strip().upper()
        return "OUTSCOPE" in result or "OUT" in result
    except Exception:
        return False

def get_welcome_message() -> str:
    return """Welcome to the AI UK Contract Law Advisor.

I specialize in UK Contract Law, including:
- Promissory Estoppel
- Misrepresentation
- Contractual Terms and Breaches
- Mutual Mistake
- Offer and Acceptance

Ask a question about a legal scenario to begin."""

st.set_page_config(page_title="AI UK Lawyer")
_apply_custom_theme()
st.title("AI UK Contract Law Advisor")

if "runtime_mode_label" not in st.session_state:
    st.session_state.runtime_mode_label = DEFAULT_RUNTIME_LABEL

with st.sidebar:
    st.subheader("Settings")
    selected_label = st.selectbox(
        "Mode",
        [label for label, _ in RUNTIME_MODE_OPTIONS],
        key="runtime_mode_label",
    )
runtime_mode = MODE_LOOKUP[selected_label]

@st.cache_resource
def load_rag_system(mode: str):
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]
    file_catalog = {file: {"categories": ["general"]} for file in pdf_files}

    if mode == "demo":
        return DemoChain(pdf_files), None, file_catalog, "Running in demo mode."

    if not pdf_files:
        return None, None, None, "No PDFs found in legal_docs."

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model=MODEL_NAME, temperature=0.1, num_ctx=8192)
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

vectorstore, knowledge_graph, loaded_files, status_msg = load_rag_system(runtime_mode)

with st.sidebar:
    st.header("Case Files")
    if loaded_files:
        for filename, info in loaded_files.items():
            display = format_category_list(info.get("categories", []))
            st.text(f"{filename} ({display})" if display else filename)
    
    st.divider()
    st.caption(status_msg)
    
    if knowledge_graph:
        st.subheader("Graph Metrics")
        st.metric("Entities", knowledge_graph.graph.number_of_nodes())
        st.metric("Relationships", knowledge_graph.graph.number_of_edges())
        if st.button("Rebuild Graph"):
            if os.path.exists(GRAPH_FILE):
                os.remove(GRAPH_FILE)
            st.cache_resource.clear()
            st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "tts_audio" not in st.session_state:
    st.session_state.tts_audio = {}
if "play_audio_idx" not in st.session_state:
    st.session_state.play_audio_idx = None
if "pending_voice_prompt" not in st.session_state:
    st.session_state.pending_voice_prompt = None

def process_prompt(prompt_text: str):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    
    with st.chat_message("user"):
        st.markdown(f"<div class='user-message'>{prompt_text}</div>", unsafe_allow_html=True)

    if runtime_mode == "demo":
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            for chunk in vectorstore.stream(prompt_text):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(f"<div class='assistant-message'>{full_response}</div>", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        return

    if not vectorstore or not knowledge_graph:
        st.error("System not ready.")
        return
    
    if is_out_of_scope(prompt_text):
        with st.chat_message("assistant"):
            msg = get_welcome_message()
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
        return
    
    refined_text = prompt_text 
    domain_tags = ["General"]
    
    with st.chat_message("assistant"):
        with st.status("Analyzing Legal Context...", expanded=False) as status:
            try:
                domain_tags, refined_text = refiner.refine(prompt_text)
                content_categories = guess_question_categories(prompt_text)
                status.update(label="Context identified", state="complete")
            except Exception:
                status.update(label="Analysis Error", state="error")
        
        hybrid_retriever = HybridGraphRetriever(vectorstore, knowledge_graph)
        domain_str = domain_tags[0] if domain_tags else "General"
        retrieved_docs = hybrid_retriever.get_relevant_documents(refined_text, domain=domain_str, k=4)
        
        graph_context = ""
        if retrieved_docs and "graph_context" in retrieved_docs[0].metadata:
            graph_context = retrieved_docs[0].metadata["graph_context"]
        
        llm_engine = ChatOllama(model=MODEL_NAME, temperature=0.1)
        prompt_template = ChatPromptTemplate.from_template(
            "System: Expert UK Contract Law consultant. Use IRAC. Context: {context} Graph: {graph} User: {question}"
        )
        
        rag_chain = (
            {"context": lambda x: "\n\n".join([d.page_content for d in retrieved_docs]), 
             "graph": lambda x: graph_context,
             "question": RunnablePassthrough()}
            | prompt_template | llm_engine | StrOutputParser()
        )
        
        response_placeholder = st.empty()
        waiting_audio = st.empty()
        _render_waiting_tune(waiting_audio)
        
        full_response = ""
        try:
            for chunk in rag_chain.stream(refined_text):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(f"<div class='assistant-message'>{full_response}</div>", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            try:
                audio_bytes = text_to_speech_bytes(full_response)
                st.session_state.tts_audio[len(st.session_state.messages) - 1] = audio_bytes
            except Exception:
                pass
                
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            waiting_audio.empty()

    st.rerun()

def render_history():
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            css_class = "assistant-message" if message["role"] == "assistant" else "user-message"
            st.markdown(f"<div class='{css_class}'>{message['content']}</div>", unsafe_allow_html=True)

            if message["role"] == "assistant":
                if st.button("Read Aloud", key=f"read_{idx}"):
                    _handle_read_aloud(idx)
                    st.rerun()
                
                if st.session_state.play_audio_idx == idx:
                    audio_bytes = st.session_state.tts_audio.get(idx) or text_to_speech_bytes(message["content"])
                    st.session_state.tts_audio[idx] = audio_bytes
                    _render_audio_player(audio_bytes)
                    st.session_state.play_audio_idx = None

render_history()

st.divider()
st.subheader("Voice Input")
audio_data = mic_recorder(start_prompt="Record", stop_prompt="Transcribe", just_once=True, key="mic")

if audio_data and audio_data.get("bytes"):
    with st.spinner("Transcribing..."):
        transcript = speech_to_text_from_audio_bytes(audio_data["bytes"], fmt=audio_data.get("format", "wav"))
        if transcript:
            st.session_state.pending_voice_prompt = transcript

if st.session_state.pending_voice_prompt:
    prompt_to_process = st.session_state.pending_voice_prompt
    st.session_state.pending_voice_prompt = None
    process_prompt(prompt_to_process)

user_input = st.chat_input("Ask a legal question...")
if user_input:
    process_prompt(user_input)