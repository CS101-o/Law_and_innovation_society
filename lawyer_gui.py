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

#GraphRAG imports
from graph_builder import LegalKnowledgeGraph
from hybrid_retriever import HybridGraphRetriever

# --- CONFIGURATION ---
MODEL_NAME = "uk-lawyer"
DATA_FOLDER = "./legal_docs"
GRAPH_FILE = "legal_knowledge_graph.pkl"

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

# NEW: Load knowledge graph
@st.cache_resource
def load_knowledge_graph():
    """Load or build the knowledge graph."""
    kg = LegalKnowledgeGraph()
    
    # Try to load existing graph
    if os.path.exists(GRAPH_FILE):
        kg.load(GRAPH_FILE)
    else:
        # Build from PDFs
        st.info("🏗️ Building knowledge graph for the first time... This may take a few minutes.")
        kg.build_from_pdfs(DATA_FOLDER)
        kg.save(GRAPH_FILE)
    
    return kg

refiner = load_refiner()

# --- PAGE SETUP ---
st.set_page_config(page_title="AI UK Lawyer", page_icon="⚖️")
st.title("⚖️ AI UK Contract Law Advisor")
st.caption("🔗 Enhanced with GraphRAG")

# --- 1. CACHED RESOURCE LOADING ---
@st.cache_resource
def load_rag_system():
    """Load embeddings, model, and build vector database from PDFs."""
    # A. Setup Embeddings & Model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(
        model=MODEL_NAME, 
        temperature=0.1, 
        num_ctx=8192,
        stop=["<|eot_id|>", "<|start_header_id|>", "📝", "Client Scenario:", "User:", "----------------"]
    )
    
    # B. Load Documents from Subfolders (The Routing Layer)
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        return None, None, None, "⚠️ Data folder created."
    
    docs = []
    found_files = []
    
    # Walk through all subfolders in legal_docs
    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                # Extract the folder name to use as the Domain Tag
                domain_name = os.path.basename(root)
                if domain_name == "legal_docs": 
                    domain_name = "General"
                
                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()
                
                # Tag every page with its domain
                for doc in loaded_docs:
                    doc.metadata["domain"] = domain_name
                
                docs.extend(loaded_docs)
                found_files.append(f"[{domain_name}] {file}")
    
    if not docs:
        return None, None, None, "❌ No PDFs found."
    
    # C. Build Database
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        collection_name="uk_law_routed"
    )
    
    # NEW: Load knowledge graph
    knowledge_graph = load_knowledge_graph()
    
    return vectorstore, knowledge_graph, found_files, "✅ System Ready"

# --- 2. INITIALIZATION ---
vectorstore, knowledge_graph, loaded_files, status_msg = load_rag_system()

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
    
    # NEW: Knowledge Graph Stats
    if knowledge_graph:
        st.divider()
        st.subheader("🔗 Knowledge Graph")
        st.metric("Entities", knowledge_graph.graph.number_of_nodes())
        st.metric("Relationships", knowledge_graph.graph.number_of_edges())
        
        # Option to rebuild graph
        if st.button("🔄 Rebuild Graph"):
            if os.path.exists(GRAPH_FILE):
                os.remove(GRAPH_FILE)
            st.cache_resource.clear()
            st.rerun()

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
    
    if not vectorstore or not knowledge_graph:
        st.error("System not initialized. Check PDF folder.")
        return
    
    # Refine and Classify the Query
    refined_text = prompt_text 
    domain_tags = ["General"]  # Default fallback
    
    with st.chat_message("assistant"):
        with st.status("👨‍💼 Receptionist is reviewing...", expanded=False) as status:
            st.write("Analyzing legal domains and refining context...")
            
            try:
                domain_tags, refined_text = refiner.refine(prompt_text)
                tags_display = ", ".join([f"`{d}`" for d in domain_tags])
                st.markdown(f"**Classified Domains:** {tags_display}")
                st.markdown(f"**Refined Query:** `{refined_text}`")
                status.update(
                    label=f"Routed to: {', '.join(domain_tags)}", 
                    state="complete", 
                    expanded=False
                )
            except ValueError:
                st.warning("Refiner returned a single value. Updating to legacy mode.")
                refined_text = refiner.refine(prompt_text)
                domain_tags = ["General"]
                status.update(label="Routed to: General", state="complete")
            except Exception as e:
                st.error(f"Refinement failed: {e}")
                status.update(label="Refinement Error", state="error")
        
        # --- HYBRID GRAPHRAG RETRIEVAL ---
        # NEW: Create hybrid retriever with GraphRAG
        hybrid_retriever = HybridGraphRetriever(vectorstore, knowledge_graph)
        
        # Get documents with graph expansion
        domain_str = domain_tags[0] if domain_tags else "General"
        retrieved_docs = hybrid_retriever.get_relevant_documents(
            refined_text, 
            domain=domain_str,
            k=4
        )
        
        # NEW: Check for graph context
        graph_context = ""
        if retrieved_docs and "graph_context" in retrieved_docs[0].metadata:
            graph_context = retrieved_docs[0].metadata["graph_context"]
            with st.expander("🔗 Knowledge Graph Context"):
                st.markdown(graph_context)
        
        # --- RE-BUILD CHAIN WITH DYNAMIC CONTEXT ---
        llm = ChatOllama(model=MODEL_NAME, temperature=0.1)
        domains_str = ", ".join(domain_tags)
        
        # NEW: Enhanced prompt with graph context
        prompt_template = ChatPromptTemplate.from_template(
            f"""<|start_header_id|>system<|end_header_id|>
You are an expert UK Contract Law consultant specializing in {domains_str}.

**INSTRUCTIONS:**
1. Use IRAC method (Issue, Rule, Analysis, Conclusion)
2. Cite specific cases with [Year] citations
3. If multiple doctrines apply, clearly distinguish PRIMARY vs ALTERNATIVE reasoning
4. Consider the knowledge graph relationships between cases

RETRIEVED CONTEXT: {{context}}

KNOWLEDGE GRAPH RELATIONSHIPS:
{graph_context if graph_context else "No additional graph connections found"}

<|eot_id|><|start_header_id|>user<|end_header_id|>
{{question}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        )
        
        # NEW: Format retrieved docs
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        # NEW: Build chain with formatted docs
        rag_chain = (
            {"context": lambda x: format_docs(retrieved_docs), "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )
        
        # --- STREAM & AUDIO RESPONSE ---
        response_placeholder = st.empty()
        full_response = ""       
        
        try:
            for chunk in rag_chain.stream(refined_text):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # Generate audio
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

# --- VOICE INPUT SECTION ---
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