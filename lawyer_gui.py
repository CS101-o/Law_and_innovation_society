import streamlit as st
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
MODEL_NAME = "uk-lawyer"
DATA_FOLDER = "./legal_docs"

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

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. MAIN CHAT INTERFACE ---
if prompt := st.chat_input("Describe the client's legal scenario..."):
    # Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    if rag_chain:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream the answer
            try:
                for chunk in rag_chain.stream(prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.error("System not initialized. Check PDF folder.")