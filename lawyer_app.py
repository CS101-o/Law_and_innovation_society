import os
import sys
# Updated import to fix the warning you saw earlier
from langchain_ollama import ChatOllama, OllamaEmbeddings 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
MODEL_NAME = "uk-lawyer"       # Your custom trained model
EMBED_MODEL = "nomic-embed-text" # The model that 'reads' the PDF text
DATA_FOLDER = "./legal_docs"   # The folder where you put your PDFs

def main():
    print(f"⚖️  Initializing {MODEL_NAME} with Document Support...")

    # 1. SETUP LLM & EMBEDDINGS
    try:
        llm = ChatOllama(
            model=MODEL_NAME, 
            temperature=0.1,
            num_ctx=8192, # Large memory for reading docs
            # Stop the model from rambling
            stop=["<|eot_id|>", "<|start_header_id|>", "📝", "Client Question:"] 
        )
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    except Exception as e:
        print(f"❌ Error starting Ollama: {e}")
        return

    # 2. LOAD PDFS
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"⚠️  Folder '{DATA_FOLDER}' created.")
        print("👉 ACTION REQUIRED: Put your PDF statutes/contracts inside it and run this script again.")
        return

    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"❌ No PDFs found in {DATA_FOLDER}.")
        print("Please add a PDF (e.g., 'Consumer_Rights_Act_2015.pdf') and retry.")
        return

    print(f"📂 Reading {len(pdf_files)} document(s)...")
    docs = []
    for file in pdf_files:
        path = os.path.join(DATA_FOLDER, file)
        print(f"   - Indexing: {file}")
        try:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"   ⚠️ Could not read {file}: {e}")

    if not docs:
        print("❌ No readable text found in documents.")
        return

    # 3. CREATE KNOWLEDGE BASE (Vector Store)
    print("🧠 Memorizing document contents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Create the database in memory
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="uk_law_rag"
    )
    retriever = vectorstore.as_retriever()

    # 4. STRICT RAG PROMPT
    # We force it to use the Context (PDF) first, then its training
    template = """<|start_header_id|>system<|end_header_id|>

You are an expert UK Contract Law consultant.
Answer the user's question using the IRAC method.
You must prioritize the information provided in the CONTEXT below.
Cite specific sections from the Context if available.

CONTEXT FROM DOCUMENTS:
{context}
<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    prompt = ChatPromptTemplate.from_template(template)

    # 5. BUILD CHAIN
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. CHAT LOOP
    print("\n✅ LAWYER + LIBRARY READY. Type 'exit' to quit.\n" + "="*50)
    while True:
        try:
            query = input("\n📝 Client Scenario: ")
            if query.lower() in ["exit", "quit"]: break
            
            print("\n🔍 Searching Documents & Thinking...\n")
            
            # Stream response
            for chunk in rag_chain.stream(query):
                print(chunk, end="", flush=True)
            
            print("\n" + "-" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()