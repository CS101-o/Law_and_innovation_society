# UOM Law - Project Structure

## Directory Organization

```
UOM-Law/
├── app/                          # User-facing applications
│   ├── lawyer_gui.py            # Main Streamlit web interface
│   ├── lawyer_app.py            # Alternative console application
│   ├── simple_lawyer.py          # Simple CLI version
│   └── __init__.py
│
├── core/                         # Core RAG components
│   ├── graph_builder.py         # Knowledge graph construction
│   ├── hybrid_retriever.py       # Vector + Graph RAG retrieval
│   ├── refiner.py               # Query refinement & domain classification
│   └── __init__.py
│
├── src/                          # Utility modules
│   ├── pdf_classifier.py        # Content type classification
│   ├── tts.py                   # Text-to-speech
│   ├── waiting_music.py         # UI audio feedback
│   └── audio/                   # Audio files
│
├── scripts/                      # Setup & testing utilities
│   ├── setup_graphrag.py        # Initialize knowledge graph
│   ├── testRef.py               # Test query refinement
│   └── __init__.py
│
├── legal_docs/                   # Input: PDF case files (organized by domain)
│   ├── Promissory Estoppel/
│   ├── Misrepresentation/
│   ├── Contractual Terms/
│   └── ...
│
├── chroma_db/                    # Vector database cache
└── legal_knowledge_graph.pkl     # Serialized knowledge graph

```

## Module Descriptions

### app/ - Applications

- **lawyer_gui.py** (Main Entry Point)
  - Streamlit web interface
  - Voice input/output support
  - Real-time analysis display
  - Chat history management
  - Run: `streamlit run app/lawyer_gui.py`

- **lawyer_app.py**
  - Alternative console application with full RAG
  - Command-line interface

- **simple_lawyer.py**
  - Minimal CLI version
  - Direct LLM interaction without RAG

### core/ - RAG Components

- **graph_builder.py**
  - Builds NetworkX knowledge graph from PDFs
  - Stores relationships between legal concepts
  - Saves/loads to legal_knowledge_graph.pkl

- **hybrid_retriever.py**
  - Combines vector similarity search with graph traversal
  - Retrieves relevant documents
  - Adds graph context to results

- **refiner.py**
  - Classifies user queries by legal domain
  - Refines queries for better retrieval
  - Domains: Promissory Estoppel, Misrepresentation, etc.

### src/ - Utilities

- **pdf_classifier.py**
  - Categorizes PDFs by content type
  - Categories: Goods, Services, Digital, Finance, Dispute

- **tts.py**
  - Text-to-speech synthesis
  - Speech-to-text transcription

- **waiting_music.py**
  - Audio feedback during processing

## How to Run

### Main Application (Web UI)
```bash
cd /Users/kaanoktem/UOM-Law
streamlit run app/lawyer_gui.py
```

### Setup Knowledge Graph
```bash
python scripts/setup_graphrag.py
```

### Test Query Refiner
```bash
python scripts/testRef.py
```

## Data Flow

```
User Query
    ↓
[refiner.py] → Classify domain + Refine query
    ↓
[hybrid_retriever.py] → Vector search + Graph expansion
    ↓
[Retrieved Documents + Graph Context]
    ↓
[LLM with prompt] → Generate IRAC response
    ↓
User Response (Text + Audio)
```

## Import Paths

After reorganization, update imports:

| Old Path | New Path |
|----------|----------|
| `from graph_builder import ...` | `from core.graph_builder import ...` |
| `from hybrid_retriever import ...` | `from core.hybrid_retriever import ...` |
| `from refiner import ...` | `from core.refiner import ...` |
| `from src.pdf_classifier import ...` | `from src.pdf_classifier import ...` (unchanged) |

## Notes

- All apps must be run from the project root
- Legal PDFs should be organized in `legal_docs/` by domain
- Vector cache is automatically managed in `chroma_db/`
- Knowledge graph is rebuilt when PDFs change
