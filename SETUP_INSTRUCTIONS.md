# Setup & Running Instructions

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup AI Models
Download the fine-tuned model from: [Google Drive Link](https://drive.google.com/file/d/1DTs8bWrYydRTBkSZhGS9Vm9Xz9-Koqu5/view?usp=sharing)

Place it in the project root directory, then create the models:
```bash
# Create the fine-tuned model
ollama create uk-lawyer -f Modelfile

# Download embedding model
ollama pull nomic-embed-text

# (Optional) Download small classification model
ollama pull qwen2.5:0.5b
```

### 3. Add Legal Documents
Create a `legal_docs` folder and organize PDFs by domain:
```
legal_docs/
├── Promissory Estoppel/
│   └── your_cases.pdf
├── Misrepresentation/
│   └── your_cases.pdf
├── Contractual Terms/
│   └── your_cases.pdf
└── ...
```

### 4. Run the Application

**Main Web Interface (Recommended):**
```bash
streamlit run app/lawyer_gui.py
```

**Alternative CLI Interface:**
```bash
python app/lawyer_app.py
```

**Simple Chat (No RAG):**
```bash
python app/simple_lawyer.py
```

## Setup & Testing

### Build Knowledge Graph
```bash
python scripts/setup_graphrag.py
```

### Test Query Refinement
```bash
python scripts/testRef.py
```

## Project Structure

```
app/                    # User-facing applications
├── lawyer_gui.py      # Main Streamlit web UI
├── lawyer_app.py      # Alternative console app
└── simple_lawyer.py    # Simple CLI version

core/                   # RAG Engine
├── graph_builder.py    # Knowledge graph
├── hybrid_retriever.py # Vector + Graph retrieval
└── refiner.py         # Query refinement

src/                    # Utilities
├── pdf_classifier.py   # Content classification
├── tts.py             # Text-to-speech
└── waiting_music.py   # UI audio

scripts/                # Setup & testing
├── setup_graphrag.py   # Initialize graph
└── testRef.py         # Test refiner

legal_docs/             # Your case PDFs
chroma_db/             # Vector database cache
```

## Features

- **Voice Input/Output** - Record questions, hear answers
- **IRAC Analysis** - Issue, Rule, Analysis, Conclusion format
- **Domain Classification** - Automatic categorization
- **GraphRAG** - Enhanced retrieval with knowledge graphs
- **Local Privacy** - All processing on your machine

## Troubleshooting

### ModuleNotFoundError: No module named 'core'
- Make sure you're running from the project root directory
- Path fixing is automatic in all app files (sys.path.insert)

### Ollama Connection Error
- Ensure Ollama is running: `ollama serve`
- Check models are installed: `ollama list`

### No PDFs Found
- Create `legal_docs` folder
- Organize PDFs in subfolders by domain
- Run `streamlit run app/lawyer_gui.py` from project root

### Graph Building Fails
- Ensure PDFs are valid and readable
- Check `legal_docs` folder exists
- Try: `python scripts/setup_graphrag.py`

## Important Notes

- This is a research tool, not a substitute for legal advice
- Always verify citations with official sources
- Requires Ollama to be running locally
- First run may take time building the knowledge graph
