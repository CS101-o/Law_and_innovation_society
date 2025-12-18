# Troubleshooting Guide

## Issue: "No PDFs found in legal_docs or subfolders"

### Causes & Solutions

#### 1. Folder Names Have Trailing Spaces
**Check:**
```bash
ls -la legal_docs/ | grep " $"
```

**Fix:**
```bash
# Rename folders without trailing spaces
mv "Folder Name " "Folder Name"
```

#### 2. Wrong Working Directory
**Check:** Are you running from project root?
```bash
pwd
# Should show: /Users/kaanoktem/UOM-Law
```

**Fix:** Always run from project root
```bash
cd /Users/kaanoktem/UOM-Law
streamlit run app/lawyer_gui.py
```

#### 3. Streamlit Cache Issue
Streamlit caches results, so old "no PDFs" result might persist.

**Fix:**
- Stop Streamlit: `Ctrl+C`
- Delete cache: `rm -rf ~/.streamlit/`
- Restart: `streamlit run app/lawyer_gui.py`
- Or use "Rebuild Graph" button in sidebar

#### 4. PDFs Not in Subfolders
PDFs must be in domain folders, not loose in `legal_docs/`

**Wrong:**
```
legal_docs/
├── file1.pdf
└── file2.pdf
```

**Right:**
```
legal_docs/
├── Misrepresentation/
│   └── file1.pdf
└── Contractual Terms/
    └── file2.pdf
```

---

## Issue: "System not ready: Failed to initialize LLM/embeddings"

**Cause:** Ollama is not running

**Fix:**
```bash
# In a separate terminal, start Ollama
ollama serve

# Then in another terminal, verify models
ollama list
```

Should show:
- `uk-lawyer` (your fine-tuned model)
- `nomic-embed-text` (for embedding)
- `qwen2.5:0.5b` (optional, for classification)

---

## Issue: "System not ready: Error loading RAG system"

**Causes:** PDF loading or vector db issue

**Solutions:**

1. **Check PDF validity:**
```bash
python3 -c "
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('legal_docs/Misrepresentation/Misrepresentation.pdf')
docs = loader.load()
print(f'Loaded {len(docs)} pages')
"
```

2. **Clear vector database:**
```bash
rm -rf chroma_db/
rm legal_knowledge_graph.pkl
```

3. **Rebuild from scratch:**
```bash
streamlit run app/lawyer_gui.py
# Click "Rebuild Graph" button in sidebar
```

---

## Issue: "ModuleNotFoundError: No module named 'core'"

**Cause:** Running from wrong directory or old Python path

**Fix:** All app files have automatic path fixing, but make sure:
1. Run from project root
2. Restart terminal/IDE
3. Check no syntax errors in imports

---

## Issue: Slow Performance

**Causes:**
- First run (building knowledge graph)
- Large PDFs
- Ollama model loading

**Solutions:**
1. Be patient on first run
2. Check available RAM: `top -l 1 | grep PhysMem`
3. Monitor Ollama: `ollama list` shows loaded models
4. Use "Demo Mode" to test without Ollama

---

## Issue: Voice Input Not Working

**Prerequisites:**
```bash
brew install ffmpeg
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Check Audio:**
- Microphone permissions: System Preferences → Security & Privacy → Microphone
- FFmpeg installed: `which ffmpeg`
- Packages: `pip show streamlit-mic-recorder`

---

## Issue: Graph Context Not Showing

**Cause:** Knowledge graph not properly built or hybrid_retriever not adding context

**Debugging:**
```bash
python scripts/setup_graphrag.py
```

Check output for:
- Entities extracted
- Relationships found
- Graph saved successfully

---

## Debug Mode

Test individual components:

### Test PDF Discovery
```bash
python3 << 'EOF'
import os
DATA_FOLDER = './legal_docs'
pdf_files = []
for root, dirs, files in os.walk(DATA_FOLDER):
    for file in files:
        if file.endswith('.pdf'):
            rel_path = os.path.relpath(os.path.join(root, file), DATA_FOLDER)
            pdf_files.append(rel_path)
print(f"Found {len(pdf_files)} PDFs:")
for f in pdf_files:
    print(f"  - {f}")
EOF
```

### Test Query Refiner
```bash
python scripts/testRef.py
```

### Test Embeddings
```bash
python3 << 'EOF'
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
result = embeddings.embed_query("test")
print(f"Embedding dimension: {len(result)}")
EOF
```

---

## Getting Help

1. Check this guide first
2. Check app error messages (expand them)
3. Run debug commands above
4. Check Ollama logs: `ollama logs`
5. Verify folder structure: `find legal_docs -name "*.pdf"`
