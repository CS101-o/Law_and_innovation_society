# AI UK Contract Law Advisor

To run the application:
```bash
streamlit run app/lawyer_gui.py
```

* **Specialized Knowledge:** Trained on UK Contract Law principles (IRAC Method).
* **Document Awareness:** Reads PDF statutes (e.g., Consumer Rights Act 2015) before answering.
* **Tiered Logic:** Automatically distinguishes between Goods, Services, and Digital Content.
* **Local Privacy:** Runs entirely on your machine using Ollama—no data leaves your computer.

1.  **Clone/Download this repository.**
2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Setup the AI Model:**
    * Place your `UK_Law_Real_Final.gguf` file in the project folder.
    * Create the model in Ollama:
        ```bash
        ollama create uk-lawyer -f Modelfile
        ```
    * Download the embedding model (for reading PDFs):
        ```bash
        ollama pull nomic-embed-text
        ```

## 🎙️ Voice Interaction
- Use the microphone widget in the Streamlit app to dictate a question instead of typing. The audio is transcribed locally with the SpeechRecognition library before being sent to the AI.
- Every AI response now includes a **Read Aloud** button powered by gTTS. Click it to hear the answer immediately inside the browser.
- Install `ffmpeg` (e.g., `brew install ffmpeg`) to enable the audio conversions required by the recorder.

## 📂 Legal Documents

Your PDFs should be organized in `legal_docs/` by legal domain:
```
legal_docs/
├── Misrepresentation/
│   ├── Misrepresentation.pdf
│   └── Misrepresentation-Act-1967.pdf
├── Contractual Terms/
│   ├── Breach of Contract.pdf
│   └── Contractual Terms.pdf
├── Offer and Acceptance/
│   ├── ukpga_20150015_en.pdf
│   └── Offer, Acceptance.pdf
├── Promissory Estoppel/
│   └── Intention, Consideration and Promissory Estoppel.pdf
└── Mistake - Mutual Mistake/
    └── Mistake.pdf
```

**Current Setup:** You have 8 PDFs already organized and ready to use!

## Troubleshooting

If you see "No PDFs found" after adding files:
1. Make sure folder names don't have trailing spaces
2. Stop and restart Streamlit: `Ctrl+C` then `streamlit run app/lawyer_gui.py` again
3. Click "Rebuild Graph" button in the sidebar to force re-indexing

## Disclaimer

This is an AI research tool, not a substitute for a qualified solicitor. Always verify legal citations and consult with professionals before making legal decisions.
