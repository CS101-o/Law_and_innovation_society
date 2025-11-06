# Law_and_innovation_society
# To download guff file (the fine_tuned model) ----> https://drive.google.com/file/d/1DTs8bWrYydRTBkSZhGS9Vm9Xz9-Koqu5/view?usp=sharing
After downloading the file, place it in the folder
# To run gui, use ---->  streamlit run lawyer_gui.py

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

## 📂 Setup Legal Documents
Create a folder named `legal_docs` in this directory. Add your reference PDFs here.
* *Recommended:* `Consumer_Rights_Act_2015.pdf`
* *Recommended:* `Tesco_v_USDAW_2024.pdf`

## This is an AI research tool, not a substitute for a qualified solicitor. Always verify legal citations.
