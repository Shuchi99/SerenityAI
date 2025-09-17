# SerenityAI – Mental Health Chat Companion 🧘

A conversational assistant designed to support emotional well-being.  
Built with **LangChain**, **FAISS**, **Llama 3.1 (8B via Ollama)**, and **HuggingFace Transformers**.

---

## 🚀 Features (Step 5)
- **Emotion-Aware Responses** – Uses a RoBERTa-based classifier to detect emotions in user input and condition the system prompt for empathetic, context-aware replies.
- **Retrieval-Augmented Generation (RAG)** – Combines a FAISS vector store + LangChain retriever to ground responses in a curated knowledge base (breathing techniques, sleep hygiene, grounding exercises).

---

## 🛠️ Tech Stack
- **Backend:** Python, LangChain, FAISS, Ollama (Llama 3.1 8B)
- **ML/NLP:** HuggingFace Transformers (RoBERTa for emotion classification)
- **UI:** Gradio (interactive chat interface)
- **Infra:** `.env` config with Pydantic Settings

---

## 📂 Project Structure (Key Parts)
```
app/
 ├── pipeline/         # RAG + emotion pipeline orchestration
 ├── rag/              # FAISS retriever & ingestion scripts
 ├── services/
 │    └── emotion/     # RoBERTa emotion classification
 ├── ui/               # Gradio chat frontend
 └── llm.py            # Llama model wrapper
knowledge_base/        # Curated mental wellness resources
scripts/               # ingest_kb.py, retrieve_demo.py
```

---

## ⚡ Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/serenity-ai.git
cd serenity-ai
python -m venv .venv
.venv\Scripts\activate    # On Windows
pip install -r requirements.txt
```

### 2. Pull Llama 3.1 model
```bash
ollama pull llama3.1:8b
```

### 3. Ingest the Knowledge Base
```bash
python -m scripts.ingest_kb
```

### 4. Launch the Gradio App
```bash
python -m app.ui.gradio_app
```

Open http://127.0.0.1:7860 in your browser.

---

## 🧠 Example Query
> **User:** I feel anxious and can't sleep.  
> **Assistant:** Suggests 4-7-8 Breathing, Box Breathing, and 5-4-3-2-1 Grounding with short explanations.

---

## 📌 Next Steps
- Step 6: Add safety guardrails (self-harm classifier + crisis resources)
- Step 7: Streaming responses & session memory
- Step 8: Persistent chat history & analytics
