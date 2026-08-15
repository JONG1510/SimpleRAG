# University Travel Assistant

An interactive Retrieval-Augmented Generation (RAG) assistant that answers workplace travel policy questions and estimates trip costs using **LlamaIndex**, **Google Gemini**, and **Gradio**.

---

**Features**

* **Policy RAG Search (`travel_policy_lookup`)**: Performs semantic retrieval over policy documents in `data/` using `SentenceSplitter` (chunk size: 512, overlap: 50). Answers are constrained strictly to context.
* **Travel Cost Estimator (`search_travel_cost`)**: A `FunctionTool` that estimates flight, hotel, and ground travel costs.
* **Input Pre-filtering**: Blocks off-topic queries and prompt injection keywords before tool execution.
* **Disk Persistence**: Stores and loads vector index data in `./storage` to avoid re-embedding on startup.
* **Session Memory & UI**: Converts Gradio chat history into LlamaIndex `ChatMessage` objects for multi-turn conversations in `gr.ChatInterface`.
* **Audit Logging**: Records query history, blocked requests, and exceptions to `agent_queries.log`.

---

**Repository Architecture**

```text

├── data/                  # Source policy documents parsed by SimpleDirectoryReader
├── storage/               # Created by StorageContext to persist vector index to disk
│   ├── docstore.json      # LlamaIndex document metadata store
│   ├── index_store.json   # Vector store index metadata
│   └── default__vector_store.json # Persisted document chunk embeddings
├── .env                   # Environment variables (GOOGLE_API_KEY, GEMINI_MODEL)
├── agent_queries.log      # Runtime log generated via logging.basicConfig()
├── main.py                # Main script containing tools, agent logic, and Gradio UI
└── requirements.txt       # Dependencies (llama-index, gradio, python-dotenv)
```
---

## ⚡ Quick Start (Local Setup)

### Prerequisites

- **Python**: v3.10 or higher
- **Google Gemini API Key**: Obtainable via [Google AI Studio](https://aistudio.google.com/)

### 1. Environment Setup

Create a .env file in the root directory:
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash

### 2. Document Preparation
Place university travel policy documents inside the data/ directory.

### 3. Run Application
python main.py

The server will launch gr.ChatInterface on 0.0.0.0 using the configured PORT or default 7860.

