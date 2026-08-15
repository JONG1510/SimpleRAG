# University Travel Assistant (RAG Agent)

An interactive Retrieval-Augmented Generation (RAG) assistant designed to help employees look up university travel policies and estimate work trip logistics. Built with **LlamaIndex**, **Google Gemini**, and **Gradio**.

---

## 🚀 Features

- **Policy RAG Search (`travel_policy_lookup`)**: Performs semantic search across official policy documents stored locally. Answers are strictly grounded in context to prevent policy hallucinations.
- **Logistics & Cost Estimator (`search_travel_cost`)**: An automated function tool that estimates real-world flight, hotel, and ground travel costs.
- **Input Guardrails**: Keyword-based pre-filtering to catch off-topic requests and prompt injection attempts before invoking agent tools.
- **Disk-Based Index Persistence**: Embeds source documents once and saves the vector index to `./storage` for fast subsequent application startups.
- **Interactive Web Interface**: Clean UI built with Gradio that handles user chat history and session states.

---

## 🛠️ Project Architecture
```
├── data/              # Place policy documents (.txt, .pdf) here
├── storage/           # Local folder generated to store persisted vector index
├── app.py            # Primary application script (Agent logic & Gradio interface)
├── requirements.txt   # Python dependency list
└── .env               # Environment file for API keys[cite: 1]  
```
# University Travel Assistant (RAG Agent)

An interactive Retrieval-Augmented Generation (RAG) assistant designed to help employees look up university travel policies and estimate work trip logistics. Built with **LlamaIndex**, **Google Gemini**, and **Gradio**.

---

