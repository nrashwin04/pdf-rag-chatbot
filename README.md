# pdf-rag-chatbot 📚

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDFs and chat
with them using natural language. Built with LangChain, ChromaDB, and your choice
of LLM backend — run locally for free with Ollama or use the Claude API.

> Built as part of my MSc CS portfolio to explore RAG pipelines, vector search, and LLM integration.

---

## Demo

> Upload any PDF → ask questions → get grounded answers with source citations

![demo placeholder](assets/demo.png)

---

## Features

- 📄 Upload multiple PDFs and query across all of them
- 🔍 Semantic search using sentence-transformers (all-MiniLM-L6-v2)
- 🗃️ Vector storage with ChromaDB — no external database needed
- 🤖 Dual LLM support — Ollama (free, local) or Claude API
- 📎 Source citations shown for every answer
- 💬 Persistent chat history within session

---

## How it works

```
PDF upload → text extraction → chunking (500 tokens, 50 overlap)
    → embeddings (HuggingFace) → ChromaDB storage
    → user query → similarity search (top 3 chunks)
    → LLM generates grounded answer → sources displayed
```

---

## Tech stack

| Layer | Tech |
|---|---|
| UI | Streamlit |
| RAG orchestration | LangChain |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| PDF parsing | PyPDFLoader |
| LLM (local) | Ollama (llama3) |
| LLM (API) | Claude (claude-sonnet) via Anthropic |

---

## Getting started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Choose your LLM backend

**Option A — Ollama (free, local):**
```bash
# Install from https://ollama.com then:
ollama serve
ollama pull llama3
```
Select "Ollama (Local)" in the sidebar.

**Option B — Claude API:**
Get a free API key at [console.anthropic.com](https://console.anthropic.com),
select "Claude (API)" in the sidebar, and paste your key.

---

## Project structure

```
pdf-rag-chatbot/
├── app.py                  # main Streamlit app
├── requirements.txt
├── .env.example            # env variable template
├── .gitignore
├── assets/
│   └── demo.png            # screenshot for README
└── README.md
```

---

## Roadmap

- [ ] Persistent vector store across sessions
- [ ] Support for .txt and .docx files
- [ ] Streaming responses
- [ ] Chat history export
- [ ] Deploy to Streamlit Community Cloud

---

*Built with LangChain + ChromaDB · MSc CS portfolio project*
