# Adaptive Drift-Aware Legal AI Chatbot for UK Law Updates

A dissertation-ready, professionally structured project for building a **drift-aware Retrieval-Augmented Generation (RAG) legal chatbot** focused on **UK legal and regulatory updates**. The project includes a working Streamlit UI, modular Python backend, ingestion and indexing scripts, drift-monitoring logic, evaluation scaffolding, and an academic report pack.

## 1. Project Overview

This system is designed to answer questions over a curated UK legal corpus while reducing hallucination through retrieval grounding and monitoring potential regulatory drift over time.

### Core features
- Streamlit-based chat UI
- Retrieval-Augmented Generation pipeline
- Local corpus ingestion and indexing
- Semantic vector search with sentence-transformer embeddings and offline TF-IDF fallback
- Optional LLM backends:
  - OpenAI-compatible API
  - Ollama local models
  - Fallback extractive mode when no generator is configured
- Drift monitoring using:
  - Population Stability Index (PSI)
  - Kullback-Leibler (KL) divergence
- Citation-aware answer formatting
- Dissertation-friendly documentation and architecture

## 2. Folder Structure

```text
legal_ai_chatbot_project/
├── app/
│   └── streamlit_app.py
├── assets/
│   └── architecture.png
├── data/
│   ├── source_documents/
│   │   ├── sample_fca_guidance.txt
│   │   ├── sample_ico_guidance.txt
│   │   └── sample_uk_legislation.txt
│   └── vector_store/
├── docs/
│   ├── final_report.docx
│   ├── final_report.md
│   └── architecture_mermaid.md
├── scripts/
│   ├── build_index.py
│   └── run_app.py
├── src/
│   └── legal_chatbot/
│       ├── __init__.py
│       ├── config.py
│       ├── drift.py
│       ├── generator.py
│       ├── ingest.py
│       ├── pipeline.py
│       ├── prompts.py
│       ├── retrieval.py
│       └── utils.py
├── tests/
│   └── test_drift.py
├── .env.example
├── requirements.txt
└── README.md
```

## 3. Installation

### Step 1: Create a virtual environment
```bash
python -m venv .venv
```

### Step 2: Activate it

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure environment variables
Copy `.env.example` to `.env` and add your settings.

```bash
cp .env.example .env
```

## 4. Quick Start

### Build the vector index
```bash
python scripts/build_index.py
```

### Run the app
```bash
python scripts/run_app.py
```

Or directly:
```bash
streamlit run app/streamlit_app.py
```

## 5. Supported LLM Modes

Set `LLM_PROVIDER` in your `.env` file.

### Option A: OpenAI-compatible API
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

### Option B: Ollama
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
OLLAMA_BASE_URL=http://localhost:11434
```

### Option C: Extractive fallback
```env
LLM_PROVIDER=extractive
```

This fallback mode does not require an external model API. It produces grounded summary-style answers from retrieved passages, which is useful for local testing and dissertation demonstration.

## 6. How the System Works

1. Legal documents are loaded from `data/source_documents/`
2. Documents are split into semantically manageable chunks
3. Chunks are embedded using `sentence-transformers` when available, with a TF-IDF fallback for restricted environments
4. Vectors are stored locally for similarity search
5. At query time:
   - the query is embedded
   - top-k passages are retrieved
   - reranking is applied with cosine similarity
   - drift metrics are computed against baseline retrieval behaviour
   - an answer is produced with citations
6. The UI displays:
   - response
   - supporting sources
   - retrieval metadata
   - drift warnings

## 7. Dissertation Alignment

This project directly supports a dissertation on:
- Retrieval-Augmented Generation for legal QA
- Hallucination reduction
- Concept drift in dynamic regulatory corpora
- Monitoring and evaluation of trustworthy legal AI

The included report in `docs/` is written in professional UK English and can be adapted into your final university submission.

## 8. Evaluation Suggestions

Recommended experiments:
- Vanilla baseline vs RAG vs Drift-aware RAG
- Top-k sensitivity testing
- Citation precision scoring
- Hallucination rate assessment
- Latency benchmarking
- PSI and KL divergence monitoring before and after corpus updates

## 9. Important Note

This system is a **decision-support tool**, not a substitute for qualified legal advice. Outputs should always be checked against official UK legal and regulatory sources.

## 10. Troubleshooting

### Streamlit import errors
Ensure you installed the full requirements list and activated the correct virtual environment.

### OpenAI or Ollama not configured
Use:
```env
LLM_PROVIDER=extractive
```
to run the system without a remote generation backend.

## 11. Suggested Dissertation Usage

You can use this package as:
- a final project prototype
- an implementation appendix
- a demonstrator for viva or presentation
- a base for experimental evaluation chapters

## 12. Licence

For academic and portfolio use.
