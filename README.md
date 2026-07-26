# Semantic RAG Pipeline with Hybrid Retrieval 🔍

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red.svg)](https://streamlit.io/)
[![Foundry Local SDK](https://img.shields.io/badge/Foundry_Local_SDK-Microsoft-0078D4.svg)](https://learn.microsoft.com/en-us/azure/ai-foundry/)
[![CUDA](https://img.shields.io/badge/CUDA-NVIDIA_RTX_5070_Ti-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade, 100% offline **Hybrid RAG (Retrieval-Augmented Generation)** application built for the **Microsoft Summer School 2026 (Building Your First Local RAG Application with Foundry Local)** project tracks.

---

## ⚡ Quick Start (Silent Windows Launcher)

On Windows, double-click **`start.vbs`** in the project directory to launch the application **without any console/terminal windows**.

---

## 🌟 Key Technical Features

- **Advanced Hybrid Retrieval:**
  - **Dense Retrieval:** Semantic vector search powered by `Qwen3-Embedding-0.6b` (1024-dimensional embeddings).
  - **Sparse Retrieval:** Keyword match powered by `BM25Okapi` with specialized text normalization.
- **Reciprocal Rank Fusion (RRF):** Merges dense and sparse ranks into a single unified score ranking ($k=60$).
- **Multilingual Cross-Encoder Reranking:** High-precision candidate reranking powered by `BAAI/bge-reranker-v2-m3` accelerated via **NVIDIA CUDA GPU**.
- **Local LLM Generation:** Real-time token streaming powered by `Phi-3.5-mini` via Microsoft Foundry Local SDK.
- **Plug-and-Play Vector Database:** Embedded **SQLite (`vector_store.db`)** database with NumPy-accelerated matrix dot-product cosine similarity (requires zero PostgreSQL installation).
- **Retrieval Guard (Zero Hallucination Gate):** Confidence thresholding gate that instantly (0.00s) rejects out-of-context queries.

---

## 📐 System Architecture & Pipeline Flow

```
[ User Query ]
       │
       ├──► Dense Vector Search (SQLite + Qwen3-Embedding) ──► Top-12 Dense Candidates
       │                                                            │
       ├──► Sparse Keyword Search (BM25 Engine) ──────────────────► Top-12 BM25 Candidates
       │                                                            │
       └─────────────────────► [ RRF Fusion ] ◄─────────────────────┘
                                     │
                                     ▼
                      [ BAAI Cross-Encoder Reranker ] ──► Top-3 Ranked Candidates (CUDA GPU)
                                     │
                                     ▼
                     [ Retrieval Guard Threshold ]
                         /                   \
            (Confidence < 0.10)         (Confidence >= 0.10)
                   /                             \
        Instant Rejection (0.00s)     [ Phi-3.5-mini LLM Stream ]
```

---

## 📂 Project Repository Structure

```
Semantic RAG Pipeline with Hybrid Retrieval/
│
├── start.vbs              # Silent Windows Launcher (Zero console windows)
├── app.py                 # Streamlit UI & interactive RAG application
├── retriever.py           # Hybrid retrieval orchestrator (Dense + Sparse + RRF + Rerank)
├── vector_store.py        # SQLite + NumPy vector database with optional Postgres fallback
├── bm25_index.py          # BM25 sparse search engine with text normalization
├── reranker.py            # Multilingual BAAI Cross-Encoder reranker (CUDA GPU)
├── embedder.py            # Foundry Local Qwen3-Embedding client
├── chat_model.py          # Foundry Local Phi-3.5-mini LLM client
├── foundry_manager.py     # Thread-safe Foundry Local SDK singleton manager
├── ingest_json.py         # Data reading, batch embedding, and indexing script
├── data.json              # Knowledge base containing 21,282 QA document pairs
├── vector_store.db        # Pre-indexed SQLite vector database (~524 MB)
├── bm25_index.pkl         # Serialized BM25 index file
└── requirements.txt       # Declared Python dependencies
```

---

## 🚀 Installation & Running

### 1. Install Dependencies
Install Python dependencies via pip:

```bash
pip install -r requirements.txt
```

---

### 2. Launch Application
Double-click `start.vbs` or execute via terminal:

```bash
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`.

---

## 🎥 Project Submission Info

- **Project Title:** Building Your First Local RAG Application with Foundry Local
- **Program:** Microsoft Summer School 2026
- **Program Director:** Barbaros Günay (CSA Manager CSU Turkey, `barbg@microsoft.com`)
- **Video Presentation (Link):** `[Insert Your 2-Minute Demo Video Link Here]` *(Google Drive / OneDrive / YouTube)*

---

## 📜 License

Distributed under the MIT License.
