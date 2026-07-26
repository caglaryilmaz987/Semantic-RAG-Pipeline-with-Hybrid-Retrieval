"""
ingest_json.py — Data Ingestion & Dual Indexing Script

Pipeline:
  1. Reads QA pairs from data.json.
  2. Embeds documents using Qwen3-Embedding-0.6b and saves them into SQLite vector_store.db.
  3. Builds the BM25 sparse keyword index and serializes it into bm25_index.pkl.

Usage:
    python ingest_json.py
"""

import os
import json
import time
from embedder import Embedder
from vector_store import VectorStore
from bm25_index import BM25Index


def read_json_data(file_path: str = "data.json") -> list[str]:
    """Reads input/output pairs from data.json and formats them into single document chunks."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    print(f"📄 Reading data file '{file_path}'...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    document_chunks = []
    for item in data:
        question = item.get("input", "").strip()
        answer = item.get("output", "").strip()

        if question and answer:
            combined_text = f"User Question: {question}\nAssistant Answer: {answer}"
            document_chunks.append(combined_text)

    print(f"✓ Formatted {len(document_chunks)} valid QA document chunks.")
    return document_chunks


class Ingestion:
    """Orchestrates embedding generation and index persistence."""
    def __init__(self):
        print("🧠 Initializing AI models...")
        self._embedder = Embedder()
        self._vector_store = VectorStore(embedding_dim=self._embedder.dimension)
        self._bm25_index = BM25Index()

    def run(self, documents: list[str], batch_size: int = 100):
        if not documents:
            print("⚠️ No data available to ingest.")
            return

        start_time = time.time()
        print(f"\n🚀 Ingestion Started: Processing {len(documents)} items (Batch size: {batch_size})...\n")

        # 1. Dense: Vector Store Ingestion
        print("🧹 Resetting existing database tables...")
        self._vector_store.reset()

        print("⚡ Generating embeddings and inserting into SQLite...")
        total_batches = (len(documents) + batch_size - 1) // batch_size
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_indices = list(range(i, i + len(batch)))
            vectors = self._embedder.embed_batch(batch)
            
            for doc_id, text, emb in zip(batch_indices, batch, vectors):
                self._vector_store.add(doc_id, text, emb)

            current_batch = (i // batch_size) + 1
            if current_batch % 10 == 0 or current_batch == total_batches:
                print(f"  [Batch {current_batch}/{total_batches}] {min(i + batch_size, len(documents))}/{len(documents)} records written...")

        print(f"✓ Dense Vector Store: {self._vector_store.count()} records indexed successfully.")

        # 2. Sparse: BM25 Indexing
        print("🔤 Building BM25 sparse index...")
        self._bm25_index.build(documents)
        self._bm25_index.save("bm25_index.pkl")
        print("✓ Sparse BM25 Index saved to 'bm25_index.pkl'.")

        # Cleanup
        self._embedder.unload()
        elapsed = time.time() - start_time
        print(f"\n🎉 Ingestion Pipeline Completed in {elapsed:.2f} seconds!")


if __name__ == "__main__":
    texts = read_json_data("data.json")
    Ingestion().run(texts)
