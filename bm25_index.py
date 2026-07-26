"""
bm25_index.py — Sparse Retrieval (BM25 Engine)
BM25Okapi index implementation with multilingual and Turkish character normalization rules.

Key Features:
  - Custom lowercasing for accurate token matching across special characters (İ->i, I->ı, etc.).
  - Assigns persistent numeric IDs to documents for Reciprocal Rank Fusion (RRF) matching.
"""

import pickle
import os
from rank_bm25 import BM25Okapi


def clean_lower(text: str) -> str:
    """
    Normalizes text for keyword matching, handling special language characters accurately.
    """
    if not text:
        return ""
    translation_table = str.maketrans({
        "İ": "i",
        "I": "ı",
        "Ğ": "ğ",
        "Ü": "ü",
        "Ş": "ş",
        "Ö": "ö",
        "Ç": "ç",
    })
    return text.translate(translation_table).lower()


class BM25Index:
    def __init__(self):
        self._documents: list[str] = []
        self._bm25: BM25Okapi | None = None

    def build(self, documents: list[str]):
        """
        Tokenizes documents and builds the BM25Okapi index.
        """
        self._documents = documents
        tokenized_corpus = [clean_lower(doc).split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Executes sparse keyword search against the BM25 index.
        
        Returns:
            [{"id": int, "text": str, "score": float}, ...]
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 Index has not been initialized. Call build() or load() first.")

        tokenized_query = clean_lower(query).split()
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        results = [
            {"id": i, "metin": self._documents[i], "skor": float(scores[i])}
            for i in range(len(self._documents))
        ]
        results.sort(key=lambda x: x["skor"], reverse=True)
        return results[:top_k]

    def save(self, path: str = "bm25_index.pkl"):
        """Saves the serialized BM25 index and corpus to disk."""
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "belgeler": self._documents}, f)

    def load(self, path: str = "bm25_index.pkl"):
        """Loads the serialized BM25 index from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"BM25 index pickle file not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._documents = data["belgeler"]