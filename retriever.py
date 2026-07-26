"""
retriever.py — Hybrid Retrieval Orchestrator
Dense (Semantic Vector) + Sparse (BM25) → Reciprocal Rank Fusion (RRF) → Cross-Encoder Reranker

Execution Flow:
    1. Query -> Dense Vector Search (VectorStore)
    2. Query -> Sparse Keyword Search (BM25Index)
    3. RRF Fusion -> Reciprocal Rank Fusion rank aggregation
    4. Reranking -> Multilingual Cross-Encoder (BAAI/bge-reranker-v2-m3)
"""

from vector_store import VectorStore
from bm25_index import BM25Index
from reranker import Reranker


class HybridRetriever:
    """
    Hybrid Search Orchestrator combining Dense and Sparse retrieval with RRF and Reranking.
    """
    def __init__(
        self,
        emb_client,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
    ):
        self._emb_client = emb_client
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._reranker = reranker
        self._rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        use_reranker: bool = True,
        rrf_k_override: int | None = None,
    ) -> list[dict]:
        """
        Executes hybrid retrieval pipeline.
        
        Returns:
            [
              {
                "id": int,
                "metin": str,
                "rrf_skor": float,
                "rerank_skor": float,
                "kaynaklar": list[str], # ["dense", "bm25"]
                "dense_rank": int | None,
                "bm25_rank": int | None,
              },
              ...
            ]
        """
        k_const = rrf_k_override if rrf_k_override is not None else self._rrf_k
        fetch_k = max(top_k * 2, 8)  # Optimized candidate pool size for speed

        # 1. Dense Vector Search (~2ms)
        query_vector = self._embed(query)
        dense_results = self._vector_store.search(query_vector, top_k=fetch_k)
        self._add_ranks(dense_results)

        # 2. Sparse BM25 Search (~5ms)
        sparse_results = self._bm25_index.search(query, top_k=fetch_k)
        self._add_ranks(sparse_results)

        # 3. Reciprocal Rank Fusion (~1ms)
        fused = self._rrf_fusion(dense_results, sparse_results, rrf_k=k_const)

        # 4. Cross-Encoder Reranking
        if use_reranker and self._reranker is not None and len(fused) > 0:
            final = self._reranker.rerank(query, fused, top_k=top_k)
        else:
            final = fused[:top_k]
            for f in final:
                if "rerank_skor" not in f:
                    f["rerank_skor"] = 0.0

        return final

    def _embed(self, text: str) -> list[float]:
        """Generates embedding for input query text."""
        if hasattr(self._emb_client, "embed"):
            return self._emb_client.embed(text)
        elif hasattr(self._emb_client, "generate_embedding"):
            response = self._emb_client.generate_embedding(text)
            return response.data[0].embedding
        else:
            raise AttributeError("Embedding client does not support embed() or generate_embedding()")

    @staticmethod
    def _add_ranks(results: list[dict]):
        """Attaches 1-indexed rank order to candidate search results."""
        for i, item in enumerate(results):
            item["rank"] = i + 1

    def _rrf_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        rrf_k: int,
    ) -> list[dict]:
        """
        Calculates Reciprocal Rank Fusion (RRF) scores across dense and sparse candidate lists.
        Score(d) = sum(1 / (k + rank(d)))
        """
        scores: dict[int, dict] = {}

        for item in dense_results:
            doc_id = item["id"]
            if doc_id not in scores:
                scores[doc_id] = {
                    "id": doc_id,
                    "metin": item["metin"],
                    "rrf_skor": 0.0,
                    "kaynaklar": [],
                    "dense_rank": item["rank"],
                    "bm25_rank": None,
                }
            scores[doc_id]["rrf_skor"] += 1.0 / (rrf_k + item["rank"])
            if "dense" not in scores[doc_id]["kaynaklar"]:
                scores[doc_id]["kaynaklar"].append("dense")

        for item in sparse_results:
            doc_id = item["id"]
            if doc_id not in scores:
                scores[doc_id] = {
                    "id": doc_id,
                    "metin": item["metin"],
                    "rrf_skor": 0.0,
                    "kaynaklar": [],
                    "dense_rank": None,
                    "bm25_rank": item["rank"],
                }
            else:
                scores[doc_id]["bm25_rank"] = item["rank"]
            scores[doc_id]["rrf_skor"] += 1.0 / (rrf_k + item["rank"])
            if "bm25" not in scores[doc_id]["kaynaklar"]:
                scores[doc_id]["kaynaklar"].append("bm25")

        return sorted(scores.values(), key=lambda x: x["rrf_skor"], reverse=True)