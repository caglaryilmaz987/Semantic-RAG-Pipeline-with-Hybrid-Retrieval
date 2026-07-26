"""
reranker.py — Cross-Encoder Reranker (GPU / CUDA Accelerated)
Evaluates query-document pairs post-RRF fusion for high-precision reranking.

Model: BAAI/bge-reranker-v2-m3
"""

import os
import warnings

# Suppress HuggingFace / Transformers warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

# Optimize PyTorch CPU thread allocation if running on CPU
num_cores = os.cpu_count() or 4
torch.set_num_threads(num_cores)

from sentence_transformers import CrossEncoder


class Reranker:
    """
    Multilingual Cross-Encoder Reranker.
    Uses CUDA (NVIDIA GPU) if available for sub-50ms inference.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        # Automatically assign PyTorch execution to CUDA GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, belgeler: list[dict], top_k: int = 3) -> list[dict]:
        """
        Rerank a list of candidate document dicts for a given search query.
        
        Args:
            query (str): The search query.
            belgeler (list[dict]): Candidate document dictionaries containing "metin".
            top_k (int): Number of top reranked documents to return.

        Returns:
            list[dict]: Sorted candidate document dicts updated with "rerank_skor".
        """
        if not belgeler:
            return []

        # Predict relevance scores for top candidates
        candidate_docs = belgeler[: min(len(belgeler), 8)]
        pairs = [(query, b["metin"]) for b in candidate_docs]

        scores = self._model.predict(pairs, batch_size=8, show_progress_bar=False)

        for belge, skor in zip(candidate_docs, scores):
            belge["rerank_skor"] = float(skor)

        # Assign default scores for remaining candidates
        for belge in belgeler[len(candidate_docs):]:
            belge["rerank_skor"] = -10.0

        belgeler.sort(key=lambda x: x["rerank_skor"], reverse=True)
        return belgeler[:top_k]