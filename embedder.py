"""
embedder.py — Embedding Client Wrapper
Loads and runs the Qwen3-Embedding-0.6b model via Microsoft Foundry Local SDK.
"""

from foundry_manager import get_manager


class Embedder:
    """
    Embedding Client Wrapper.
    Uses Foundry Local SDK to load Qwen3-Embedding-0.6b and generate dense vector embeddings.
    """
    MODEL_NAME = "qwen3-embedding-0.6b"

    def __init__(self):
        manager = get_manager()
        self._model = manager.catalog.get_model(self.MODEL_NAME)
        self._model.download()
        self._model.load()
        self._client = self._model.get_embedding_client()
        self._dimension: int | None = None

    def embed(self, text: str) -> list[float]:
        """
        Generates a 1024-dimensional dense embedding vector for a single text string.
        """
        response = self._client.generate_embedding(text)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generates embedding vectors for a batch of text strings.
        """
        return [self.embed(t) for t in texts]

    @property
    def client(self):
        """Returns the underlying embedding client instance."""
        return self._client

    @property
    def dimension(self) -> int:
        """
        Dynamically detects and caches the output embedding dimension (e.g., 1024).
        """
        if self._dimension is None:
            sample_emb = self.embed("dimension_check")
            self._dimension = len(sample_emb)
        return self._dimension

    def unload(self):
        """Unloads the embedding model from memory."""
        self._model.unload()