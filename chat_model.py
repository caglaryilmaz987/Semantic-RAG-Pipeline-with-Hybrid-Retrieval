"""
chat_model.py — LLM Client Wrapper (Foundry Local SDK)
Loads and manages the local Phi-3.5-mini language model.
"""

from foundry_manager import get_manager


class ChatModel:
    """
    LLM Chat Completion Wrapper for Phi-3.5-mini.
    Provides synchronous completion and real-time streaming interfaces.
    """
    MODEL_NAME = "phi-3.5-mini"

    def __init__(self):
        manager = get_manager()
        self._model = manager.catalog.get_model(self.MODEL_NAME)
        self._model.download()
        self._model.load()
        self._client = self._model.get_chat_client()

    def complete(self, messages: list[dict]) -> str:
        """
        Executes a non-streaming chat completion request.
        """
        response = self._client.complete_chat(messages)
        return response.choices[0].message.content

    def complete_stream(self, messages: list[dict]):
        """
        Yields token-by-token streaming chat completions in real time.
        """
        if hasattr(self._client, "complete_streaming_chat"):
            for chunk in self._client.complete_streaming_chat(messages):
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        yield delta.content
        else:
            # Fallback to standard completion if streaming is unsupported
            yield self.complete(messages)

    def unload(self):
        """Unloads the LLM model from memory."""
        self._model.unload()