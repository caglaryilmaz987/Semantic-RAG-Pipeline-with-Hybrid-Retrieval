"""
foundry_manager.py — Foundry Local Singleton Manager
Manages a single global instance of FoundryLocalManager throughout the application lifecycle.

Key Features:
  - Thread-safe double-checked locking mechanism using threading.Lock.
  - Safe exception handling for singleton re-initialization.
"""

import threading
from foundry_local_sdk import Configuration, FoundryLocalManager

_initialized = False
_lock = threading.Lock()


def get_manager(app_name: str = "SemanticRAG") -> FoundryLocalManager:
    """
    Returns the singleton instance of FoundryLocalManager.
    
    Args:
        app_name (str): Name of the application registered in Foundry Local SDK.
        
    Returns:
        FoundryLocalManager: Active singleton instance.
    """
    global _initialized
    if not _initialized:
        with _lock:
            if not _initialized:
                try:
                    config = Configuration(app_name=app_name)
                    FoundryLocalManager.initialize(config)
                except Exception:
                    # If already initialized (Singleton re-init in Streamlit), reuse instance silently
                    pass
                _initialized = True
    return FoundryLocalManager.instance