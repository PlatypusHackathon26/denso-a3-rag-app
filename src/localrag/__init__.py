"""localrag — RAG chạy hoàn toàn cục bộ với Ollama (Qwen) và LanceDB."""

from .config import Settings, load_settings
from .models import RAGAnswer, SearchResult
from .pipeline import RAGPipeline

__all__ = ["RAGPipeline", "RAGAnswer", "SearchResult", "Settings", "load_settings"]
__version__ = "0.2.0"
