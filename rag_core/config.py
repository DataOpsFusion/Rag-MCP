from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass
class RagConfig:

    model_name: str = "jinaai/jina-embeddings-v2-base-code"
    model_device: str = "cpu"
    vector_store: str = "qdrant"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_https: bool = False
    qdrant_api_key: Optional[str] = None
    chroma_persist_directory: str = "./chroma_data"
    collection_prefix: str = ""
    default_chunk_size: int = 500
    default_chunk_overlap: int = 50

    @classmethod
    def from_env(cls) -> "RagConfig":
        defaults = cls()

        def getenv_int(name: str, fallback: int) -> int:
            try:
                return int(os.getenv(name, fallback))
            except (TypeError, ValueError):
                return fallback

        def getenv_bool(name: str, fallback: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return fallback
            return raw.lower() in {"1", "true", "yes", "on"}

        return cls(
            model_name=os.getenv("MODEL_NAME", defaults.model_name),
            model_device=os.getenv("MODEL_DEVICE", defaults.model_device),
            vector_store=os.getenv("VECTOR_STORE", defaults.vector_store),
            qdrant_host=os.getenv("QDRANT_HOST", defaults.qdrant_host),
            qdrant_port=getenv_int("QDRANT_PORT", defaults.qdrant_port),
            qdrant_https=getenv_bool("QDRANT_HTTPS", defaults.qdrant_https),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", defaults.qdrant_api_key),
            chroma_persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", defaults.chroma_persist_directory),
            collection_prefix=os.getenv("COLLECTION_PREFIX", defaults.collection_prefix),
            default_chunk_size=getenv_int("CHUNK_SIZE", defaults.default_chunk_size),
            default_chunk_overlap=getenv_int("CHUNK_OVERLAP", defaults.default_chunk_overlap),
        )


@lru_cache(maxsize=1)
def get_config() -> RagConfig:
    return RagConfig.from_env()
