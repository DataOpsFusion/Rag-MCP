"""
RAG ingest/search helper that handles chunking, embedding, and vector store interactions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

try:
    # Newer LangChain splits out text splitters
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("langchain and langchain-text-splitters are required for chunking") from exc

from .config import RagConfig, get_config
from .embedding import Embedding
from .model import Model
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RagService:
    """Coordinates chunking, embedding, and vector search against an injected backend."""

    def __init__(self, model: Model, vector_store: VectorStore, config: Optional[RagConfig] = None):
        self._model = model
        self._embedder = Embedding(model)
        self._store = vector_store
        self._config = config or get_config()

    def _split_texts(
        self, texts: Iterable[str], *, chunk_size: int, overlap: int
    ) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        chunks: List[str] = []
        for text in texts:
            chunks.extend(splitter.split_text(text))
        return chunks

    def ingest(
        self,
        collection: str,
        documents: Sequence[Dict[str, Any]],
        *,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Chunk, embed, and upsert a batch of documents."""

        # Ensure backend is ready for this embedding dimension
        self._model._inspect()
        if self._model.dim is None:
            raise RuntimeError("Embedding dimension is unknown; failed to inspect model")
        self._store.ensure_collection(collection, dim=self._model.dim)

        chunk_size = chunk_size or self._config.default_chunk_size
        overlap = overlap or self._config.default_chunk_overlap

        doc_ids: List[str] = []
        chunks_indexed = 0
        errors: Dict[str, str] = {}
        points: List[Dict[str, Any]] = []

        for doc in documents:
            raw_text = doc.get("text") or ""
            if not raw_text.strip():
                logger.warning("Skipping empty document payload")
                continue

            doc_id = doc.get("id") or str(uuid4())
            metadata = doc.get("metadata") or {}

            try:
                chunk_texts = self._split_texts([raw_text], chunk_size=chunk_size, overlap=overlap)
                for idx, chunk_text in enumerate(chunk_texts):
                    chunk_id = f"{doc_id}:{idx}"
                    vector = self._embedder.embed_text(chunk_text)
                    points.append(
                        {
                            "id": chunk_id,
                            "vector": vector,
                            "payload": {
                                "chunk_id": chunk_id,
                                "doc_id": doc_id,
                                "text": chunk_text,
                                "metadata": metadata,
                                "chunk_index": idx,
                            },
                        }
                    )
                doc_ids.append(doc_id)
                chunks_indexed += len(chunk_texts)
            except Exception as exc:  # pragma: no cover - runtime errors surfaced in response
                logger.exception("Failed to ingest doc %s", doc_id)
                errors[doc_id] = str(exc)

        if points:
            self._store.upsert(collection=collection, points=points)

        return {
            "doc_ids": doc_ids,
            "chunks_indexed": chunks_indexed,
            "errors": errors or None,
        }

    def search(
        self,
        collection: str,
        query: str,
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        self._model._inspect()
        if self._model.dim is None:
            raise RuntimeError("Embedding dimension is unknown; failed to inspect model")
        self._store.ensure_collection(collection, dim=self._model.dim)

        vector = self._embedder.embed_query(query)
        results = self._store.search(collection=collection, vector=vector, limit=top_k)

        hits: List[Dict[str, Any]] = []
        for hit in results:
            if score_threshold and hit.get("score", 0) < score_threshold:
                continue
            payload = hit.get("payload", {}) or {}
            hits.append(
                {
                    "chunk_id": payload.get("chunk_id") or str(hit.get("id")),
                    "text": payload.get("text", ""),
                    "score": float(hit.get("score", 0.0)),
                    "doc_id": payload.get("doc_id", ""),
                    "metadata": payload.get("metadata"),
                }
            )
        return hits

    def get_chunk(self, collection: str, chunk_id: str) -> Dict[str, Any]:
        records = self._store.retrieve(collection=collection, ids=[chunk_id])
        if not records:
            raise KeyError(f"Chunk {chunk_id} not found in {collection}")

        payload = records[0].get("payload", {}) or {}
        return {
            "chunk_id": payload.get("chunk_id") or chunk_id,
            "text": payload.get("text", ""),
            "doc_id": payload.get("doc_id", ""),
            "metadata": payload.get("metadata"),
        }

    def list_doc_ids(self, collection: str, *, limit: int = 10_000) -> List[str]:
        return self._store.list_doc_ids(collection=collection, limit=limit)

    def delete_doc(self, collection: str, doc_id: str) -> Tuple[bool, Optional[str]]:
        return self._store.delete_doc(collection=collection, doc_id=doc_id)
