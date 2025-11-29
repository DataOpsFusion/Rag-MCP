"""
Pluggable vector-store interfaces and a Qdrant implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple
from typing_extensions import TypedDict

from .config import RagConfig

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
except ImportError:  # pragma: no cover - optional dependency until Qdrant is used
    QdrantClient = None
    rest = None

logger = logging.getLogger(__name__)


class VectorHit(TypedDict, total=False):
    id: str
    score: float
    payload: Dict[str, Any]


class VectorStore(Protocol):
    """Protocol for vector database backends."""

    def ensure_collection(self, name: str, dim: int) -> None:
        ...

    def upsert(self, collection: str, points: List[Dict[str, Any]]) -> None:
        ...

    def search(self, collection: str, vector: List[float], limit: int) -> List[VectorHit]:
        ...

    def retrieve(self, collection: str, ids: List[str]) -> List[VectorHit]:
        ...

    def list_doc_ids(self, collection: str, *, limit: int = 10_000) -> List[str]:
        ...

    def delete_doc(self, collection: str, doc_id: str) -> Tuple[bool, Optional[str]]:
        ...


class QdrantVectorStore:
    """Qdrant-backed VectorStore implementation."""

    def __init__(
        self,
        client: Any = None,
        *,
        host: str = "localhost",
        port: int = 6333,
        https: bool = False,
        api_key: Optional[str] = None,
        collection_prefix: str = "",
    ):
        if client is None:
            if QdrantClient is None or rest is None:  # pragma: no cover
                raise RuntimeError("Install qdrant-client to use QdrantVectorStore")
            client = QdrantClient(host=host, port=port, https=https, api_key=api_key)

        self._client = client
        self._collection_prefix = collection_prefix

    @classmethod
    def from_config(cls, config: RagConfig) -> "QdrantVectorStore":
        return cls(
            host=config.qdrant_host,
            port=config.qdrant_port,
            https=config.qdrant_https,
            api_key=config.qdrant_api_key,
            collection_prefix=config.collection_prefix,
        )

    def _full(self, name: str) -> str:
        return f"{self._collection_prefix}{name}"

    def ensure_collection(self, name: str, dim: int) -> None:
        collection_name = self._full(name)
        try:
            self._client.get_collection(collection_name=collection_name)
            return
        except Exception:
            pass

        if rest is None:  # pragma: no cover
            raise RuntimeError("qdrant-client models not available")

        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE),
        )

    def upsert(self, collection: str, points: List[Dict[str, Any]]) -> None:
        collection_name = self._full(collection)
        payloads: List[rest.PointStruct] = []
        for point in points:
            payloads.append(
                rest.PointStruct(
                    id=point["id"],
                    vector=point["vector"],
                    payload=point["payload"],
                )
            )
        self._client.upsert(collection_name=collection_name, points=payloads)

    def search(self, collection: str, vector: List[float], limit: int) -> List[VectorHit]:
        collection_name = self._full(collection)
        results = self._client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
            with_payload=True,
        )
        hits: List[VectorHit] = []
        for hit in results:
            hits.append(
                {
                    "id": str(hit.id),
                    "score": float(hit.score),
                    "payload": hit.payload or {},
                }
            )
        return hits

    def retrieve(self, collection: str, ids: List[str]) -> List[VectorHit]:
        collection_name = self._full(collection)
        records = self._client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_payload=True,
        )
        hits: List[VectorHit] = []
        for record in records:
            hits.append(
                {
                    "id": str(record.id),
                    "score": 1.0,
                    "payload": record.payload or {},
                }
            )
        return hits

    def list_doc_ids(self, collection: str, *, limit: int = 10_000) -> List[str]:
        collection_name = self._full(collection)
        ids: List[str] = []
        next_offset: Optional[int] = None

        while True:
            scroll = self._client.scroll(
                collection_name=collection_name,
                limit=256,
                with_payload=True,
                offset=next_offset,
            )
            for point in scroll.points:
                payload = point.payload or {}
                doc_id = payload.get("doc_id")
                if doc_id and doc_id not in ids:
                    ids.append(doc_id)
                    if len(ids) >= limit:
                        return ids
            next_offset = scroll.next_page_offset
            if next_offset is None:
                break
        return ids

    def delete_doc(self, collection: str, doc_id: str) -> Tuple[bool, Optional[str]]:
        if rest is None:  # pragma: no cover
            raise RuntimeError("qdrant-client models not available")

        collection_name = self._full(collection)
        try:
            filter_payload = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="doc_id",
                        match=rest.MatchValue(value=doc_id),
                    )
                ]
            )
            self._client.delete(
                collection_name=collection_name,
                points_selector=rest.FilterSelector(filter=filter_payload),
            )
            return True, None
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to delete doc %s from %s", doc_id, collection_name)
            return False, str(exc)
