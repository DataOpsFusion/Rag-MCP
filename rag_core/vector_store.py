from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple
from typing_extensions import TypedDict

from .config import RagConfig

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
except ImportError:
    QdrantClient = None
    rest = None

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None
    ChromaSettings = None

logger = logging.getLogger(__name__)


class VectorHit(TypedDict, total=False):
    id: str
    score: float
    payload: Dict[str, Any]


class VectorStore(Protocol):

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
            if QdrantClient is None or rest is None:
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

        if rest is None:
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
        results = self._client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        hits: List[VectorHit] = []
        for hit in results.points:
            hits.append(
                {
                    "id": str(hit.id),
                    "score": float(hit.score),
                    "payload": hit.payload or {},
                }
            )
        return hits

    def retrieve(self, collection: str, ids: List[str]) -> List[VectorHit]:
        if rest is None:
            raise RuntimeError("qdrant-client models not available")
        
        collection_name = self._full(collection)
        hits: List[VectorHit] = []
        
        for chunk_id in ids:
            points, _ = self._client.scroll(
                collection_name=collection_name,
                scroll_filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="chunk_id",
                            match=rest.MatchValue(value=chunk_id),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
            )
            for point in points:
                hits.append(
                    {
                        "id": str(point.id),
                        "score": 1.0,
                        "payload": point.payload or {},
                    }
                )
        return hits

    def list_doc_ids(self, collection: str, *, limit: int = 10_000) -> List[str]:
        collection_name = self._full(collection)
        ids: List[str] = []
        next_offset = None

        while True:
            points, next_offset = self._client.scroll(
                collection_name=collection_name,
                limit=256,
                with_payload=True,
                offset=next_offset,
            )
            for point in points:
                payload = point.payload or {}
                doc_id = payload.get("doc_id")
                if doc_id and doc_id not in ids:
                    ids.append(doc_id)
                    if len(ids) >= limit:
                        return ids
            if next_offset is None:
                break
        return ids

    def delete_doc(self, collection: str, doc_id: str) -> Tuple[bool, Optional[str]]:
        if rest is None:
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
        except Exception as exc:
            logger.exception("Failed to delete doc %s from %s", doc_id, collection_name)
            return False, str(exc)


class ChromaVectorStore:

    def __init__(
        self,
        client: Any = None,
        *,
        persist_directory: Optional[str] = None,
        collection_prefix: str = "",
    ):
        if client is None:
            if chromadb is None:
                raise RuntimeError("Install chromadb to use ChromaVectorStore: pip install chromadb")
            if persist_directory:
                client = chromadb.PersistentClient(path=persist_directory)
            else:
                client = chromadb.Client()

        self._client = client
        self._collection_prefix = collection_prefix
        self._collections: Dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: RagConfig) -> "ChromaVectorStore":
        persist_dir = getattr(config, "chroma_persist_directory", "./chroma_data")
        return cls(
            persist_directory=persist_dir,
            collection_prefix=config.collection_prefix,
        )

    def _full(self, name: str) -> str:
        return f"{self._collection_prefix}{name}"

    def ensure_collection(self, name: str, dim: int) -> None:
        collection_name = self._full(name)
        if collection_name not in self._collections:
            self._collections[collection_name] = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def upsert(self, collection: str, points: List[Dict[str, Any]]) -> None:
        collection_name = self._full(collection)
        coll = self._collections.get(collection_name)
        if coll is None:
            coll = self._client.get_collection(collection_name)
            self._collections[collection_name] = coll

        ids = [str(point["id"]) for point in points]
        embeddings = [point["vector"] for point in points]
        metadatas = [point["payload"] for point in points]

        coll.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def search(self, collection: str, vector: List[float], limit: int) -> List[VectorHit]:
        collection_name = self._full(collection)
        coll = self._collections.get(collection_name)
        if coll is None:
            coll = self._client.get_collection(collection_name)
            self._collections[collection_name] = coll

        results = coll.query(query_embeddings=[vector], n_results=limit)

        hits: List[VectorHit] = []
        if results["ids"] and len(results["ids"]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                score = 1.0 - distance
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                hits.append(
                    {
                        "id": str(doc_id),
                        "score": float(score),
                        "payload": metadata or {},
                    }
                )
        return hits

    def retrieve(self, collection: str, ids: List[str]) -> List[VectorHit]:
        collection_name = self._full(collection)
        coll = self._collections.get(collection_name)
        if coll is None:
            coll = self._client.get_collection(collection_name)
            self._collections[collection_name] = coll

        results = coll.get(ids=ids, include=["metadatas"])

        hits: List[VectorHit] = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                hits.append(
                    {
                        "id": str(doc_id),
                        "score": 1.0,
                        "payload": metadata or {},
                    }
                )
        return hits

    def list_doc_ids(self, collection: str, *, limit: int = 10_000) -> List[str]:
        collection_name = self._full(collection)
        coll = self._collections.get(collection_name)
        if coll is None:
            coll = self._client.get_collection(collection_name)
            self._collections[collection_name] = coll

        results = coll.get(limit=limit, include=["metadatas"])

        doc_ids: List[str] = []
        seen = set()
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                doc_id = metadata.get("doc_id")
                if doc_id and doc_id not in seen:
                    doc_ids.append(doc_id)
                    seen.add(doc_id)
        return doc_ids

    def delete_doc(self, collection: str, doc_id: str) -> Tuple[bool, Optional[str]]:
        collection_name = self._full(collection)
        coll = self._collections.get(collection_name)
        if coll is None:
            coll = self._client.get_collection(collection_name)
            self._collections[collection_name] = coll

        try:
            results = coll.get(where={"doc_id": doc_id}, include=[])
            if results["ids"]:
                coll.delete(ids=results["ids"])
            return True, None
        except Exception as exc:
            logger.exception("Failed to delete doc %s from %s", doc_id, collection_name)
            return False, str(exc)
