from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Mapping, Optional, Sequence, Type

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
except ImportError:
    QdrantClient = None
    rest = None


class BaseStorage(ABC):
    """Base interface every storage backend must implement."""

    @abstractmethod
    def save(self, key: str, value: str) -> None:
        """Persist the provided value under the given key."""

    @abstractmethod
    def load(self, key: str) -> str:
        """Return the value previously stored for ``key`` or raise ``KeyError`` if missing."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove an existing value for ``key``. Should not propagate if the key is already absent."""


class InMemoryStorage(BaseStorage):
    """Lightweight dictionary-backed implementation used for testing or as a default backend."""

    def __init__(self, initial_data: Optional[Mapping[str, str]] = None):
        self._store: Dict[str, str] = dict(initial_data or {})

    def save(self, key: str, value: str) -> None:
        self._store[key] = value

    def load(self, key: str) -> str:
        try:
            return self._store[key]
        except KeyError as exc:
            raise KeyError(f"{key} not found in in-memory storage") from exc

    def delete(self, key: str) -> None:
        self._store.pop(key, None)


class QdrantStorage(BaseStorage):
    """Example qdrant-backed storage. Stores the value as payload and keeps a lightweight vector."""

    def __init__(
        self,
        client: Any,
        collection_name: str,
        vectorizer: Optional[Callable[[str], Sequence[float]]] = None,
    ):
        if QdrantClient is None or rest is None:  # pragma: no cover
            raise RuntimeError("Install qdrant-client before using QdrantStorage")

        self._client = client
        self._collection_name = collection_name
        self._vectorizer = vectorizer or (lambda _: [0.0])

    def _to_vector(self, value: str) -> List[float]:
        vector = list(self._vectorizer(value))
        if not vector:
            raise ValueError("Vectorizer must return at least one dimension")
        return vector

    def save(self, key: str, value: str) -> None:
        if rest is None:  # pragma: no cover
            raise RuntimeError("qdrant-client is unavailable")

        point = rest.PointStruct(
            id=key,
            vector=self._to_vector(value),
            payload={"key": key, "value": value},
        )
        self._client.upsert(collection_name=self._collection_name, points=[point])

    def load(self, key: str) -> str:
        if rest is None:  # pragma: no cover
            raise RuntimeError("qdrant-client is unavailable")

        filter_payload = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="key",
                    match=rest.MatchValue(value=key),
                )
            ]
        )
        scroll = self._client.scroll(
            collection_name=self._collection_name,
            filter=filter_payload,
            limit=1,
        )
        if not scroll.points:
            raise KeyError(f"{key} not found in qdrant collection {self._collection_name}")

        payload = scroll.points[0].payload or {}
        return str(payload.get("value", ""))

    def delete(self, key: str) -> None:
        self._client.delete(collection_name=self._collection_name, points=[key])


class StorageRegistry:
    """Registry that keeps track of available storage backends and instantiates them."""

    _backends: ClassVar[Dict[str, Type[BaseStorage]]] = {}

    @classmethod
    def register_backend(cls, name: str, backend: Type[BaseStorage]) -> None:
        cls._backends[name] = backend

    @classmethod
    def get_backend(cls, name: str) -> Type[BaseStorage]:
        try:
            return cls._backends[name]
        except KeyError as exc:
            raise ValueError(f"Unknown storage backend '{name}'") from exc

    @classmethod
    def create(cls, name: str, **options: Any) -> BaseStorage:
        backend = cls.get_backend(name)
        return backend(**options)


@dataclass
class StorageConfig:
    """Configuration that describes which storage implementation should be injected."""

    backend: str = "in_memory"
    options: Dict[str, Any] = field(default_factory=dict)

    def build(self) -> BaseStorage:
        """Construct the configured backend with the provided options."""

        return StorageRegistry.create(self.backend, **self.options)


class StorageProvider:
    """Simple dependency provider that lazy-instantiates a storage backend."""

    def __init__(self, config: StorageConfig):
        self._config = config
        self._instance: Optional[BaseStorage] = None

    def get(self) -> BaseStorage:
        if self._instance is None:
            self._instance = self._config.build()
        return self._instance


# Register default backends
StorageRegistry.register_backend("in_memory", InMemoryStorage)
StorageRegistry.register_backend("qdrant", QdrantStorage)