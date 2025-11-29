from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class Document(TypedDict, total=False):
    """A single document to ingest."""

    id: Optional[str]
    text: str
    metadata: Optional[Dict[str, Any]]


Documents = List[Document]


class ChunkingOptions(TypedDict, total=False):
    """Options controlling how documents are chunked before indexing."""

    chunk_size: int
    overlap: int


class IngestionResult(TypedDict):
    """Result of an ingestion request."""

    doc_ids: List[str]
    chunks_indexed: int
    errors: Optional[Dict[str, Any]]


class SearchResult(TypedDict):
    """A single search hit."""

    chunk_id: str
    text: str
    score: float
    doc_id: str
    metadata: Optional[Dict[str, Any]]


# Convenience alias for a search response
SearchResults = List[SearchResult]


class GetChunkResult(TypedDict):
    chunk_id: str
    text: str
    doc_id: str
    metadata: Optional[Dict[str, Any]]


class GetListResult(TypedDict):
    ids: List[str]


class DeleteResult(TypedDict):
    success: bool
    errors: Optional[Dict[str, Any]]
