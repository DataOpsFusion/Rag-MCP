from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict, NotRequired


class Document(TypedDict):
    text: str
    metadata: NotRequired[Dict[str, Any]]


Documents = List[Document]


class ChunkingOptions(TypedDict, total=False):
    chunk_size: int
    overlap: int


class IngestionResult(TypedDict):
    doc_ids: List[str]
    chunks_indexed: int
    errors: Optional[Dict[str, Any]]


class SearchResult(TypedDict):
    chunk_id: str
    text: str
    score: float
    doc_id: str
    metadata: Optional[Dict[str, Any]]


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
