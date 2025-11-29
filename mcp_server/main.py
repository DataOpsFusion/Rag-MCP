from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from rag_core.config import RagConfig, get_config
from rag_core.model import Model
from rag_core.search import RagService
from rag_core.vector_store import QdrantVectorStore

from .models import (
    ChunkingOptions,
    DeleteResult,
    Documents,
    GetChunkResult,
    GetListResult,
    IngestionResult,
    SearchResults,
)

mcp = FastMCP("RAG MCP")

_config: RagConfig = get_config()
_model = Model(model_name=_config.model_name, device=_config.model_device)
_vector_store = QdrantVectorStore.from_config(_config)
_rag = RagService(model=_model, vector_store=_vector_store, config=_config)


@mcp.tool(
    description="Ingest and index documents into a collection using LangChain chunking and embeddings.",
)
def ingest_documents(
    collection: Annotated[str, Field(description="Target collection name.")],
    documents: Annotated[Documents, Field(description="List of documents: {id?, text, metadata?}.")],
    chunking: Annotated[
        ChunkingOptions,
        Field(description="Chunking options: {chunk_size?, overlap?}; falls back to config defaults."),
    ],
) -> IngestionResult:
    chunk_size = (chunking or {}).get("chunk_size") or _config.default_chunk_size
    overlap = (chunking or {}).get("overlap") or _config.default_chunk_overlap
    return _rag.ingest(
        collection=collection,
        documents=documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )


@mcp.tool(
    description="Semantic search across indexed chunks.",
)
def search(
    collection: Annotated[str, Field(description="Target collection name.")],
    query: Annotated[str, Field(description="Query text to embed and search for.")],
    top_k: Annotated[int, Field(description="Maximum number of hits to return.")] = 5,
    score_threshold: Annotated[float, Field(description="Minimum similarity score to include.")] = 0.0,
) -> SearchResults:
    return _rag.search(
        collection=collection,
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
    )


@mcp.tool(
    description="Fetch a specific chunk by id.",
)
def get_chunk(
    collection: Annotated[str, Field(description="Target collection name.")],
    chunk_id: Annotated[str, Field(description="Chunk identifier returned from search/ingest.")],
) -> GetChunkResult:
    return _rag.get_chunk(collection=collection, chunk_id=chunk_id)


@mcp.tool(
    description="List unique document ids stored in the collection.",
)
def get_list(
    collection: Annotated[str, Field(description="Target collection name.")],
) -> GetListResult:
    return {"ids": _rag.list_doc_ids(collection=collection)}


@mcp.tool(
    description="Delete all chunks for a given document id.",
)
def delete(
    collection: Annotated[str, Field(description="Target collection name.")],
    doc_id: Annotated[str, Field(description="Document identifier whose chunks should be removed.")],
) -> DeleteResult:
    success, error = _rag.delete_doc(collection=collection, doc_id=doc_id)
    return {"success": success, "errors": {"message": error} if error else None}


if __name__ == "__main__":
    mcp.run()
