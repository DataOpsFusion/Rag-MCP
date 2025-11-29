# Rag-MCP

A minimal RAG service with MCP tools. Documents are chunked via LangChain’s `RecursiveCharacterTextSplitter`, embedded with a HuggingFace model, and stored through an injectable `VectorStore` (Qdrant by default). All runtime knobs are centralized in `rag_core.config.RagConfig`.

## Quick start
1) Install Python deps (example):
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Run Qdrant (required for the default backend):
```bash
docker-compose up qdrant-service
```

3) Configure via environment (all optional):
- `MODEL_NAME` (default `jinaai/jina-embeddings-v2-base-code`)
- `MODEL_DEVICE` (default `cpu`)
- `QDRANT_HOST` (default `localhost`)
- `QDRANT_PORT` (default `6333`)
- `QDRANT_HTTPS` (default `false`)
- `QDRANT_API_KEY` (default `None`)
- `COLLECTION_PREFIX` (default ``)
- `CHUNK_SIZE` (default `500`)
- `CHUNK_OVERLAP` (default `50`)

4) Use the RAG service directly:
```python
from rag_core.config import get_config
from rag_core.model import Model
from rag_core.vector_store import QdrantVectorStore
from rag_core.search import RagService

cfg = get_config()
model = Model(model_name=cfg.model_name)
store = QdrantVectorStore.from_config(cfg)
rag = RagService(model=model, vector_store=store, config=cfg)

docs = [{"id": "doc1", "text": "Hello world", "metadata": {"source": "demo"}}]
rag.ingest(collection="demo", documents=docs)
hits = rag.search(collection="demo", query="hello", top_k=3, score_threshold=0.0)
print(hits)
```

## MCP tools
`mcp_server/main.py` exposes MCP tools via `FastMCP`:
- `ingest_documents(collection, documents, chunking)`
- `search(collection, query, top_k, score_threshold)`
- `get_chunk(collection, chunk_id)`
- `get_list(collection)`
- `delete(collection, doc_id)`

Each tool accepts/returns JSON-serializable payloads (see docstrings in `mcp_server/main.py`). Point your MCP-compatible host at this module to register the tools; the shared config is loaded once via `get_config()`. To swap vector backends, implement `VectorStore` and inject it in place of `QdrantVectorStore`.

## Docker / Compose
- Build the MCP image from `requirements.txt` with multi-stage Dockerfile:
```bash
docker compose build mcp
```
- Environment for the `mcp` service is user-overridable (Compose will use your shell or `.env` values):
  - `MODEL_NAME` (default `jinaai/jina-embeddings-v2-base-code`)
  - `MODEL_DEVICE` (default `cpu`)
  - `QDRANT_HOST` (default `qdrant-service`)
  - `QDRANT_PORT` (default `6333`)
  - `CHUNK_SIZE` (default `500`)
  - `CHUNK_OVERLAP` (default `50`)
- Start qdrant and run MCP attached (needed for MCP stdio):
```bash
docker compose up -d qdrant-service
docker compose run --rm mcp
```
  (Adjust env vars via `.env` or inline before the command.)
