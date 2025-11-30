# RAG-MCP

A minimal RAG (Retrieval-Augmented Generation) service with MCP tools for document ingestion and semantic search.

Documents are chunked via LangChain's `RecursiveCharacterTextSplitter`, embedded with a HuggingFace model, and stored in Qdrant or ChromaDB.

## Features

- **5 MCP Tools**: `ingest_documents`, `search`, `get_chunk`, `get_list`, `delete`
- **Multiple Vector Stores**: Qdrant (default) or ChromaDB
- **Configurable**: All settings via environment variables
- **Docker Ready**: One-command setup with docker-compose

## Quick Start

### 1. Start Qdrant

```bash
docker compose up -d qdrant-service
```

### 2. Configure MCP Client

Add to your MCP client config (LM Studio, Claude Desktop):

```json
{
  "mcpServers": {
    "rag-mcp": {
      "command": "uvx",
      "args": ["--from", "/path/to/Rag-MCP", "rag-mcp-qdrant"],
      "env": {
        "MODEL_NAME": "jinaai/jina-embeddings-v2-base-code",
        "MODEL_DEVICE": "cpu",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333"
      }
    }
  }
}
```

Replace `/path/to/Rag-MCP` with your actual path.

### 3. Use the Tools

- **Ingest**: `ingest_documents(collection="docs", documents=[{"text": "..."}])`
- **Search**: `search(collection="docs", query="find this", top_k=5)`
- **Get Chunk**: `get_chunk(collection="docs", chunk_id="...")`
- **List Docs**: `get_list(collection="docs")`
- **Delete**: `delete(collection="docs", doc_id="...")`

## Installation

### Method 1: uvx (Recommended)

```bash
uvx --from /path/to/Rag-MCP rag-mcp-qdrant
```

First run downloads ~800MB of dependencies and may take 2-3 minutes.

**Pre-install to avoid timeout:**
```bash
uvx --from /path/to/Rag-MCP rag-mcp-qdrant --help
```

### Method 2: pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Then use this MCP config:
```json
{
  "mcpServers": {
    "rag-mcp": {
      "command": "/path/to/Rag-MCP/.venv/bin/rag-mcp-qdrant",
      "args": [],
      "env": {
        "QDRANT_HOST": "localhost"
      }
    }
  }
}
```

### Method 3: Docker

```bash
docker compose up -d qdrant-service
docker compose run --rm mcp
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `jinaai/jina-embeddings-v2-base-code` | HuggingFace embedding model |
| `MODEL_DEVICE` | `cpu` | Device: `cpu`, `cuda`, `mps` |
| `VECTOR_STORE` | `qdrant` | Backend: `qdrant` or `chroma` |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_HTTPS` | `false` | Use HTTPS |
| `QDRANT_API_KEY` | `None` | API key for Qdrant Cloud |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma_data` | ChromaDB storage path |
| `COLLECTION_PREFIX` | `` | Prefix for collection names |
| `CHUNK_SIZE` | `500` | Text chunk size |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |

## MCP Client Configuration

### LM Studio

File: Settings → MCP Servers

```json
{
  "mcpServers": {
    "rag-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/Rag-MCP", "rag-mcp-qdrant"],
      "env": {
        "MODEL_DEVICE": "cpu",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333"
      }
    }
  }
}
```

### Claude Desktop (macOS)

File: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "rag-mcp": {
      "command": "uvx",
      "args": ["--from", "/path/to/Rag-MCP", "rag-mcp-qdrant"],
      "env": {
        "MODEL_DEVICE": "cpu",
        "QDRANT_HOST": "localhost"
      }
    }
  }
}
```

### Claude Desktop (Windows)

File: `%APPDATA%\Claude\claude_desktop_config.json`

Same configuration as macOS.

## Usage Examples

### Python API

```python
from rag_core.config import get_config
from rag_core.model import Model
from rag_core.vector_store import QdrantVectorStore
from rag_core.search import RagService

cfg = get_config()
model = Model(model_name=cfg.model_name)
store = QdrantVectorStore.from_config(cfg)
rag = RagService(model=model, vector_store=store, config=cfg)

docs = [{"text": "Hello world", "metadata": {"source": "demo"}}]
rag.ingest(collection="demo", documents=docs)

hits = rag.search(collection="demo", query="hello", top_k=3)
print(hits)
```

### MCP Tool Calls

```python
ingest_documents(
    collection="knowledge",
    documents=[{"text": "AI is transforming industries."}],
    chunking={"chunk_size": 500, "overlap": 50}
)

search(
    collection="knowledge",
    query="AI transformation",
    top_k=5,
    score_threshold=0.5
)
```

## Troubleshooting

### Connection Refused
```bash
docker compose up -d qdrant-service
curl http://localhost:6333/health
```

### Model Download Slow
First run downloads the embedding model from HuggingFace (~400MB).

### GPU Support
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
# Set MODEL_DEVICE=cuda in config
```

## License

MIT
