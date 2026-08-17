# tapestry

Hybrid agentic RAG pipeline for Obsidian knowledge vaults. Weaves dense vectors, sparse SPLADE, and a wikilink graph into a single retrieval pipeline — served by a Go API and exposed as an MCP server.

## Architecture

```
Obsidian vault (.md files)
        │
        ▼
┌──────────────────────┐
│   fde-ingest         │  Python · :7910
│                      │
│  watchdog file mon   │
│  tiktoken chunker    │
│  nomic-embed dense   │  → Qdrant  obsidian_hybrid
│  SPLADE sparse       │     dense(768) + sparse
│  ms-marco reranker   │
│  wikilink graph      │  → Neo4j   fde-rag database
│                      │
│  /embed              │  ← called by fde-rag at query time
│  /embed-sparse       │
│  /rerank             │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│   fde-rag            │  Go · :7900
│                      │
│  Qdrant gRPC         │
│  Prefetch dense+     │
│  Prefetch sparse →   │
│  FusionQuery(RRF)    │
│  Neo4j 1-hop expand  │
│  cross-encoder rerank│
│  Qwen LLM answer     │
│                      │
│  GET  /health        │
│  POST /search        │
│  POST /ask           │
│  POST /mcp           │  ← MCP streamable-http
└──────────────────────┘
```

## Services

| Service | Lang | Port | Role |
|---|---|---|---|
| `fde-ingest` | Python | :7910 | Embed API + vault watcher + Qdrant indexer + Neo4j graph |
| `fde-rag` | Go | :7900 | Hybrid search + RRF + rerank + LLM answer + MCP server |

## Prerequisites

- Docker + Docker Compose (Portainer optional)
- Qdrant ≥ 1.12.0 running on the target host
- Neo4j Enterprise 5.x with a `fde-rag` database
- An OpenAI-compatible LLM endpoint (Qwen, DeepSeek, etc.)

## Deploy

```bash
# Create secrets file (never commit this)
cat > .env <<EOF
NEO4J_PASSWORD=<your-neo4j-password>
SPARK_VLLM_API_KEY=local-dummy
EOF

# Edit compose.yml to set QWEN_URL and vault path, then:
docker compose build
docker compose up -d
```

Or via the deploy script (from the source machine):

```bash
# Requires ~/.fde-secrets.env with NEO4J_PASSWORD and SPARK_VLLM_API_KEY
bash deploy-to-spark2.sh
```

## MCP integration

Add to your agent config (dsh / pi):

```yaml
- id: mcp-tapestry
  name: '@deepseek-ai/dsh-mcp-client'
  config:
    serverName: tapestry
    transport: streamable-http
    url: http://<host>:7900/mcp
```

### Available MCP tools

| Tool | Description |
|---|---|
| `search_vault` | Hybrid semantic+sparse search over the vault |
| `ask_vault` | Ask a question, answered by RAG + LLM |
| `vault_status` | Returns indexed chunk count |

## Search pipeline detail

1. Query → fde-ingest `/embed` (768-dim nomic-embed-text-v1.5)
2. Query → fde-ingest `/embed-sparse` (SPLADE PP en v1)
3. Qdrant gRPC `QueryPoints` with two `Prefetch` legs → server-side `FusionQuery(RRF)`
4. Neo4j 1-hop wikilink expansion (graph context)
5. fde-ingest `/rerank` (ms-marco-MiniLM-L-6-v2 cross-encoder)
6. Top-k hits → Qwen LLM for `/ask` responses

## Running tests

```bash
cd fde-rag
RAG_URL=http://localhost:7900 EMBED_URL=http://localhost:7910 \
  go test -v -timeout 120s ./cmd/fde-rag/ -run Test
```
