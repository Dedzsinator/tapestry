"""
fde-ingest: Obsidian vault indexer + embedding API.

Dual role:
  1. Background watchdog: chunk .md files → Qdrant hybrid collection + Neo4j graph.
  2. HTTP API on $EMBED_PORT (7910): /embed, /embed-sparse, /rerank — called by fde-rag (Go).
"""
import os, re, time, threading, asyncio
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from fastembed import TextEmbedding, SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, SparseVectorParams, SparseIndexParams,
    SparseVector, PointStruct, NamedSparseVector, NamedVector,
)
from neo4j import GraphDatabase
import tiktoken
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

VAULT_PATH   = os.environ["VAULT_PATH"]
QDRANT_URL   = os.environ.get("QDRANT_URL", "http://localhost:6333")
NEO4J_URI    = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER   = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS   = os.environ.get("NEO4J_PASSWORD", "")
EMBED_PORT   = int(os.environ.get("EMBED_PORT", "7910"))
COLLECTION   = "obsidian_hybrid"
CHUNK_TOKENS = 400
CHUNK_OVERLAP = 50

# ── models (loaded once at startup) ──────────────────────────────────────────
dense_model   = TextEmbedding("nomic-ai/nomic-embed-text-v1.5")
sparse_model  = SparseTextEmbedding("prithivida/Splade_PP_en_v1")
rerank_model  = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
enc           = tiktoken.get_encoding("cl100k_base")
qdrant        = QdrantClient(url=QDRANT_URL, timeout=30)
neo4j_driver  = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ── Qdrant collection setup ───────────────────────────────────────────────────
def ensure_collection():
    names = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in names:
        qdrant.create_collection(
            COLLECTION,
            vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
        )

# ── helpers ───────────────────────────────────────────────────────────────────
def _embed_dense(text: str) -> list[float]:
    return next(dense_model.embed([text])).tolist()

def _embed_sparse(text: str) -> SparseVector:
    sv = next(sparse_model.embed([text]))
    return SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())

def chunk_text(text: str) -> list[str]:
    toks = enc.encode(text)
    step = CHUNK_TOKENS - CHUNK_OVERLAP
    chunks = []
    for i in range(0, len(toks), step):
        chunks.append(enc.decode(toks[i : i + CHUNK_TOKENS]))
    return chunks or [text[:2000]]

def wikilinks(text: str) -> list[str]:
    return re.findall(r"\[\[([^\]|#]+)", text)

def stable_id(path: str, chunk: int) -> int:
    return abs(hash(f"{path}:{chunk}")) % (2**63)

# ── indexing ──────────────────────────────────────────────────────────────────
def index_file(p: Path):
    rel  = str(p.relative_to(VAULT_PATH))
    text = p.read_text(errors="replace")
    chunks = chunk_text(text)
    points = []
    for i, chunk in enumerate(chunks):
        dv = _embed_dense(chunk)
        sv = _embed_sparse(chunk)
        points.append(PointStruct(
            id=stable_id(rel, i),
            vector={"dense": dv, "sparse": sv},
            payload={"path": rel, "chunk": i, "text": chunk},
        ))
    qdrant.upsert(collection_name=COLLECTION, points=points, wait=False)

    links = wikilinks(text)
    with neo4j_driver.session(database="fde-rag") as s:
        s.run("MERGE (n:Note {path:$p})", p=rel)
        for lnk in links:
            s.run(
                "MERGE (n:Note {path:$p}) "
                "MERGE (m:Note {path:$m}) "
                "MERGE (n)-[:LINKS_TO]->(m)",
                p=rel, m=lnk + ".md",
            )

# ── watchdog ──────────────────────────────────────────────────────────────────
class VaultHandler(FileSystemEventHandler):
    def on_modified(self, ev):
        if not ev.is_directory and ev.src_path.endswith(".md"):
            try:
                index_file(Path(ev.src_path))
            except Exception as e:
                print(f"[ingest] error {ev.src_path}: {e}")
    on_created = on_modified

def run_watcher():
    ensure_collection()
    for md in Path(VAULT_PATH).rglob("*.md"):
        try:
            index_file(md)
        except Exception as e:
            print(f"[ingest] init error {md}: {e}")
    obs = Observer()
    obs.schedule(VaultHandler(), VAULT_PATH, recursive=True)
    obs.start()
    print("[ingest] watcher running")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()

# ── FastAPI embed/rerank endpoints (called by fde-rag Go service) ─────────────
app = FastAPI()

class EmbedReq(BaseModel):
    text: str

class RerankReq(BaseModel):
    query: str
    passages: list[str]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/embed")
def embed(req: EmbedReq):
    return {"vector": _embed_dense(req.text)}

@app.post("/embed-sparse")
def embed_sparse(req: EmbedReq):
    sv = _embed_sparse(req.text)
    return {"indices": sv.indices, "values": sv.values}

@app.post("/rerank")
def rerank(req: RerankReq):
    scores = list(rerank_model.rerank(req.query, req.passages))
    return {"scores": [float(s) for s in scores]}

# ── entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=run_watcher, daemon=True)
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=EMBED_PORT)
