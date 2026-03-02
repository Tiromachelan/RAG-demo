"""RAG module: indexes the example codebase in memory and provides retrieval.

Uses numpy cosine similarity — no ChromaDB required.
Embeddings are fetched from the OpenAI API directly via httpx.
"""
import os
import re
from pathlib import Path
from typing import Any

import httpx
import numpy as np

CODEBASE_DIR = Path(__file__).parent / "example_codebase"
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"


# ── Chunking ───────────────────────────────────────────────────────────────

def _chunk_file(path: Path) -> list[dict[str, Any]]:
    """Split a Python file into function/class-level chunks with metadata."""
    source = path.read_text(encoding="utf-8")
    chunks: list[dict[str, Any]] = []

    block_pattern = re.compile(r"^(def |class )", re.MULTILINE)
    positions = [m.start() for m in block_pattern.finditer(source)] + [len(source)]

    if len(positions) == 1:
        return [{"text": source, "file": path.name, "start_line": 1,
                 "end_line": len(source.splitlines())}]

    for i in range(len(positions) - 1):
        chunk_text = source[positions[i]: positions[i + 1]].strip()
        if len(chunk_text) < 20:
            continue
        start_line = source[: positions[i]].count("\n") + 1
        end_line = start_line + chunk_text.count("\n")
        chunks.append({"text": chunk_text, "file": path.name,
                        "start_line": start_line, "end_line": end_line})
    return chunks


# ── Embeddings via httpx ───────────────────────────────────────────────────

def _embed(texts: list[str]) -> np.ndarray:
    """Fetch embeddings from OpenAI — returns float32 array (n, dim)."""
    api_key = os.environ["OPENAI_API_KEY"]
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            OPENAI_EMBED_URL,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": EMBEDDING_MODEL, "input": texts},
        )
        resp.raise_for_status()
    data = resp.json()["data"]
    # Sort by index to guarantee order
    data.sort(key=lambda d: d["index"])
    return np.array([d["embedding"] for d in data], dtype=np.float32)


# ── In-memory vector store ─────────────────────────────────────────────────

class VectorStore:
    def __init__(self):
        self.chunks: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None  # shape (n, dim)

    def add(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.embeddings = _embed([c["text"] for c in chunks])

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        query_emb = _embed([query])[0]
        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_emb)
        scores = (self.embeddings @ query_emb) / (norms * query_norm + 1e-10)
        top_idx = np.argsort(scores)[::-1][: min(k, len(self.chunks))]
        results = []
        for i in top_idx:
            chunk = dict(self.chunks[i])
            chunk["score"] = round(float(scores[i]), 4)
            results.append(chunk)
        return results


# ── API Public ──────────────────────────────────────────────────

def build_index() -> VectorStore:
    """Chunk and embed all Python files in the example codebase."""
    store = VectorStore()
    all_chunks: list[dict[str, Any]] = []
    for py_file in sorted(CODEBASE_DIR.glob("*.py")):
        all_chunks.extend(_chunk_file(py_file))
    store.add(all_chunks)
    return store


def retrieve(store: VectorStore, query: str, k: int = 5) -> list[dict[str, Any]]:
    return store.retrieve(query, k)
