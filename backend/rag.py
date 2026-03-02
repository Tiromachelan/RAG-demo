"""RAG module: indexes the example codebase into ChromaDB and provides retrieval."""
import os
import re
import shutil
from pathlib import Path
from typing import Any

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
import httpx
from openai import OpenAI

CODEBASE_DIR = Path(__file__).parent / "example_codebase"
CHROMA_PATH = Path(__file__).parent.parent / ".chroma_db"
COLLECTION_NAME = "codebase"


def _chunk_file(path: Path) -> list[dict[str, Any]]:
    """Split a Python file into function/class-level chunks with metadata."""
    source = path.read_text(encoding="utf-8")
    chunks = []

    # Chunk by top-level def/class blocks
    block_pattern = re.compile(r"^(def |class )", re.MULTILINE)
    positions = [m.start() for m in block_pattern.finditer(source)] + [len(source)]

    if len(positions) == 1:
        # No functions/classes — treat whole file as one chunk
        chunks.append({"text": source, "file": path.name, "start_line": 1, "end_line": len(source.splitlines())})
        return chunks

    for i in range(len(positions) - 1):
        chunk_text = source[positions[i] : positions[i + 1]].strip()
        if len(chunk_text) < 20:
            continue
        before_text = source[: positions[i]]
        start_line = before_text.count("\n") + 1
        end_line = start_line + chunk_text.count("\n")
        chunks.append({"text": chunk_text, "file": path.name, "start_line": start_line, "end_line": end_line})

    return chunks


def build_index() -> chromadb.Collection:
    """Index the example codebase into ChromaDB. Returns the collection."""
    # Wipe the on-disk DB so there's never a version-mismatch with stale data
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    class OpenAIEmbedFn(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            # Pass a pre-built httpx client to avoid openai<1.52 passing
            # unsupported 'proxies' kwarg to newer httpx versions
            client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                http_client=httpx.Client(),
            )
            response = client.embeddings.create(model="text-embedding-3-small", input=list(input))
            return [item.embedding for item in response.data]

    openai_ef = OpenAIEmbedFn()

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks: list[dict] = []
    for py_file in sorted(CODEBASE_DIR.glob("*.py")):
        all_chunks.extend(_chunk_file(py_file))

    if all_chunks:
        collection.add(
            ids=[f"chunk_{i}" for i in range(len(all_chunks))],
            documents=[c["text"] for c in all_chunks],
            metadatas=[{"file": c["file"], "start_line": c["start_line"], "end_line": c["end_line"]} for c in all_chunks],
        )

    return collection


def retrieve(collection: chromadb.Collection, query: str, k: int = 5) -> list[dict[str, Any]]:
    """Retrieve the top-k most relevant code chunks for the given query."""
    results = collection.query(query_texts=[query], n_results=min(k, collection.count()))
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            {
                "text": doc,
                "file": meta["file"],
                "start_line": meta["start_line"],
                "end_line": meta["end_line"],
                "score": round(1 - dist, 4),  # cosine similarity (higher = more relevant)
            }
        )
    return chunks
